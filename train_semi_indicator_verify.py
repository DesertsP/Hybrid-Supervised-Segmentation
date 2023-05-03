import copy
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from utils import config_parser, build_module, reduce_tensor, init_log, AverageMeter
from datasets.subset import OverSampledSubset
import argparse
import torch.backends.cudnn as cudnn
import shutil
import logging
import time
from utils.eval_semantic_segmentation import intersectionAndUnion
from torch.utils.tensorboard import SummaryWriter
import numpy as np
torch.autograd.set_detect_anomaly(True)
cudnn.benchmark = True


class DDPEngine(object):
    def __init__(self, config, logger, local_rank=-1):
        # parallel
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0

        self.logger = logger
        self.logger.info(config)
        if self.is_distributed:
            self.logger.info('Distributed training enabled.')
            self.world_size = int(os.environ['WORLD_SIZE'])
            device = torch.device('cuda:{}'.format(local_rank))
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", init_method='env://')
        self.config = config
        self.num_epochs = config.train.num_epochs
        # dataset
        train_set = build_module(**config.trainset)
        if 'trainset_unsup' in config:
            unsup_cfg = config.trainset_unsup
        else:
            self.logger.info('Infer trainset_unsup since it is not given in the config.')
            unsup_cfg = {k: v for k, v in config.trainset.items()}
            unsup_cfg['split'] = unsup_cfg['split'].replace("labeled.txt", "unlabeled.txt")
        train_set_unsup = build_module(**unsup_cfg)
        val_set = build_module(**config.valset)
        assert len(train_set) <= len(train_set_unsup), "We assume that the unlabeled subset is larger."
        self.train_set = train_set

        batch_size = config.train.batch_size // self.world_size if self.is_distributed else config.train.batch_size
        batch_size_unsup = config.train.get('batch_size_unsup', config.train.batch_size)
        batch_size_unsup = batch_size_unsup // self.world_size if self.is_distributed else batch_size_unsup
        train_set_ov = OverSampledSubset(train_set, indices=list(range(len(train_set))),
                                         length=len(train_set_unsup) // batch_size_unsup * batch_size)
        self.train_loader = torch.utils.data.DataLoader(
            train_set_ov,
            batch_size=batch_size,
            shuffle=False if self.is_distributed else True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(train_set_ov) if self.is_distributed else None
        )
        self.train_loader_unsup = torch.utils.data.DataLoader(
            train_set_unsup,
            batch_size=batch_size_unsup,
            shuffle=False if self.is_distributed else True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(train_set_unsup) if self.is_distributed else None
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size + batch_size_unsup,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=DistributedSampler(val_set, shuffle=False) if self.is_distributed else None)
        # --- data loaders for indicator learning
        batch_size = config.train.indicator_batch_size // self.world_size \
            if self.is_distributed else config.train.indicator_batch_size
        self.unbiased_loader = torch.utils.data.DataLoader(
            val_set,                                                # ONLY for verification
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            sampler=DistributedSampler(val_set) if self.is_distributed else None
        )
        self.indicator_loader = torch.utils.data.DataLoader(
            train_set_unsup,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            sampler=DistributedSampler(train_set_unsup, shuffle=False) if self.is_distributed else None
        )
        self.unbiased_iter = None
        # ---
        self.logger.info(f'training set size: {len(train_set)}, unsupervised set size: {len(train_set_unsup)}')
        self.logger.info(f'training loader size: {len(self.train_loader)},'
                         f'unsupervised loader size: {len(self.train_loader_unsup)}')

        # model & optim
        model = build_module(**config.model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # optimizers
        param_groups = model.parameter_groups()
        lr_multi = [1.0, 10.0]
        lr = config.train.optimizer.lr
        param_groups = [
            {'params': param_groups[0], 'lr': lr_multi[0] * lr},
            {'params': param_groups[1], 'lr': lr_multi[1] * lr},
        ]
        cfg_optim = config.train.optimizer
        cfg_optim.update({'params': param_groups})
        self.optimizer = build_module(**cfg_optim)
        lr_schd_cfg = copy.deepcopy(config.train.lr_scheduler)
        lr_schd_cfg.update(dict(data_size=len(self.train_loader), optimizer=self.optimizer,
                           num_epochs=config.train.num_epochs))
        self.lr_scheduler = build_module(**lr_schd_cfg)

        if self.is_distributed:
            # model
            model = model.cuda()
            self.model = DistributedDataParallel(model, device_ids=[local_rank],
                                                 output_device=local_rank, find_unused_parameters=False)
        else:
            model = model.cuda()
            self.model = nn.DataParallel(model)

        self.tensorboard = SummaryWriter(config.misc.tensorboard_log_dir) if local_rank <= 0 else None
        # --- indicator optimizer
        cfg_indicator_optim = config.train.indicator_optimizer
        cfg_indicator_optim.update({'weights': model.indicators})
        self.indicator_optimizer = build_module(**cfg_indicator_optim)
        indicator_lr_schd_cfg = copy.deepcopy(config.train.lr_scheduler)
        indicator_lr_schd_cfg.update(dict(data_size=len(self.train_loader), optimizer=self.indicator_optimizer,
                                          num_epochs=config.train.num_epochs))
        self.indicator_lr_scheduler = build_module(**indicator_lr_schd_cfg)
        # ---

    def resume(self, strict=True):
        assert os.path.isfile(self.config.misc.resume), 'resume checkpoint not found.'
        checkpoint = torch.load(self.config.misc.resume, map_location={'cuda:0': 'cpu'})
        self.model.module.load_state_dict(checkpoint['state_dict'], strict=strict)
        self.logger.info("=> load checkpoint from {}".format(self.config.misc.resume))
        start_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return start_epoch

    def train(self):
        start_epoch = 0
        min_loss = 10e9
        best_metric = 0.0
        if self.config.misc.resume:
            ep = self.resume(strict=False)
            if self.is_distributed:
                torch.distributed.barrier()

        for i in range(start_epoch, self.num_epochs):
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(i)
                self.train_loader_unsup.sampler.set_epoch(i)
                self.unbiased_loader.sampler.set_epoch(i)

            loss = self.train_epoch(i)
            if i % self.config.misc.eval_freq != 0:
                continue
            # eval
            val_metric = self.validate(i)

            if self.local_rank <= 0:
                state = {'state_dict': self.model.module.state_dict(), 'epoch': i,
                         'optimizer': self.optimizer.state_dict()}
                if val_metric > best_metric:
                    best_metric = val_metric
                    torch.save(state, os.path.join(self.config.misc.checkpoint_dir, f'best.pth.tar'))
                torch.save(state, os.path.join(self.config.misc.checkpoint_dir, f'latest.pth.tar'))
                self.logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(best_metric * 100))

    def train_epoch(self, epoch):
        loss_meter, data_times, batch_times = AverageMeter(), AverageMeter(), AverageMeter()
        batch_end = time.time()
        for step, (data_one, data_two) in enumerate(zip(self.train_loader, self.train_loader_unsup)):
            self.model.train()
            batch_start = time.time()
            data_times.update(batch_start - batch_end)
            global_step = len(self.train_loader) * epoch + step
            output = self.model(data_one['image'].cuda(), data_one['target'].cuda(),
                                data_two['image'].cuda(), data_two['index'].cuda(), mode=1)

            if isinstance(output, (dict,)):
                loss = 0.0
                for k in output:
                    if k.endswith('_loss'):
                        loss += config.train.loss_coefficient[k] * output[k]
            else:
                loss = output
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            # for indicator lr scheduler
            self.indicator_lr_scheduler.step()

            if self.is_distributed:
                loss = loss.clone().detach()
                dist.all_reduce(loss)
                loss_meter.update(loss.item())
                for k in output:
                    if k.endswith('_loss'):
                        dist.all_reduce(output[k])

            batch_end = time.time()
            batch_times.update(batch_end - batch_start)

            if self.local_rank <= 0 and (global_step + 1) % self.config.misc.log_freq == 0:
                self.logger.info(
                    "Iter [{}/{}]\t"
                    "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                    "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f}) {loss_components}\t"
                    "LR {lr:.6f}\t".format(
                        global_step,
                        self.config.train.num_epochs * len(self.train_loader),
                        data_time=data_times,
                        batch_time=batch_times,
                        loss=loss_meter,
                        loss_components=['{l:.4f}'.format(l=output[k].item()) for k in output if k.endswith('_loss')],
                        lr=self.optimizer.param_groups[-1]['lr'],
                    )
                )
                self.tensorboard.add_scalars("lr", {"lr": self.optimizer.param_groups[0]['lr']}, global_step)
                self.tensorboard.add_scalars("loss_train", {"loss": loss_meter.avg}, global_step)
            # ---
            if global_step >= self.config.train.indicator_update_from_iter and \
                    global_step % self.config.train.indicator_update_freq == 0:
                self.train_indicator()
            # ---
        return loss_meter.avg

    def validate(self, epoch):
        self.model.eval()
        intersection_meter, union_meter = AverageMeter(), AverageMeter()
        num_classes = self.train_set.num_classes
        ignore_label = 255
        for step, data in enumerate(self.val_loader):
            with torch.no_grad():
                image, target = data['image'], data['target']
                output = self.model(image.cuda())['pred']
                output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=True)
                output = torch.argmax(output, dim=1)
            output = output.cpu().numpy()
            target = target.cpu().numpy()
            intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)
            # gather all validation information
            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)
            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        mIoU = np.nanmean(iou_class)
        if self.local_rank <= 0:
            for i, iou in enumerate(iou_class):
                self.logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
            self.logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))
            self.tensorboard.add_scalars("validation", {"mIoU": mIoU}, epoch)
        return mIoU

    def train_indicator(self):
        torch.cuda.empty_cache()
        self.model.eval()
        self.indicator_optimizer.zero_grad()
        start_time = time.time()
        accumulate_iters = len(self.unbiased_loader)
        for step, data in enumerate(self.unbiased_loader):
            # manually sync gradients among gpus via the sync flag
            if step == 0:
                self.model(data['image'].cuda(), data['target'].cuda(), mode=2, zero_grad=True, sync=False,
                           accumulate_iters=accumulate_iters)
            elif step < len(self.unbiased_loader) - 1:
                self.model(data['image'].cuda(), data['target'].cuda(), mode=2, zero_grad=False, sync=False,
                           accumulate_iters=accumulate_iters)
            elif step == len(self.unbiased_loader) - 1:
                self.model(data['image'].cuda(), data['target'].cuda(), mode=2, zero_grad=False, sync=True,
                           accumulate_iters=accumulate_iters)
            else:
                raise ValueError
        torch.cuda.empty_cache()
        unbiased_elapse = time.time() - start_time
        for step, data in enumerate(self.indicator_loader):
            # manually sync indicator gradients among gpus via the sync flag
            self.model(data['image'].cuda(), target=None,
                       index=data['index'].cuda(), mode=3, sync=True)
        self.indicator_optimizer.step()
        indicators = self.indicator_optimizer.weights.detach().cpu().numpy()
        total_elapse = time.time() - start_time
        if self.local_rank <= 0:
            self.logger.info(
                "Indicator\t"
                "Unbiased Time {unbiased_elapse:.2f}\t"
                "Indicator Time {indicator_elapse:.2f}\t"
                "Total Time {total_elapse:.2f}\t"
                "Mean {mean_val:.2f}\t"
                "Max {max_val:.2f}\t"
                "Min {min_val:.2f}\t"
                "LR {lr:.6f}".format(
                    unbiased_elapse=unbiased_elapse,
                    indicator_elapse=total_elapse - unbiased_elapse,
                    total_elapse=total_elapse,
                    mean_val=indicators.mean(),
                    min_val=indicators.min(),
                    max_val=indicators.max(),
                    lr=self.indicator_optimizer.param_groups[-1]['lr']
                ))

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--run', type=str, default='', help="running ID")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = config_parser.parse_config(args.config, args.opts)
    output_dir = os.path.join(os.path.dirname(args.config), args.run)
    if args.local_rank <= 0:
        backup_dir = os.path.join(output_dir, 'backups')
        config.misc.tensorboard_log_dir = os.path.join(output_dir, 'tblogs')
        config.misc.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(config.misc.checkpoint_dir, exist_ok=True)
        os.makedirs(config.misc.tensorboard_log_dir, exist_ok=True)
        shutil.copy(__file__, backup_dir)
        shutil.copy(args.config, backup_dir)
        shutil.copy(os.path.join(*config.model.name.rsplit('.')[:-1]) + '.py', backup_dir)
        shutil.copy(os.path.join(*config.trainset.name.rsplit('.')[:-1]) + '.py', backup_dir)

    if args.seed > 0:
        print('Seeding with', args.seed)
        torch.manual_seed(args.seed)
    log_file = os.path.join(output_dir, str(datetime.datetime.now()) + '.log')
    logger = init_log('global', logging.INFO, filename=log_file if args.local_rank <= 0 else None)
    logger.propagate = False
    engine = DDPEngine(config, logger, local_rank=args.local_rank)
    engine.train()
