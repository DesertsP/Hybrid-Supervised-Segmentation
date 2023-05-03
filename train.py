import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from utils.meters import Progbar
from utils import config_parser
from utils.log_utils import create_log_dir
import argparse
import torch.backends.cudnn as cudnn
import logging
from utils.visualize import mask_rgb, make_grid, denorm
from utils.utils import reduce_tensor
import shutil
cudnn.benchmark = True
logging.basicConfig(level=logging.INFO)
from sklearn.metrics import average_precision_score
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)
from datasets import create_dataset
from models import get_model
from utils.eval_semantic_segmentation import eval_custom, calc_semantic_segmentation_confusion


class TrainingEngine(object):
    def __init__(self, config, local_rank=-1):
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        if self.is_distributed:
            logging.info('Distributed training enabled.')
        self.config = config
        self.num_epochs = config.train.num_epochs
        model = get_model(**config.model)

        # optimizer
        lr = config.train.lr
        wd = config.train.weight_decay
        if config.model.backbone == 'resnet38':
            lr_multi = [1.0, 2.0, 10.0, 20.0]
        else:
            lr_multi = [1.0, 1.0, 10.0, 10.0]
        param_groups = model.parameter_groups()
        param_groups = [
            {'params': param_groups[0], 'lr': lr_multi[0] * lr, 'weight_decay': wd},
            {'params': param_groups[1], 'lr': lr_multi[1] * lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': lr_multi[2] * lr, 'weight_decay': wd},
            {'params': param_groups[3], 'lr': lr_multi[3] * lr, 'weight_decay': 0},
        ]
        # optimizer
        self.optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9,
                                         weight_decay=self.config.train.weight_decay, nesterov=True)

        # parallel
        if self.is_distributed:
            self.device = torch.device('cuda:{}'.format(local_rank))
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend="nccl", init_method='env://')
            # model
            model = model.to(self.device)

            self.model = DistributedDataParallel(model, device_ids=[local_rank],
                                                 output_device=local_rank,
                                                 find_unused_parameters=True)
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
            model = model.to(self.device)
            self.model = nn.DataParallel(model)

        # dataset
        train_set = create_dataset(**config.trainset)
        val_set = create_dataset(**config.valset)

        self.sampler = DistributedSampler(train_set) if self.is_distributed else None
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.train.batch_size,
            shuffle=False if self.is_distributed else True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self.sampler)
        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config.train.batch_size,
            shuffle=False if self.is_distributed else True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(val_set) if self.is_distributed else None)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                       (self.num_epochs - self.config.train.pretrain_epochs) * len(self.train_loader),
                                                                       eta_min=0.0)
        self.tensorboard = SummaryWriter(config.misc.tensorboard_log_dir) if local_rank <= 0 else None
        self.visualize_samples = None
        if os.path.isfile(config.misc.visualize_samples):
            logging.info(f'loading visualize samples from {config.misc.visualize_samples}')
            self.visualize_samples = torch.load(config.misc.visualize_samples)

    def resume(self):
        assert os.path.isfile(self.config.misc.resume), 'resume checkpoint not found.'
        checkpoint = torch.load(self.config.misc.resume, map_location={'cuda:0': 'cpu'})
        self.model.module.load_state_dict(checkpoint['state_dict'], strict=True)
        logging.info("=> load checkpoint from {}".format(self.config.misc.resume))
        start_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return start_epoch

    def train(self):
        start_epoch = 0
        max_val = 0
        if self.config.misc.resume:
            start_epoch = self.resume()
            start_epoch += 1
            if self.is_distributed:
                torch.distributed.barrier()
        for i in range(start_epoch, self.num_epochs):
            if self.is_distributed:
                self.sampler.set_epoch(i)
            loss = self.train_one_epoch(i)
            val_metric = self.validation(i)
            if self.local_rank <= 0:
                best_ckp_path = os.path.join(self.config.misc.checkpoint_dir, 'best.pth.tar')
                current_ckp_path = os.path.join(self.config.misc.checkpoint_dir, f'epoch{i + 1}.pth.tar')
                # last_ckp_path = os.path.join(config.misc.checkpoint_dir, f'latest.pth.tar')
                if self.config.misc.visualize:
                    self.visualize(i)
                if i >= self.config.misc.save_from_epoch:
                    torch.save({'state_dict': self.model.module.state_dict(), 'epoch': i+1,
                                'optimizer': self.optimizer.state_dict()}, current_ckp_path)

                if val_metric >= max_val:
                    max_val = val_metric

                    torch.save({'state_dict': self.model.module.state_dict(), 'epoch': i+1,
                                'optimizer': self.optimizer.state_dict()}, best_ckp_path)

    def train_one_epoch(self, epoch):
        progbar = Progbar(len(self.train_loader), prefix="train[{}/{}]".format(epoch + 1, self.num_epochs),
                          verbose=self.config.misc.verbose) if self.local_rank <= 0 else None

        for step, data in enumerate(self.train_loader):
            self.model.train()
            global_step = len(self.train_loader) * epoch + step
            img, label = data['img'], data['label']
            if self.local_rank <= 0 and self.visualize_samples is None:
                self.visualize_samples = (img.clone(), label.clone())
            img_raw = denorm(img.clone()).to(self.device)
            img = img.to(self.device)
            label = label.to(self.device)
            output = self.model(img, img_raw, label)

            loss = 0
            if isinstance(output, (dict,)):
                for k in output:
                    if k.endswith('_loss') and not k.startswith('seg'):
                        loss += config.train.loss_coeff[k] * output[k].mean()
                if epoch > self.config.train.pretrain_epochs:
                    loss += config.train.loss_coeff['seg_loss'] * output['seg_loss'].mean()
            else:
                loss = output

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch >= self.config.train.pretrain_epochs:
                self.lr_scheduler.step()

            reduced_loss = reduce_tensor(loss) if self.is_distributed else loss

            if self.local_rank <= 0:
                progbar.update(step + 1,
                               values=[("loss", reduced_loss.item()), ])

            if self.local_rank <= 0 and (global_step + 1) % self.config.misc.log_freq == 0:
                self.tensorboard.add_scalars("lr", {"lr": self.optimizer.param_groups[0]['lr']}, global_step)
                self.tensorboard.add_scalars("loss_train", {"loss": progbar['loss'], }, global_step)

                if config.misc.debug:
                    logging.info("DEBUG MODE ON: break")
                    break
        return progbar['loss'] if self.local_rank <= 0 else 0

    def visualize(self, step):
        """
        :param step:
        :return:
        """
        assert self.visualize_samples is not None
        img, label = self.visualize_samples
        self.model.eval()
        with torch.no_grad():
            img_raw = denorm(img.clone())
            img = img.to(self.device)
            output = self.model(img, img_raw.to(self.device), label.to(self.device))

            vis_list = [img_raw, mask_rgb(output['mask'], img_raw),
                        mask_rgb(output['refined'], img_raw)]
            pseudo_mask = mask_rgb(output["pseudo"], img_raw)
            ambiguous = 1 - pseudo_mask.sum(1, keepdim=True).cpu()
            pseudo_mask = ambiguous * img_raw + (1 - ambiguous) * pseudo_mask
            vis_list.append(pseudo_mask)

            vis_elements = torch.cat(vis_list, dim=-1)
            summary_grid = make_grid(vis_elements, label)
            self.tensorboard.add_image('vis', summary_grid, step)

    def validation(self, epoch):
        """
        DDP is not tested yet.
        """
        progbar = Progbar(len(self.val_loader), prefix="valid[{}/{}]".format(epoch + 1, self.num_epochs),
                          verbose=self.config.misc.verbose) if self.local_rank <= 0 else None
        self.model.eval()

        cls_predictions = []
        cls_labels = []
        seg_predictions = []
        seg_labels = []
        with torch.no_grad():
            for step, data in enumerate(self.val_loader):
                img, label, seg_label = data['img'], data['label'], data['mask']
                img = img.to(self.device)
                label = label.to(self.device)
                img_raw = denorm(img.clone())
                output = self.model(img, img_raw.to(self.device), label)

                loss = 0
                if isinstance(output, (dict,)):
                    for k in output:
                        if k.endswith('_loss'):
                            loss = loss + config.train.loss_coeff[k] * output[k].mean()
                else:
                    loss = output
                reduced_loss = reduce_tensor(loss) if self.is_distributed else loss

                # ----
                cls_sigmoid = torch.sigmoid(output['cls']).cpu()
                cls_predictions.append(cls_sigmoid)
                cls_labels.append(label.cpu())
                masks_pred = output['mask']
                # amend bg
                # masks_pred[0, :, :] = torch.pow(masks_pred[0], self.config.test.bg_pow)
                seg_pred = masks_pred.argmax(dim=1)
                for batch_idx in range(seg_pred.shape[0]):
                    seg_predictions.append(seg_pred[batch_idx].cpu().numpy())
                    seg_labels.append(seg_label[batch_idx].cpu().numpy())
                # ----

                if self.local_rank <= 0:
                    progbar.update(step + 1,
                                   values=[("loss", reduced_loss.item()), ])

        # classification
        targets_stacked = torch.vstack(cls_labels).numpy()
        preds_stacked = torch.vstack(cls_predictions).numpy()
        aps = average_precision_score(targets_stacked, preds_stacked, average=None)

        # skip BG AP
        for name, i in zip(self.train_loader.dataset.CLASSES, aps):
            print(f'{name: <15}\t {i:.4f}')
        print('mean:', np.mean(aps))

        # segmentation
        confusion = calc_semantic_segmentation_confusion(seg_predictions, seg_labels)
        if self.is_distributed:
            confusion = torch.from_numpy(confusion).cuda().float()
            confusion_reduced = reduce_tensor(confusion)
            confusion = confusion_reduced.cpu().numpy()

        metrics = eval_custom(confusion, class_names=self.train_loader.dataset.CLASSES)
        if self.local_rank <= 0:
            self.tensorboard.add_scalars("metrics", {"miou": metrics['miou'], "aps": np.mean(aps)}, epoch)
            self.tensorboard.add_scalars("loss_val",
                                         {"loss": progbar['loss'], },
                                         epoch+1)
        return metrics['miou']


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=128)
    parser.add_argument('--run', type=str, default='', help="running ID")
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
    if args.local_rank <= 0:
        log_dir, tensorboard_log_dir, checkpoint_dir = create_log_dir(config.misc.log_dir,
                                                                      os.path.basename(args.config).split('.')[0],
                                                                      run_name=args.run)
        config.misc.log_dir = log_dir
        config.misc.tensorboard_log_dir = tensorboard_log_dir
        config.misc.checkpoint_dir = checkpoint_dir
        print(config)
        # backup models and other scripts
        if os.path.exists(os.path.join(log_dir, 'models')):
            shutil.rmtree(os.path.join(log_dir, 'models'))
        shutil.copytree('models', os.path.join(log_dir, 'models'))
        shutil.copy(args.config, log_dir)
        shutil.copy(__file__, log_dir)

    if args.seed > 0:
        print('Seeding with', args.seed)
        torch.manual_seed(args.seed)


    engine = TrainingEngine(config=config, local_rank=args.local_rank)
    engine.train()