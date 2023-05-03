import argparse
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime
import torch.nn as nn
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

import yaml
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from utils.dist_helper import setup_distributed
from utils import AverageMeter, build_module, init_log, parse_config, set_random_seed
from utils.eval_semantic_segmentation import intersectionAndUnion
from utils.lr_helper import get_scheduler
parser = argparse.ArgumentParser(description="Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=9999, type=int)
logger = init_log("global", logging.INFO)
logger.propagate = 0


def main():
    global args, cfg
    args = parser.parse_args()
    cfg = parse_config(args.config)

    output_path = os.path.dirname(args.config)
    ckp_path = os.path.join(output_path, 'checkpoints')

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(output_path, "log/events_seg/" + current_time)
        )
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(ckp_path) and rank == 0:
        os.makedirs(ckp_path)

    # Create network.
    model = build_module(**cfg.model)
    modules_back = [model.backbone]
    modules_head = [model.decoder, model.classifier]

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()

    # dataset
    train_set = build_module(**cfg.trainset)
    val_set = build_module(**cfg.valset)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=DistributedSampler(train_set))
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=False,
        drop_last=False,
        sampler=DistributedSampler(val_set))

    # Optimizer and lr decay scheduler

    params_list = []
    lr = cfg.train.optimizer.lr
    for module in modules_back:
        params_list.append(dict(params=module.parameters(), lr=lr))
    for module in modules_head:
        params_list.append(dict(params=module.parameters(), lr=lr * 10))

    cfg_optim = cfg.train.optimizer
    cfg_optim.update({'params': params_list})
    optimizer = build_module(**cfg_optim)

    best_prec = 0
    last_epoch = 0

    optimizer_old = build_module(**cfg_optim)
    lr_scheduler = get_scheduler(
        cfg.train, len(train_loader), optimizer_old, start_epoch=last_epoch
    )

    # Start to train model
    for epoch in range(last_epoch, cfg.train.num_epochs):
        # Training
        train(
            model,
            optimizer,
            lr_scheduler,
            criterion,
            train_loader,
            epoch,
            tb_logger,
        )

        # Validation and store checkpoint
        prec = validate(model, val_loader, epoch)

        if rank == 0:
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_miou": best_prec,
            }

            if prec > best_prec:
                best_prec = prec
                state["best_miou"] = prec
                torch.save(
                    state, osp.join(ckp_path, "ckpt_best.pth")
                )

            torch.save(state, osp.join(ckp_path, "ckpt.pth"))

            logger.info(
                "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                    best_prec * 100
                )
            )
            tb_logger.add_scalar("mIoU val", prec, epoch)


def train(
    model,
    optimizer,
    lr_scheduler,
    criterion,
    data_loader,
    epoch,
    tb_logger,
):
    model.train()

    data_loader.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)

    rank, world_size = dist.get_rank(), dist.get_world_size()

    losses = AverageMeter()
    data_times = AverageMeter()
    batch_times = AverageMeter()
    learning_rates = AverageMeter()

    batch_end = time.time()
    for step in range(len(data_loader)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(data_loader) + step

        image, label = data_loader_iter.next()
        batch_size, h, w = label.size()
        image, label = image.cuda(), label.cuda()
        outs = model(image)
        pred = outs["pred"]
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        # gather all loss from different gpus
        reduced_loss = loss.clone().detach()
        dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)


        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "LR {lr:.5f}\t".format(
                    i_iter,
                    cfg.train.num_epochs * len(data_loader),
                    data_time=data_times,
                    batch_time=batch_times,
                    loss=losses,
                    lr=optimizer.param_groups[-1]['lr'],
                )
            )
            tb_logger.add_scalar("Loss", losses.avg, i_iter)


def validate(
    model,
    data_loader,
    epoch,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = 21, 255
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        batch_size, h, w = labels.shape

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

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
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


if __name__ == "__main__":
    main()
