import torch
import torch.distributed as dist
import importlib
from functools import partial
import logging
import os
import random
import numpy as np


def get_module(name, *args, **kwargs):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return partial(getattr(module, class_name), **kwargs)


def build_module(name, *args, **kwargs):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(*args, **kwargs)


def make_infinite(loader):
    while True:
        yield from loader


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


def adjust_learning_rate(optimizer, base_lr, num_warmup_steps, num_decay_steps, cur_step, decay_rate=0.5):
    """
    param_groups = [
            {'params': param_groups[0], 'lr': self.lr, 'weight_decay': self.config.train.weight_decay},
            {'params': param_groups[1], 'lr': 2.0 * self.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10.0 * self.lr, 'weight_decay': self.config.train.weight_decay},
            {'params': param_groups[3], 'lr': 20.0 * self.lr, 'weight_decay': 0},
        ]
    :param optimizer:
    :param base_lr:
    :param num_warmup_steps:
    :param num_decay_steps:
    :param cur_step:
    :param decay_rate:
    :return:
    """
    if cur_step <= num_warmup_steps:
        lr = (cur_step) * base_lr / num_warmup_steps
    elif cur_step <= num_warmup_steps + num_decay_steps:
        lr = base_lr * decay_rate ** ((cur_step - num_warmup_steps) / num_decay_steps)
    else:
        lr = base_lr * decay_rate
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = 10. * lr
    elif len(optimizer.param_groups) == 4:
        optimizer.param_groups[1]['lr'] = 2. * lr
        optimizer.param_groups[2]['lr'] = 10. * lr
        optimizer.param_groups[3]['lr'] = 20. * lr
    return lr


def init_log(name, level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)

    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def load_state(path, model, optimizer=None, key="state_dict"):
    rank = dist.get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    if rank == 0:
                        print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )

        for k in ignore_keys:
            checkpoint.pop(k)

        model.load_state_dict(state_dict, strict=False)

        if rank == 0:
            ckpt_keys = set(state_dict.keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_miou"]
            last_iter = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                        path, last_iter
                    )
                )
            return best_metric, last_iter
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))

def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    pass
