import torch
# while True: x=torch.randn(1024, 1024).cuda().inverse()
import functorch as ft

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from functorch import vmap, grad, hessian, make_functional

from functorch import make_functional, vmap, grad
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.backends.cudnn as cudnn

# torch.autograd.set_detect_anomaly(True)
cudnn.benchmark = True

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.classifier = nn.Conv2d(256, 21, 1)

    def per_sample_grad(self, x, y):
        cls_func, params = make_functional(self.classifier)

        def compute_loss(params, x, y):
            return F.cross_entropy(cls_func(params, x), y)

        per_sample_grad_cls_func = vmap(grad(compute_loss), in_dims=(None, 0, 0))
        return per_sample_grad_cls_func(params, x, y)

    def forward(self, x, y, mode=0):
        if mode > 0:
            return self.per_sample_grad(x, y)
        out = self.classifier(x)
        loss = F.cross_entropy(out, y)
        return loss


def main(local_rank):
    # dist.init_process_group("nccl")
    # world_size = int(os.environ['WORLD_SIZE'])
    # device = torch.device('cuda:{}'.format(local_rank))
    # torch.cuda.set_device(device)

    cls = nn.Conv2d(256, 21, kernel_size=1).cuda()
    cls_func, params = make_functional(cls, disable_autograd_tracking=True)
    x = torch.randn(1024, 1, 256, 1, 1).cuda()
    y = torch.randint(0, 21, (1024, 1)).cuda()

    def compute_loss(params, inputs, targets):
        return F.cross_entropy(cls_func(params, inputs).squeeze(-1).squeeze(-1), targets, reduction='mean')

    vmap(grad(compute_loss), in_dims=(None, 0, 0))(params, x, y)
    print('all done')
    # model = Model()
    # model_dist = DistributedDataParallel(model.cuda())
    #
    #
    # torch.testing.assert_close(out, foo(x))
    #
    # print(rank, "per_sample_grads:", per_sample_grads.shape, per_sample_grads)
    # torch.testing.assert_close(per_sample_grads, expected)


if __name__ == "__main__":
    main(1)
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()
    #
    # local_rank = args.local_rank
    # main(local_rank)
    # #
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29501"
    # ws = 2
    # mp.spawn(main, nprocs=ws, args=(ws,))
