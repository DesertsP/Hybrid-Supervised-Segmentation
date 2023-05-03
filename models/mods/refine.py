import torch
import torch.nn.functional as F
import torch.nn as nn


class RefineModule(nn.Module):
    def __init__(self, num_iter=10, dilations=(1,), kernel_size=3):
        super().__init__()
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.num_iterations = num_iter

    def forward(self, feats, mask):
        aff = self.local_affinity(feats)  # (B, 1, k*k, H, W)
        for _ in range(self.num_iterations):
            v = self.neighbors(mask, remove_self=True)  # (B, C, k*k, H, W)
            mask = (v * aff).sum(2)
        return mask

    def neighbors(self, x, remove_self=True):
        xs = [self.sliding_window(x, self.kernel_size, d, remove_self) for d in self.dilations]
        return torch.cat(xs, 2)

    def sliding_window(self, x, kernel_size, dilation, remove_self=False):
        batch, channels, height, width = x.size()
        # when stride=1, padding = (kernel-1)*dilation/2
        padding = (kernel_size - 1) * dilation // 2
        x = F.pad(x, [padding] * 4, mode='replicate')
        x = F.unfold(x, kernel_size=kernel_size, dilation=dilation)
        x = x.view(batch, channels, kernel_size ** 2, height, width)
        if remove_self:
            # remove self-loops
            selected_idx = list(range(kernel_size**2//2)) + list(range(kernel_size**2//2+1, kernel_size**2))
            x = x[:, :, selected_idx, :, :]
        return x

    def local_affinity(self, x):
        q = x.unsqueeze(2)  # (B, C, 1, H, W)
        k = self.neighbors(x, remove_self=True)
        k2 = self.neighbors(x, remove_self=False)
        aff = -torch.abs(k - q) / (1e-8 + 0.1 * k2.std(2, keepdim=True))     # (B, C, k*k, H, W)
        aff = aff.mean(1, keepdim=True)
        # aff = (k * q).sum(1, keepdim=True)
        # (B, 1, k*k, H, W)
        aff = F.softmax(aff, 2)
        return aff


class RefineModule2(nn.Module):
    def __init__(self, num_iter=10, dilation=1, kernel_size=7):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num_iterations = num_iter

    def forward(self, feats, mask):
        aff = self.local_affinity(feats)  # (B, 1, k*k, H, W)
        for _ in range(self.num_iterations):
            v = self.sliding_window(mask, remove_self=True)  # (B, C, k*k, H, W)
            mask = (v * aff).sum(2)
        return mask

    def sliding_window(self, x, remove_self=False):
        batch, channels, height, width = x.size()
        # when stride=1, padding = (kernel-1)*dilation/2
        kernel_size = self.kernel_size
        padding = (self.kernel_size - 1) * self.dilation // 2
        x = F.pad(x, [padding] * 4, mode='replicate')
        x = F.unfold(x, kernel_size=kernel_size, dilation=self.dilation)
        x = x.view(batch, channels, kernel_size ** 2, height, width)
        if remove_self:
            # remove self-loops
            selected_idx = list(range(kernel_size**2//2)) + list(range(kernel_size**2//2+1, kernel_size**2))
            x = x[:, :, selected_idx, :, :]
        return x

    def local_affinity(self, x):
        q = x.unsqueeze(2)  # (B, C, 1, H, W)
        k = self.sliding_window(x, remove_self=True)
        # k2 = self.neighbors(x, remove_self=False)
        aff = -torch.abs(k - q) / (1e-8 + 0.1 * k.std(2, keepdim=True))     # (B, C, k*k, H, W)
        aff = aff.mean(1, keepdim=True)
        # aff = (k * q).sum(1, keepdim=True)
        # (B, 1, k*k, H, W)
        aff = F.softmax(aff, 2)
        return aff


if __name__ == '__main__':
    r = RefineModule(dilations=(1,2), num_iter=10).cuda()
    p = PAMR(dilations=(1,), num_iter=10).cuda()
    torch.manual_seed(10)
    x = torch.randn(1, 3, 512, 512).cuda()
    f = torch.randn(1, 200, 512, 512).cuda()
    with torch.no_grad():
        r(x, f)
        # print(torch.all(r(x, f) == p(x, f)))
    # r(x, f)
    import time
    time.sleep(20)