import torch
from typing import Tuple, List, Union, Dict, cast, Optional
from torch.distributions import Bernoulli
import torch.nn.functional as F
import warnings
from functools import wraps
from datasets.augmentation import generate_cutout_mask, generate_class_mask
import einops

@torch.no_grad()
def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2) # (N, P, 2) -> (N, P, 1, 2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def weighted_gather(x, p):
    x = einops.rearrange(x, 'b k h w -> b (h w) k')
    p = einops.rearrange(p, 'b c h w -> b c (h w)')
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-6)
    mean_ = p.matmul(x)     # (b, c, n)(b, n, k)->(b, c, k)
    return mean_


def generate_mix_masks(image, target=None, mode="cutout"):
    batch_size, _, im_h, im_w = image.shape
    device = image.device
    mask_list = []
    for i in range(batch_size):
        if mode == "cutmix":
            mask_list.append(generate_cutout_mask([im_h, im_w]))
        elif mode == "classmix":
            mask_list.append(generate_class_mask(target[i]))
        else:
            raise ValueError(f'Unexpected mode: {mode}')
    masks = torch.stack(mask_list, dim=0)       # (b, h, w)
    masks = masks.to(device)
    return masks


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def label_onehot(tensor, num_classes, ignore_index=255, channel_last=False):
    _tensor = tensor.clone()
    mask = tensor == ignore_index
    _tensor[mask] = 0
    output = F.one_hot(_tensor, num_classes)
    output[mask] = 0
    if channel_last:
        return output
    else:
        return output.permute(0, 3, 1, 2)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def update_model_moving_average(ema_decay, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        ma_params.data = ema_decay * ma_params.data + (1 - ema_decay) * current_params.data


def update_tensor_moving_average(ema_decay, old, new):
    if old is None or torch.sum(torch.abs(old)) == 0:
        return new
    return ema_decay * old + (1 - ema_decay) * new


def copy_parameters(tgt, src):
    for t_p, o_p in zip(tgt.parameters(), src.parameters()):
        t_p.data.copy_(o_p.data)
    return tgt


class EMA(object):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def resize_as(x, y, mode='bilinear', align_corners=True):
    """
    rescale x to the same size as y
    """
    return F.interpolate(x, size=y.shape[-2:], mode=mode, align_corners=align_corners)


def resize_to(x, size):
    """
    rescale x to the same size as y
    """
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


def _adapted_sampling(
        shape: Union[Tuple, torch.Size],
        dist: torch.distributions.Distribution,
        same_on_batch=False
) -> torch.Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    if same_on_batch:
        return dist.sample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
    else:
        return dist.sample(shape)


def random_prob_generator(
        batch_size: int, p: float = 0.5, same_on_batch: bool = False,
        device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    r"""Generate random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability to generate an 1-d binary mask. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        torch.Tensor: parameters to be passed for transformation.
            - probs (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    if not isinstance(p, (int, float)) or p > 1 or p < 0:
        raise TypeError(f"The probability should be a float number within [0, 1]. Got {type(p)}.")

    _bernoulli = Bernoulli(torch.tensor(float(p), device=device, dtype=dtype))
    probs_mask: torch.Tensor = _adapted_sampling((batch_size,), _bernoulli, same_on_batch).bool()

    return probs_mask


def hflip(input: torch.Tensor) -> torch.Tensor:
    r"""Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Returns:
        torch.Tensor: The horizontally flipped image tensor

    """
    w = input.shape[-1]
    return input[..., torch.arange(w - 1, -1, -1, device=input.device)]


def random_hflip(input: torch.Tensor, p=0.5, return_p=False):
    batch_size, _, h, w = input.size()
    output = input.clone()
    if torch.is_tensor(p):
        to_apply = p
    else:
        to_apply = random_prob_generator(batch_size, p=p)
    output[to_apply] = hflip(input[to_apply])
    if return_p:
        return output, to_apply
    return output


if __name__ == '__main__':
    # print(random_prob_generator(10))
    x = (torch.randn(2, 3, 5, 5))
    print(x[0])
    x_fp, p = random_hflip(x, return_p=True)
    print(x_fp[0])
    x_rec = random_hflip(x_fp, p=p)
    print(x_rec[0])
