import torch
import torch.nn as nn
import torch.nn.functional as F

class Crop1d(nn.Module):
    def __init__(self, mode="both"):
        super(Crop1d, self).__init__()
        self.mode = mode

    def forward(self, x, target):
        if x is None:
            return None
        if target is None:
            return x

        target_shape = target.shape
        diff = x.shape[-1] - target_shape[-1]
        if self.mode == "both":
            assert(diff % 2 == 0)
            crop = diff // 2
        else:
            crop = diff

        if crop == 0:
            return x
        if crop < 0:
            raise ArithmeticError

        if self.mode == "front":
            return x[:, :, crop:].contiguous()
        elif self.mode == "back":
            return x[:, :, :-crop].contiguous()
        else:
            assert(self.mode == "both")
            return x[:, :, crop:-crop].contiguous()

def interleave(x, y):
    comb = torch.cat([x[:, :, :-1].unsqueeze(2), y], dim=2)  # (N, C, 2, W)
    comb = comb.transpose(2, 3).contiguous()  # (N, C, W, 2)
    return torch.cat([comb.view(comb.shape[0], comb.shape[1], -1), x[:, :, -1:]], dim=2)  # (N, C, W*2+1)

def zero_interleave(x, stride):
    zero_shape = [x.shape[0], x.shape[1], stride-1, x.shape[2]-1]
    zeros = torch.zeros(zero_shape).to(x.device).detach()
    return interleave(x, zeros)

def interpolate(x):
    interp = F.avg_pool1d(x, 2, stride=1, padding=0)
    return interleave(x, interp)

def duplicate(x):
    return interleave(x, x[:,:,:-1])