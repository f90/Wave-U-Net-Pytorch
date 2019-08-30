import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sinc(x):
    x[np.abs(x) < 1e-20] = 1e-20
    return np.sin(np.pi * x) / (np.pi * x)

def build_sinc_filter(kernel_size, cutoff):
    # FOLLOWING https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch16.pdf
    # Sinc lowpass filter
    # Build sinc kernel
    assert(kernel_size % 2 == 1)
    M = kernel_size - 1
    filter = np.zeros(kernel_size, dtype=np.float32)
    for i in range(kernel_size):
        if i == M//2:
            filter[i] = 2 * np.pi * cutoff
        else:
            filter[i] = (np.sin(2 * np.pi * cutoff * (i - M//2)) / (i - M//2)) * \
                    (0.42 - 0.5 * np.cos((2 * np.pi * i) / M) + 0.08 * np.cos(4 * np.pi * M))

    filter = filter / np.sum(filter)
    return filter

class Resample1d(nn.Module):
    def __init__(self, channels, kernel_size, stride, transpose=False, padding="reflect", trainable=False):
        super(Resample1d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.transpose = transpose
        self.channels = channels

        cutoff = 0.5 / stride

        assert(kernel_size > 2)
        assert ((kernel_size - 1) % 2 == 0)

        assert(padding == "reflect" or padding == "valid")

        filter = build_sinc_filter(kernel_size, cutoff)

        self.filter = torch.nn.Parameter(torch.from_numpy(np.repeat(np.reshape(filter, [1, 1, kernel_size]), channels, axis=0)), requires_grad=trainable)

    def forward(self, x):
        # Pad here if not using transposed conv
        input_size = x.shape[2]
        if self.padding != "valid":
            num_pad = (self.kernel_size-1)//2
            out = F.pad(x, (num_pad, num_pad), mode=self.padding)
        else:
            out = x

        # Lowpass filter (+ 0 insertion if transposed)
        if self.transpose:
            expected_steps = ((input_size - 1) * self.stride + 1)
            if self.padding == "valid":
                expected_steps = expected_steps - self.kernel_size + 1

            out = F.conv_transpose1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)
            diff_steps = out.shape[2] - expected_steps
            if diff_steps > 0:
                assert(diff_steps % 2 == 0)
                out = out[:,:,diff_steps//2:-diff_steps//2]
        else:
            assert(input_size % self.stride == 1)
            out = F.conv1d(out, self.filter, stride=self.stride, padding=0, groups=self.channels)

        return out

    def get_output_size(self, input_size):
        assert(input_size > 1)
        if self.transpose:
            if self.padding == "valid":
                return ((input_size - 1) * self.stride + 1) - self.kernel_size + 1
            else:
                return ((input_size - 1) * self.stride + 1)
        else:
            assert(input_size % self.stride == 1) # Want to take first and last sample
            if self.padding == "valid":
                return input_size - self.kernel_size + 1
            else:
                return input_size

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        if self.padding == "valid":
            curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size


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