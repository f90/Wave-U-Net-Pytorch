from torch import nn as nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, conv_type, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv_type = conv_type

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        if self.transpose:
            self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size-1)
        else:
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride)

        if conv_type == "gn":
            assert(n_outputs % NORM_CHANNELS == 0)
            self.norm = nn.GroupNorm(n_outputs // NORM_CHANNELS, n_outputs)
        elif conv_type == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
        # Add you own types of variations here!

    def forward(self, x):
        # Apply the convolution
        if self.conv_type == "gn" or self.conv_type == "bn":
            out = F.relu(self.norm((self.filter(x))))
        else: # Add your own variations here with elifs conditioned on "conv_type" parameter!
            assert(self.conv_type == "normal")
            out = F.leaky_relu(self.filter(x))
        return out

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size

        # Conv
        curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size

    def get_output_size(self, input_size):
        # Transposed
        if self.transpose:
            assert(input_size > 1)
            curr_size = (input_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = input_size

        # Conv
        curr_size = curr_size - self.kernel_size + 1 # o = i + p - k + 1
        assert (curr_size > 0)

        # Strided conv/decimation
        if not self.transpose:
            assert ((curr_size - 1) % self.stride == 0)  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1

        return curr_size