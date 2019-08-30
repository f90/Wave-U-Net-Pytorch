import torch
import torch.nn as nn
import torch.nn.functional as F

from waveunet_utils import Crop1d, zero_interleave, Resample1d


class ConvLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, norm, transpose=False):
        super(ConvLayer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size

        if self.transpose:
            self.filter = nn.ConvTranspose1d(n_inputs, n_outputs, self.kernel_size, stride, padding=kernel_size-1, bias=True) #TODO Direct transposed conv produces artifacts?
        else:
            self.filter = nn.Conv1d(n_inputs, n_outputs, self.kernel_size, stride, bias=True)

        if norm == "gn":
            assert(n_outputs % 8 == 0)
            self.norm = nn.GroupNorm(n_outputs // 8, n_outputs)
        elif norm == "bn":
            self.norm = nn.BatchNorm1d(n_outputs, momentum=0.01)
        else:
            self.norm = None

        self.identity_crop = Crop1d("both")

    def forward(self, x, conditional=None):
        # BEGINNING OF RESIDUAL BLOCK COMPUTATION
        # If we have conditional input, concatenate it so the convolution operates on the input and conditional
        if conditional is None:
            res_input = x
        else:
            res_input = torch.cat([x, conditional], dim=1)

        # Apply the convolution
        if self.norm == None:
            out = F.leaky_relu(self.filter(res_input))
        else:
            out = F.relu(self.norm((self.filter(res_input))))

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

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, norm, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, norm, transpose=True)

        # Crop operation for the shortcut connection that might have more samples!
        self.crop = Crop1d("both")

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, norm)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, norm) for _ in range(depth-1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, norm)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, norm) for _ in range(depth-1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = self.crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(combined, self.crop(upsampled, combined))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, norm, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride
        self.crop = Crop1d("both")

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, norm)] +
                                   [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, norm) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, norm)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, norm) for _ in
                                                 range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride)
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, norm)

    def forward(self, x):
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size, target_output_size, norm, res, depth=1, strides=2):
        super(Waveunet, self).__init__()

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_outputs = num_channels
        self.depth = depth
        self.instruments = instruments

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)

        self.waveunets = nn.ModuleDict()

        for instrument in instruments:
            module = nn.Module()
            module.crop = Crop1d("both")

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, strides, depth, norm, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides, depth, norm, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, norm) for _ in range(depth)])

            module.output_conv = nn.Conv1d(num_channels[0], num_outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward(self, x):
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        out_dict = {}

        for instrument, module in self.waveunets.items():
            shortcuts = []
            out = x

            # DOWNSAMPLING BLOCKS
            for block in module.downsampling_blocks:
                out, short = block(out)
                shortcuts.append(short)

            # BOTTLENECK CONVOLUTION
            for conv in module.bottlenecks:
                out = conv(out)

            # UPSAMPLING BLOCKS
            for idx, block in enumerate(module.upsampling_blocks):
                out = block(out, shortcuts[-1-idx])

            # OUTPUT CONV
            out = module.output_conv(out)
            if not self.training: # At test time clip predictions to valid amplitude range
                out = out.clamp(min=-1.0, max=1.0)

            out_dict[instrument] = out

        return out_dict