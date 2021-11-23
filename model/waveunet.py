import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.dataset import center_crop_audio
from model.conv import ResidualConvBlock


class UpsamplingBlock(nn.Module):
    def __init__(
        self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, norm
    ):
        super(UpsamplingBlock, self).__init__()
        assert stride > 1
        assert kernel_size % 2 == 1

        # output size = (input_size - 1) * stride - 2*padding + kernel_size + output_padding,
        # output size = stride*input_size
        # set padding to same parameter as strided conv:
        upconv_padding = (kernel_size - 1) // 2
        # => output_padding = stride + 2*padding - kernel_size
        self.upconv = nn.ConvTranspose1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            output_padding=stride + 2 * upconv_padding - kernel_size,
        )

        self.combine_conv = nn.Conv1d(
            n_outputs + n_shortcut,
            n_outputs,
            kernel_size,
            padding="same",
            padding_mode="reflect",
        )
        self.convs = nn.ModuleList(
            [
                ResidualConvBlock(n_outputs, n_outputs, kernel_size, norm)
                for _ in range(depth)
            ]
        )

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        # Combine with low-level features: Concat with shortcut features and reduce feature channels
        combined = self.combine_conv(torch.cat([upsampled, shortcut], dim=1))

        # Residual conv stack
        for conv in self.convs:
            combined = conv(combined)
        return combined


class DownsamplingBlock(nn.Module):
    def __init__(
        self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, norm
    ):
        super(DownsamplingBlock, self).__init__()
        assert stride > 1
        assert kernel_size % 2 == 1

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.prep_conv = ResidualConvBlock(n_inputs, n_shortcut, kernel_size, norm)
        self.convs = nn.ModuleList(
            [
                ResidualConvBlock(n_shortcut, n_shortcut, kernel_size, norm)
                for _ in range(depth)
            ]
        )

        # CONV 2 with striding
        # output size = floor(((input_size + 2x*padding - kernel_size) / stride) + 1)
        # with padding = (kernel_size-1)/2 and odd kernel size: output size = floor(((input_size - 1)/stride)+1)
        self.downconv = nn.Conv1d(
            n_shortcut,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            padding_mode="reflect",
        )

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = self.prep_conv(x)
        for conv in self.convs:
            shortcut = conv(shortcut)

        out = shortcut
        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut


class Waveunet(nn.Module):
    def __init__(self, args, logger: SummaryWriter):
        super(Waveunet, self).__init__()

        self.num_levels = args.levels
        self.strides = args.strides
        self.kernel_size = args.kernel_size
        self.input_channels = args.channels
        self.output_channels = self.input_channels
        self.depth = args.depth
        self.instruments = args.instruments
        self.norm = args.norm
        self.output_crop = args.output_crop

        self.logger = logger

        self.num_features = (
            [args.features * i for i in range(1, args.levels + 1)]
            if args.feature_growth == "add"
            else [args.features * 2 ** i for i in range(0, args.levels)]
        )

        # Input conv to get an initial number of feature channels
        self.input_conv = nn.Conv1d(
            self.input_channels,
            self.num_features[0],
            self.kernel_size,
            padding="same",
            padding_mode="reflect",
        )

        # Downsampling blocks
        self.downsampling_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.downsampling_blocks.append(
                DownsamplingBlock(
                    self.num_features[i],
                    self.num_features[i],
                    self.num_features[i + 1],
                    self.kernel_size,
                    self.strides,
                    self.depth,
                    self.norm,
                )
            )

        # Upsampling blocks
        self.upsampling_blocks = nn.ModuleList()
        for i in range(0, self.num_levels - 1):
            self.upsampling_blocks.append(
                UpsamplingBlock(
                    self.num_features[-1 - i],
                    self.num_features[-2 - i],
                    self.num_features[-2 - i],
                    self.kernel_size,
                    self.strides,
                    self.depth,
                    self.norm,
                )
            )

        self.bottleneck = nn.ModuleList(
            [
                ResidualConvBlock(
                    self.num_features[-1],
                    self.num_features[-1],
                    self.kernel_size,
                    self.norm,
                )
                for _ in range(self.depth)
            ]
        )

        # Output conv
        self.output_conv = nn.Conv1d(
            self.num_features[0], self.output_channels * len(self.instruments), 1
        )

    def forward(self, mix, step=None):
        shortcuts = []

        # Input convolution
        out = self.input_conv(mix)

        # DOWNSAMPLING BLOCKS
        for idx, block in enumerate(self.downsampling_blocks):
            out, short = block(out)
            if step and step % 100 == 0:
                self.logger.add_scalar(
                    f"ds_{idx}_l2",
                    torch.mean(torch.norm(out.view(out.shape[0], -1), p=2, dim=1)),
                    global_step=step,
                )
                self.logger.add_histogram(f"ds_{idx}_hist", out, global_step=step)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in self.bottleneck:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(self.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])
            if step and step % 100 == 0:
                self.logger.add_scalar(
                    f"us_{idx}_l2",
                    torch.mean(torch.norm(out.view(out.shape[0], -1), p=2, dim=1)),
                    global_step=step,
                )

        # OUTPUT CONV
        out = self.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)

        # Center-crop output if we want to ignore predictions near output window borders
        out = center_crop_audio(out, self.output_crop)

        return out


# TODO Conditioned Wave-U-Net
