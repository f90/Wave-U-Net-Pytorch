from torch import nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, norm, conv_per_residual=2):
        super(ResidualConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.norm = norm
        self.conv_per_residual = conv_per_residual

        # How many channels should be normalised as one group if GroupNorm is activated
        # WARNING: Number of channels has to be divisible by this number!
        NORM_CHANNELS = 8

        self.main_ops = []
        for i in range(conv_per_residual):
            if i == 0:
                curr_inputs = n_inputs
                curr_outputs = n_outputs
            else:
                curr_inputs, curr_outputs = n_outputs, n_outputs

            # NORM
            if norm == "gn":
                assert curr_inputs % NORM_CHANNELS == 0
                self.main_ops.append(
                    nn.GroupNorm(curr_inputs // NORM_CHANNELS, curr_inputs)
                )
            elif norm == "bn":
                self.main_ops.append(nn.BatchNorm1d(curr_inputs, momentum=0.01))
            else:
                raise NotImplementedError(
                    f"We don't support normalization method {norm}!"
                )
            # Add you own types of variations here!

            # CONV
            self.main_ops.append(
                nn.Conv1d(
                    curr_inputs,
                    curr_outputs,
                    self.kernel_size,
                    padding="same",
                    padding_mode="reflect",
                )
            )
            # Non-linear activation
            self.main_ops.append(nn.LeakyReLU(0.01))
        self.main_ops = nn.ModuleList(self.main_ops)

        self.residual_conv = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, x):
        main_out = x
        for op in self.main_ops:
            main_out = op(main_out)
        residual = self.residual_conv(x)
        return main_out + residual
