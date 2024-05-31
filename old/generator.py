# Author: Nathan Trouvain at 29/08/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging

import torch
from torch import nn

from .utils import same_padding
from .params import SliceLengths

logger = logging.getLogger("canarygan/generator")


class UpsampleConv1d(nn.Module):
    """
    With upsampling (using nearest neighbours):
        x -> Upsample -> Padding -> Conv1d [ -> BatchNorm ] -> Activation
    Without upsampling (upsample = None):
        x -> ConvTranspose1d [ -> BatchNorm ] -> Activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_len,
        stride,
        padding,
        output_padding=1,
        upsample="zeros",
    ):
        super(UpsampleConv1d, self).__init__()

        if upsample == "nn":
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                nn.Conv1d(
                    in_channels, out_channels, kernel_len, stride=1, padding=padding
                ),
            )
        else:
            self.block = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel_len,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )

    def forward(self, x):
        return self.block(x)


class CanaryGANGenerator(nn.Module):
    """

    Parameters
    ----------
        dim : int
        nch : int
        kernel_len : int, default to 25
        upsample : "zeros" or "nn", default to "zeros"
        use_batchnorm : bool
        slice_len : SliceLengths
    """

    def __init__(
        self,
        dim=64,
        latent_dim=3,
        nch=1,
        kernel_len=25,
        stride=4,
        padding="same",
        upsample="zeros",
        use_batchnorm=False,
        slice_len=SliceLengths.SHORT,
    ):
        super(CanaryGANGenerator, self).__init__()

        if slice_len not in SliceLengths:
            raise ValueError(
                f"slice_len should be one of SliceLengths enum values: "
                f"SliceLengths.SHORT (16384), "
                f"SliceLengths.MEDIUM (32768), "
                f"SliceLengths.LONG. (65536)"
            )

        self.dim = dim
        self.kernel_len = kernel_len
        self.stride = stride
        self.use_batchnorm = use_batchnorm
        self.upsample = upsample
        self.padding = padding

        if upsample == "zeros":
            self.padding = same_padding(slice_len.value, stride, kernel_len)

        dim_mul = 16 if slice_len == SliceLengths.SHORT else 32

        # In following comments, all dimensions are computed using dim=64

        # Linear block (projection to correct dimension)
        # [z] -> [16, 1024]
        linear = nn.Linear(latent_dim, 4 * 4 * self.dim * dim_mul)
        relu = nn.ReLU()
        if use_batchnorm:
            batchnorm = nn.BatchNorm1d(num_features=self.dim * dim_mul)
            self.linear_block = nn.Sequential(linear, batchnorm, relu)
        else:
            self.linear_block = nn.Sequential(linear, relu)

        deconvnet = [
            # Layer 1
            # [16, 1024] -> [64, 512]
            *self._deconv_block(dim * dim_mul, dim * dim_mul // 2),
            nn.ReLU(),
            # Layer 2
            # [64, 512] -> [256, 256]
            *self._deconv_block(dim * dim_mul // 2, dim * dim_mul // 4),
            nn.ReLU(),
            # Layer 3
            # [256, 256] -> [1024, 128]
            *self._deconv_block(dim * dim_mul // 4, dim * dim_mul // 8),
            nn.ReLU(),
            # Layer 4
            # [1024, 128] -> [4096, 64]
            *self._deconv_block(dim * dim_mul // 8, dim * dim_mul // 16),
            nn.ReLU(),
        ]

        if slice_len == SliceLengths.SHORT:
            deconvnet += [
                # Layer 5
                # [4096, 64] -> [16384, nch]
                *self._deconv_block(dim * dim_mul // 16, nch),
                nn.Tanh(),
            ]
        elif slice_len == SliceLengths.MEDIUM:
            reduced_stride = stride // 2
            if upsample == "zeros":
                padding = same_padding(slice_len.value, reduced_stride, kernel_len)
            else:
                padding = self.padding

            deconvnet += [
                # Layer 5
                # [4096, 128] -> [16384, 64]
                *self._deconv_block(dim * dim_mul // 16, dim),
                nn.ReLU(),
                # Layer 6
                # [16384, 64] -> [32768, nch]
                *self._deconv_block(dim, nch, stride=reduced_stride, padding=padding),
                nn.Tanh(),
            ]
        elif slice_len == SliceLengths.LONG:
            deconvnet += [
                # Layer 5
                # [4096, 128] -> [16384, 64]
                *self._deconv_block(dim * dim_mul // 16, dim),
                nn.ReLU(),
                # Layer 6
                # [16384, 64] -> [65536, nch]
                *self._deconv_block(dim, nch),
                nn.Tanh(),
            ]

        self.deconvnet = nn.Sequential(*deconvnet)

        self._init_deconv_layers()

    def _deconv_block(
        self,
        in_channels,
        out_channels,
        use_batchnorm=None,
        stride=None,
        padding=None,
    ):
        stride = self.stride if stride is None else stride
        padding = self.padding if padding is None else padding
        use_batchnorm = self.use_batchnorm if use_batchnorm is None else use_batchnorm

        deconv = UpsampleConv1d(
            in_channels,
            out_channels,
            kernel_len=self.kernel_len,
            stride=stride,
            padding=padding,
            upsample=self.upsample,
        )

        block = [deconv]

        if use_batchnorm:
            block += [nn.BatchNorm1d(out_channels)]

        return block

    def _init_deconv_layers(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, z):
        h = self.linear_block(z).view(z.size(0), -1, 16)
        output = self.deconvnet(h)
        return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    for slice_len in [SliceLengths.SHORT, SliceLengths.MEDIUM, SliceLengths.LONG]:
        for upsample in ["nn", "zeros"]:
            logger.debug(
                f"Params: upsample = {upsample}, " f"slice_len = {slice_len.value}"
            )

            generator = CanaryGANGenerator(
                dim=64,
                latent_dim=3,
                nch=1,
                kernel_len=25,
                stride=4,
                padding="same",
                upsample=upsample,
                use_batchnorm=False,
                slice_len=slice_len,
            )

            z = torch.Tensor([[0.1, 0.2, 0.1]])

            out = generator(z)
            assert out.shape[-1] == slice_len.value
