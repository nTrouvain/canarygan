# Author: Nathan Trouvain at 29/08/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
# Author: https: // github.com / mostafaelaraby / wavegan - pytorch / blob / master / models.py
# Author: https://github.com/auroracramer/wavegan/blob/main/wavegan.py
import logging

import torch
import torch.nn.functional as F
from torch import nn

from .utils import same_padding
from .params import SliceLengths


logger = logging.getLogger("canarygan/discriminator")


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary
    """

    def __init__(self, rad, padding="reflect"):
        super(PhaseShuffle, self).__init__()

        self.rad = rad
        self.padding = padding

    def forward(self, x):
        zero = torch.zeros(1).to(x)
        phase = torch.randint(-self.rad, self.rad + 1, (1,)).to(x)
        pad_l = torch.maximum(phase, zero).int().item()
        pad_r = torch.maximum(-phase, zero).int().item()
        x = F.pad(x, (pad_l, pad_r), mode=self.padding)
        return x[..., pad_r : -(1 + pad_l)]


class CanaryGANDiscriminator(nn.Module):
    def __init__(
        self,
        dim=64,
        nch=1,
        kernel_len=25,
        stride=4,
        use_batchnorm=False,
        phasesuffle_rad=0,
        alpha=0.2,
        slice_len=SliceLengths.SHORT,
    ):
        super(CanaryGANDiscriminator, self).__init__()

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
        self.alpha = alpha
        self.phasesuffle_rad = phasesuffle_rad
        self.padding = same_padding(slice_len.value, stride, kernel_len)

        convnet = [
            # Layer 0
            # [16384, 1] -> [4096, 64]
            *self._conv_block(nch, dim, phaseshuffle=True, use_batchnorm=False),
            nn.LeakyReLU(negative_slope=alpha),
            # Layer 1
            # [4096, 64] -> [1024, 128]
            *self._conv_block(dim, 2 * dim, phaseshuffle=True),
            nn.LeakyReLU(negative_slope=alpha),
            # Layer 2
            # [1024, 128] -> [256, 256]
            *self._conv_block(2 * dim, 4 * dim, phaseshuffle=True),
            nn.LeakyReLU(negative_slope=alpha),
            # Layer 3
            # [256, 256] -> [64, 512]
            *self._conv_block(4 * dim, 8 * dim, phaseshuffle=True),
            nn.LeakyReLU(negative_slope=alpha),
            # Layer 4
            # [64, 512] -> [16, 1024]
            *self._conv_block(8 * dim, 16 * dim),
            nn.LeakyReLU(negative_slope=alpha),
        ]

        dim_mul = 16

        if slice_len == SliceLengths.MEDIUM:
            reduced_stride = stride // 2
            padding = same_padding(slice_len.value, reduced_stride, kernel_len)

            convnet += [
                # Layer 5
                # [32, 1024] -> [16, 2048]
                *self._conv_block(
                    16 * dim, 32 * dim, stride=reduced_stride, padding=padding
                ),
                nn.LeakyReLU(negative_slope=alpha),
            ]
            dim_mul = 32

        elif slice_len == SliceLengths.LONG:
            convnet += [
                # Layer 5
                # [64, 1024] -> [16, 2048]
                *self._conv_block(16 * dim, 32 * dim),
                nn.LeakyReLU(negative_slope=alpha),
            ]
            dim_mul = 32

        self.convnet = nn.Sequential(*convnet)

        # Single logit readout
        self.readout = nn.Linear(dim_mul * 16 * dim, 1)

        self._init_conv_layers()

    def _conv_block(
        self,
        in_channels,
        out_channels,
        phaseshuffle=False,
        use_batchnorm=None,
        stride=None,
        padding=None,
    ):
        stride = self.stride if stride is None else stride
        padding = self.padding if padding is None else padding
        use_batchnorm = self.use_batchnorm if use_batchnorm is None else use_batchnorm

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_len,
            stride=stride,
            padding=padding,
        )

        block = [conv]

        if use_batchnorm:
            block += [nn.BatchNorm1d(out_channels)]

        if phaseshuffle:
            block += [PhaseShuffle(self.phasesuffle_rad, padding="reflect")]

        return block

    def _init_conv_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        h = self.convnet(x)
        output = self.readout(h.view(x.size(0), -1))
        return output.squeeze()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    for slice_len in [SliceLengths.SHORT, SliceLengths.MEDIUM, SliceLengths.LONG]:
        logger.debug(f"Params: slice_len = {slice_len.value}")

        discriminator = CanaryGANDiscriminator(
            dim=64,
            nch=1,
            kernel_len=25,
            stride=4,
            phasesuffle_rad=8,
            use_batchnorm=True,
            slice_len=slice_len,
        )

        x = torch.rand((1, 1, slice_len.value))
        out = discriminator(x)
        assert out.shape[-1] == 1
