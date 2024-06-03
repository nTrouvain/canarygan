# Author: Nathan Trouvain at 29/08/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Pytorch + Lightning implementation of WaveGAN (Donahue et al. 2018).

This code is heavily inspired on the work of 
Mostafa El Araby (https://github.com/mostafaelaraby/wavegan-pytorch)
and Aurora Cramer (https://github.com/auroracramer/wavegan).

We recommend using and citing their work if solely trying to play with WaveGAN.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as pl

from .losses import WassersteinGP
from .params import SliceLengths


def same_padding(input_len, stride, kernel_len):
    """
    "Same" padding mode for Pytorch.

    Adapted from https://www.tensorflow.org/api_docs/python/tf/nn
    """
    if input_len % stride == 0:
        return int(np.ceil(max(kernel_len - stride, 0) / 2))
    else:
        return int(np.ceil(max(kernel_len - (input_len % stride), 0) / 2))


def interpolate(real_x, fake_x):
    """
    Random interpolation of real and generated data,
    as required by the training policy defined in
    Donahue et al. (2018).
    """
    alpha = torch.rand_like(real_x)
    diff = fake_x - real_x
    interp_x = real_x + (alpha * diff)
    interp_x.requires_grad = True  # Important! used for gradient penalty
    return interp_x


def load_generator(generator_ckpt, device="cpu"):
    """
    Load CanaryGAn generator from a saved checkpoint and transfer it
    to device.
    """
    generator_ckpt = Path(generator_ckpt)

    if not generator_ckpt.is_file():
        raise FileNotFoundError(generator_ckpt)

    ckpt = torch.load(
        generator_ckpt,
        map_location=lambda storage, _: (
            storage.cuda(device) if isinstance(device, int) else storage.cpu()
        ),
    )
    state_dict = {
        ".".join(k.split(".")[1:]): v
        for k, v in ckpt["state_dict"].items()
        if "generator" in k
    }

    generator = CanaryGANGenerator().to(device)

    status = generator.load_state_dict(state_dict)
    print(f"Ckpt load status: {str(status)}")

    return generator


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
    """
    WaveGAN discriminator for CanaryGAN.


    Parameters
    ----------
    dim : int, default to 64
        Internal base dimension.
    nch : int, default to 1
        Number of audio channels.
    kernel_len : int, default to 25
        Convolutionnal kernel size.
    stride : int, default to 4
        Convolutionnal kernel stride.
    use_batchnorm : bool, default to False
        If True, apply batch normalization.
    phasesuffle_rad : float, default to 0
        Phase shuffle applied to signal in radiants.
    alpha : float, default to 0.2
        LeakyReLU negative slope value.
    slice_len : SliceLengths, default to 16384
        Length of input audio samples, in number of timesteps.
    """

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
    WaveGAN generator used for CanaryGAN.

    Parameters
    ----------
    dim : int, default to 64
        Internal base dimension.
    latent_dim : int, default to 3
        Latent space dimension.
    nch : int, default to 1
        Number of audio channels.
    kernel_len : int, default to 25
        Convolutionnal kernel size.
    upsample : "zeros" or "nn", default to "zeros"
        Upsample signal using zero padding ("zeros") or nearest neighbours ("nn").
    use_batchnorm : bool, default to False
        If True, apply batch normalization.
    slice_len : SliceLengths, default to 16384
        Length of input audio samples, in number of timesteps.
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


class CanaryGAN(pl.LightningModule):
    """
    WaveGAN as CanaryGAN, as a Pytorch Lightning module for fast distributed training and inference.

    Allows multi-GPU, multi-node training and inference. Has been tested on a cluster equipped with
    SLURM workload manager.

    Note that it will also work locally on a single-GPU single-node setup. It can also be
    converted to vanilla Pytorch module by loading weights in CanaryGANGenerator or CanaryGANDiscriminator.

    Parameters
    ----------
    dim : int, default to 64
        Internal base dimension.
    latent_dim : int, default to 3
        Latent space dimension.
    nch : int, default to 1
        Number of audio channels.
    kernel_len : int, default to 25
        Convolutionnal kernel size.
    upsample : "zeros" or "nn", default to "zeros"
        Upsample signal using zero padding ("zeros") or nearest neighbours ("nn").
    use_batchnorm : bool, default to False
        If True, apply batch normalization.
    phasesuffle_rad : float, default to 0
        Phase shuffle applied to signal in radiants.
    alpha : float, default to 0.2
        LeakyReLU negative slope value.
    lmbda : float, default to 10.0
        Slope of gradient penalty function within loss function.
    slice_len : SliceLengths, default to 16384
        Length of input audio samples, in number of timesteps.
    critic_update_per_batch : int, default to 5
        Number of discriminator training steps performed per batch before
        updating generator parameters.
    lr : float, default to 1e-4
        Learning rate.
    beta1 : float, default to 0.5
        Adam optimizer parameter.
    beta2 : float, default to 0.9
        Adam optimizer parameter.
    seed : int, default to 0
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
        phasesuffle_rad=0,
        alpha=0.2,
        lmbda=10.0,
        slice_len=SliceLengths.SHORT,
        critic_update_per_batch=5,
        lr=1e-4,
        beta1=0.5,
        beta2=0.9,
        seed=0,
    ):
        super(CanaryGAN, self).__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = CanaryGANGenerator(
            dim=dim,
            latent_dim=latent_dim,
            nch=nch,
            kernel_len=kernel_len,
            stride=stride,
            padding=padding,
            upsample=upsample,
            use_batchnorm=use_batchnorm,
            slice_len=slice_len,
        )

        self.critic = CanaryGANDiscriminator(
            dim=dim,
            nch=nch,
            kernel_len=kernel_len,
            stride=stride,
            use_batchnorm=use_batchnorm,
            phasesuffle_rad=phasesuffle_rad,
            alpha=alpha,
            slice_len=slice_len,
        )

        self.wgp_criterion = WassersteinGP(lmbda=lmbda)

        self.example_input_array = torch.rand(20, latent_dim)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch):
        batch_size = batch.size(0)
        real_x = batch

        # We need to instanciate two different optimizers,
        # as discriminator and generator are trained
        # separetely.
        g_opt, c_opt = self.optimizers()

        latent_dim = self.hparams.latent_dim

        # Train the critic on batch
        self.toggle_optimizer(c_opt)  # No gradients for generator!

        c_loss = torch.Tensor([0.0]).to(real_x)

        for i in range(self.hparams.critic_update_per_batch):
            # Sample latent space z ~ U[-1, 1]
            z = torch.rand(real_x.size(0), latent_dim).to(real_x) * 2 - 1

            # x̃ = G(z)
            fake_x = self.generator(z)

            # Compute discriminator judgement probability
            # D(x)
            p_real = self.critic(real_x)
            # D(x̃)
            p_fake = self.critic(
                fake_x.detach()
            )  # .detach to consider G(z) as an input
            # D(αx + (1-α)x̃) = D(x̂)
            interp_x = interpolate(real_x, fake_x.detach())
            p_interp = self.critic(interp_x)

            # Discriminator loss (with gradient penalty)
            c_loss, grad_penalty = self.wgp_criterion(
                p_real, p_fake, p_interp, interp_x
            )

            self.manual_backward(c_loss)
            c_opt.step()
            c_opt.zero_grad()

        self.log(
            "disc_loss",
            c_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Train the generator
        self.untoggle_optimizer(c_opt)
        self.toggle_optimizer(g_opt)

        z = torch.rand(real_x.size(0), latent_dim).to(real_x) * 2 - 1

        # x̃ = G(z)
        fake_x = self.generator(z)

        # Generator loss (-E[D(x̃)])
        g_criterion = -self.critic(fake_x).mean()

        self.manual_backward(g_criterion)
        g_opt.step()
        g_opt.zero_grad()

        self.log(
            "gen_loss",
            g_criterion,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        self.untoggle_optimizer(g_opt)

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = self.hparams.beta1, self.hparams.beta2

        g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        c_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas)
        return g_opt, c_opt
