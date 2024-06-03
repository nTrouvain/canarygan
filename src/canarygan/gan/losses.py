# Author: Nathan Trouvain at 30/08/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import torch

from torch import autograd
from torch import nn


class WassersteinGP(nn.Module):
    """
    Wassertein distance loss with gradient penalty, as introduced in
    Gulrajani et al. (2017) and used in WaveGAN (Donahue et al. (2018)).

    This code is heavily inspired on the work of Mostafa El Araby
    (https://github.com/mostafaelaraby/wavegan-pytorch)
    and Aurora Cramer (https://github.com/auroracramer/wavegan).

    We recommend using and citing their work if solely trying to play with WaveGAN.

    Parameters
    ----------
    lmbda : float
        Slope of gradient penalty function.
    """

    def __init__(self, lmbda):
        super(WassersteinGP, self).__init__()
        self.lmbda = lmbda

    def forward(self, p_real, p_fake, p_interp, interp_x):
        """
        Loss forward function.

        Parameters
        ----------
        p_real : Tensor
            Discriminator response to real sound samples.
        p_fake : Tensor
            Discriminator response to generated sound samples.
        p_interp : Tensor
            Discriminator response to an interpolation of real and generated sounds.
        interp_x : Tensor
            Interpolation of real and generated sounds.
        """
        # Calculate gradient penalty
        # from Gulrajani et al (2017)
        # Norm of second order gradients from
        # sound interpolation serve as a penalty.
        gradients = autograd.grad(
            outputs=p_interp,
            inputs=interp_x,
            grad_outputs=torch.ones(p_interp.size()).to(interp_x),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        grad_penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
        loss = p_fake.mean() - p_real.mean() + self.lmbda * grad_penalty
        return loss, grad_penalty
