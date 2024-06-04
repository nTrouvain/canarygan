# Author: Nathan Trouvain at 8/30/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from ..canarygan.losses import GPDiscriminatorLoss
from ..canarygan.discriminator import CanaryGANDiscriminator
from ..canarygan.const import SliceLengths

import torch


if __name__ == "__main__":
    critic = CanaryGANDiscriminator()

    x = torch.rand(10, 1, SliceLengths.SHORT.value, requires_grad=True)
    x_fake = torch.rand_like(x, requires_grad=True)
    alpha = torch.rand_like(x)
    x_interp = x + alpha * (x_fake - x)

    loss = GPDiscriminatorLoss(lmbda=10.0)

    p_real = critic(x)
    p_fake = critic(x_fake)
    p_interp = critic(x_interp)

    print(loss(p_real, p_fake, p_interp, x_interp))
