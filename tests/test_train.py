# Author: Nathan Trouvain at 8/30/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from ..canarygan.train import train

if __name__ == "__main__":
    train(epochs=1, latent_dim=3, wavegan_disc_nupdates=5)
