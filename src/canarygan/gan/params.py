# Author: Nathan Trouvain at 29/08/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from enum import Enum


class SliceLengths(Enum):
    """
    Admissible lengths of generated audio samples.

    At a 16000kHz sampling rate, these corresponds 
    approximately to 1s, 2s and 4s of sound.
    """
    SHORT = 16384
    MEDIUM = 32768
    LONG = 65536
