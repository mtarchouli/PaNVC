"""
 The following modules are from : https://github.com/Orange-OpenSource/AIVC
 They are licenced under BSD 3-Clause "New"
 
 The licences on which our framework depends are to be found in the folder named "licence_dependency"
""" 
 
"""
Module gathering useful mathematical functions, mainly used for rate estimation 
"""

import torch
import numpy as np
from torch import log10


LOG_NUM_STAB = 2 ** (-16)
DIV_NUM_STAB = 1e-12

PROBA_MIN = LOG_NUM_STAB
PROBA_MAX = 1.0

# Log var cannot be > 10
# (i.e: sigma < exp(0.5 * 10) = 148,41)
LOG_VAR_MAX = 10.
# Log var cannot be < -18.4207
# (i.e: sigma > exp(0.5 * -18.4207) = 0.0001)
LOG_VAR_MIN = -18.4207
