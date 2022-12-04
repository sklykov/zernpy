# -*- coding: utf-8 -*-
"""
Make folder containing this script the importable module.

@author: Sergei Klykov
@licence: MIT
"""
if __name__ == "zernpy":
    # use absolute imports for importing as module
    __all__ = ['zernikepol']  # for specifying from zernpy import *
else:
    # use relative imports for importing as installed package
    __all__ = ['.zernikepol']  # for specifying from zernpy import *

# Automatically bring the main class to the name space when the import called
from .zernikepol import ZernPol
from .calculations.calc_zernike_pol import normalization_factor, triangular_function, radial_polynomial
