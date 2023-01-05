# -*- coding: utf-8 -*-
"""
The "zernpy" package is intended for initialization and calculation attributes / properties of Zernike polynomials.

@author: Sergei Klykov
@licence: MIT, @year: 2023
"""
if __name__ == "__main__":
    # use absolute imports for importing as module
    __all__ = ['zernikepol']  # for specifying from zernpy import * if package imported from some script
elif __name__ == "zernpy":
    pass  # do not add module "zernikepol" to __all__ attribute, because it demands to construct explicit path

# Automatically bring the main class to the name space when the used command both commands:
# 1) from zernpy import ZernPol; 2) from zernpy import *
if __name__ != "__main__":
    from .zernikepol import ZernPol
