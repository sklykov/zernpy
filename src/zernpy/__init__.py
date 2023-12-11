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

# Automatically bring the main class and some methods to the name space when one of import command is used commands:
# 1) from zernpy import ZernPol, ... functions; 2) from zernpy import *
if __name__ != "__main__" and __name__ != "__mp_main__":
    from .zernikepol import ZernPol  # main class auto export on the import call of the package
    # functions auto export - when everything imported from the module
    from .zernikepol import generate_polynomials, fit_polynomials, generate_random_phases, fit_polynomials_vectors, generate_phases_image
