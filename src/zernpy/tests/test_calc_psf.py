# -*- coding: utf-8 -*-
"""
Test the implemented PSF calculation functions for module calc_psfs and ZernPSF methods by using pytest library.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in the command line.

@author: Sergei Klykov
@licence: MIT

"""
import numpy as np

# Importing the written in the modules test functions for letting pytest library their automatic exploration
if __name__ != "__main__":
    from ..calculations.calc_psfs import (get_psf_kernel)
    from ..zernpsf import ZernPSF
    from ..zernikepol import ZernPol
else:
    from zernpy import ZernPol


# Testing functions
def test_psf_kernel_calc():
    # Common physical properties
    NA = 0.65; wavelength = 0.55; pixel_size = wavelength / 4.0; ampl = -0.4
    # Basic test - calculating kernel by numerical integration and by Airy pattern exact equation
    piston = ZernPol(m=0, n=0)
    airy_p = get_psf_kernel(zernike_pol=piston, len2pixels=pixel_size, alpha=ampl, wavelength=wavelength, NA=NA,
                            normalize_values=True, airy_pattern=True)
