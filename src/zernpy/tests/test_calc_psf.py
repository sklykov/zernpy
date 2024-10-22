# -*- coding: utf-8 -*-
"""
Test the implemented PSF calculation functions for module calc_psfs and ZernPSF methods by using pytest library.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in the command line.

@author: Sergei Klykov
@licence: MIT

"""
import numpy as np
from pathlib import Path
import os

# Importing the written in the modules test functions for letting pytest library their automatic exploration
if __name__ != "__main__":
    from ..calculations.calc_psfs import (get_psf_kernel)
    from ..zernpsf import ZernPSF, force_get_psf_compilation
    from ..zernikepol import ZernPol
else:
    from zernpy import ZernPol


# Testing functions
def test_psf_kernel_calc():
    NA = 0.35; wavelength = 0.55; pixel_size = wavelength / 3.05; ampl = -0.28  # Common physical properties
    # Basic test - calculating kernel by numerical integration and by Airy pattern exact equation and compare them
    piston = ZernPol(m=0, n=0)
    airy_p = get_psf_kernel(zernike_pol=piston, len2pixels=pixel_size, alpha=ampl, wavelength=wavelength, NA=NA,
                            normalize_values=True, airy_pattern=True)
    airy_int = get_psf_kernel(zernike_pol=piston, len2pixels=pixel_size, alpha=ampl, wavelength=wavelength, NA=NA,
                              normalize_values=True, airy_pattern=False)
    diff_airy = np.abs(airy_int - airy_p)
    assert np.max(diff_airy) < 0.01, print("Difference between exact Airy pattern and calculated by the numerical integral is bigger than 1%:",
                                           np.max(diff_airy))


def test_zernpsf_usage():
    NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 5.0; ampl = 0.55  # Common physical properties
    zp1 = ZernPol(m=0, n=2)  # defocus
    test_init = True
    try:
        zpsf = ZernPSF(zernpol=(0, 2)); test_init = False
    except ValueError:
        pass
    assert test_init, print("ZernPSF wrong initialization passed through")
    zpsf = ZernPSF(zp1)  # proper initialization
    try:
        zpsf.set_physical_props(NA=2.0, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)  # wrong NA
        test_init = False
    except ValueError:
        pass
    assert test_init, print("Wrong NA assigned but no ValueError thrown")
    zpsf.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)  # normal assignment
    zpsf.set_calculation_props(kernel_size=zpsf.kernel_size, n_integration_points_r=250, n_integration_points_phi=300)  # normal assignment
    try:
        zpsf.set_calculation_props(kernel_size=-1, n_integration_points_r=250, n_integration_points_phi=300)
        test_init = False
    except ValueError:
        pass
    assert test_init, print("Wrong kernel size assigned but no ValueError thrown")


def test_save_load_zernpsf():
    zp2 = ZernPol(m=1, n=3); zpsf2 = ZernPSF(zp2)  # horizontal coma
    zp3 = ZernPol(osa=21); zpsf3 = ZernPSF(zp3)  # additional classes for testing reading and reassigning values
    NA = 0.4; wavelength = 0.4; pixel_size = wavelength / 3.2; ampl = 0.16  # Common physical properties
    zpsf2.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
    zpsf2.set_calculation_props(kernel_size=zpsf2.kernel_size, n_integration_points_r=200, n_integration_points_phi=180)
    zpsf3.set_physical_props(NA=1.0, wavelength=0.6, expansion_coeff=-0.1, pixel_physical_size=0.05)
    zpsf2.calculate_psf_kernel(normalized=True)  # normal calculation of a kernel
    zpsf2.save_json(overwrite=True)  # save in the standard location (package folder)
    assert Path(zpsf2.json_file_path).is_file(), "File hasn't been saved in the standard location"
    if Path(zpsf2.json_file_path).is_file():
        zpsf3.read_json(zpsf2.json_file_path)  # test reading of the stored in json file information
        assert (zpsf2.NA == zpsf3.NA and zpsf2.zernpol == zpsf3.zernpol and zpsf2.n_int_phi_points == zpsf3.n_int_phi_points
                and zpsf2.expansion_coeff == zpsf3.expansion_coeff), "Saved and Read PSFs have differences, check I/O operations"
        os.remove(path=str(Path(zpsf2.json_file_path).absolute()))  # remove saved file for not cluttering folder


def test_numba_compilation():
    force_get_psf_compilation()  # force compilation of computation methods
    # Test the difference between accelerated and not accelerated calculation methods
    NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 4.6; ampl = -0.16
    zp6 = ZernPol(m=0, n=2); zpsf6 = ZernPSF(zp6)  # defocus
    zpsf6.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
    kernel_acc = zpsf6.calculate_psf_kernel(normalized=True, accelerated=True)
    kernel_norm = zpsf6.calculate_psf_kernel(normalized=True)
    assert np.max(np.abs(kernel_acc - kernel_norm) < 1E-5), "Accelerated and not one calculation of kernel methods have significant differences"
