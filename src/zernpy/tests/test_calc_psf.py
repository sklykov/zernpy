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
    assert np.max(diff_airy) < 0.01, ("Difference between exact Airy pattern and calculated by the numerical integral is bigger than 1%:"
                                      + str(np.max(diff_airy)))
    # Test normal calculation of several polynomials
    pols = (ZernPol(osa=10), ZernPol(osa=15)); coeffs = (0.08, -0.07)
    zpsf = ZernPSF(pols); zpsf.set_physical_props(NA, wavelength, expansion_coeff=coeffs, pixel_physical_size=wavelength / 2.8)
    zpsf.set_calculation_props(kernel_size=23, n_integration_points_r=150, n_integration_points_phi=120)
    psf_kernel = zpsf.calculate_psf_kernel(); psf_kernel_size = zpsf.kernel_size
    w_orig_kernel, h_orig_kernel = psf_kernel.shape
    assert np.max(psf_kernel) > 0.5 and np.min(psf_kernel) > -1E-5, "Check calculation of a kernel for several polynomials"
    # Test cropping of calculated kernel
    zpsf.crop_kernel()  # cropping the calculated kernel
    cropped_kernel = zpsf.kernel; cropped_kernel_size = zpsf.kernel_size
    w_crop_kernel, h_crop_kernel = cropped_kernel.shape
    assert psf_kernel_size > cropped_kernel_size, (f"Cropped kernel size ({psf_kernel_size}) is equal "
                                                   + f"or more than original ({cropped_kernel_size})")
    assert w_orig_kernel > w_crop_kernel and h_orig_kernel > h_crop_kernel, (f"Cropped kernel shape ({cropped_kernel.shape}) is equal "
                                                                             + f"or more than original({psf_kernel.shape})")


def test_zernpsf_usage():
    NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 5.0; ampl = 0.55  # Common physical properties
    zp1 = ZernPol(m=0, n=2)  # defocus
    test_init = True  # flag, if set to False, the test failed (e.g., expected Error not caught)
    # Test for a wrong ZernPol
    try:
        zpsf = ZernPSF(zernpol=(0, 2)); test_init = False
    except ValueError:
        pass
    assert test_init, "ZernPSF initialization with wrong ZernPol (as a tuple with orders) passed through"
    zpsf = ZernPSF(zp1)  # proper initialization
    # Wrong NA provided
    try:
        zpsf.set_physical_props(NA=2.0, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)  # wrong NA
        test_init = False
    except ValueError:
        pass
    assert test_init, "Wrong NA assigned (=2.0) but no ValueError thrown"
    # Wrong kernel size set
    zpsf.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)  # normal assignment
    zpsf.set_calculation_props(kernel_size=zpsf.kernel_size, n_integration_points_r=225, n_integration_points_phi=240)  # normal assignment
    try:
        zpsf.set_calculation_props(kernel_size=-1, n_integration_points_r=250, n_integration_points_phi=300)
        test_init = False
    except ValueError:
        pass
    assert test_init, "Wrong kernel size (=-1) assigned but no ValueError thrown"
    # Several polynomials provided but with single amplitude
    zpsf = ZernPSF([ZernPol(osa=3), ZernPol(osa=10)]); ampls = (0.21)  # test case - 2 pol-s, 1 ampl.
    std_na = 0.95; std_wl = 0.52; std_pixel_size = 6.5 / 63  # standard physical parameters
    try:
        zpsf.set_physical_props(NA=std_na, wavelength=std_wl, expansion_coeff=ampls, pixel_physical_size=std_pixel_size)
        test_init = False
    except ValueError:
        pass
    assert test_init, "2 pol-s provided but with only 1 amplitude and not ValueError thrown"
    # Several polynomials provided but with more amplitudes than polynomials + provided Piston
    zpsf = ZernPSF((ZernPol(osa=3), ZernPol(osa=0))); ampls = [0.1, -0.2, -1.0]  # test case - 2 pol-s, 3 ampl-s
    try:
        zpsf.set_physical_props(NA=std_na, wavelength=std_wl, expansion_coeff=ampls, pixel_physical_size=std_pixel_size)
        test_init = False
    except ValueError:
        pass
    assert test_init, "2 pol-s provided but with 3 amplitudes and not ValueError thrown"
    # Single polynomial + amplitude as Sequence provided
    ampls = [0.75]; ampls = np.asarray(ampls); pols = (ZernPol(noll=14), )  # test case
    zpsf = ZernPSF(pols); zpsf.set_physical_props(NA=std_na, wavelength=std_wl, expansion_coeff=ampls, pixel_physical_size=std_pixel_size)


def test_save_load_zernpsf():
    zp2 = ZernPol(m=1, n=3); zpsf2 = ZernPSF(zp2)  # horizontal coma
    zp3 = ZernPol(osa=21); zpsf3 = ZernPSF(zp3)  # additional classes for testing reading and reassigning values
    NA = 0.4; wavelength = 0.4; pixel_size = wavelength / 3.0; ampl = 0.11  # Common physical properties
    zpsf2.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
    zpsf2.set_calculation_props(kernel_size=zpsf2.kernel_size, n_integration_points_r=200, n_integration_points_phi=180)
    zpsf3.set_physical_props(NA=1.0, wavelength=0.6, expansion_coeff=-0.08, pixel_physical_size=0.06)
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
    NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 4.25; ampl = -0.12
    zp6 = ZernPol(m=0, n=2); zpsf6 = ZernPSF(zp6); zpsf7 = ZernPSF(zp6)  # defocus
    zpsf6.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
    zpsf7.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
    kernel_acc = zpsf6.calculate_psf_kernel(normalized=True, accelerated=True)  # accelerated by numba compilation
    kernel_norm = zpsf7.calculate_psf_kernel(normalized=True)  # normal calculation
    assert np.max(np.abs(kernel_acc - kernel_norm) < 1E-6), "Accelerated and not one calculation of kernel methods have significant differences"
