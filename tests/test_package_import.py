# -*- coding: utf-8 -*-
"""
Test the import of "zernpy" package for basic functionality.

@author: Sergei Klykov, @year: 2025, @licence: MIT

"""
import importlib.util


def test_initialization():
    """
    Test basic imports, initialization and usage of implemented classes.

    Returns
    -------
    None

    """
    from zernpy import ZernPol, ZernPSF, force_get_psf_compilation, generate_polynomials
    # clean way to check that numba is installed
    if importlib.util.find_spec("numba") is not None:
        force_get_psf_compilation(True)
    zp = ZernPol(m=-2, n=2)
    assert len(zp.get_polynomial_name()) > 0, "Failed simple function call for getting polynomial name"
    pols = generate_polynomials(4)
    assert "Piston" in str(pols[0]), "Generated polynomials not started with Piston"
    zpsf = ZernPSF(ZernPol(m=-1, n=1))
    zpsf.set_physical_props(NA=0.1, wavelength=0.4, expansion_coeff=0.01, pixel_physical_size=0.4/1.5)
    zpsf.set_calculation_props(kernel_size=25, n_integration_points_r=100, n_integration_points_phi=180)
    zpsf.calculate_psf_kernel(normalized=True, verbose_info=False, suppress_warnings=True); zpsf.crop_kernel()
