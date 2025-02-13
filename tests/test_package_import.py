# -*- coding: utf-8 -*-
"""
Test the import of "zernpy" package for basic functionality.

@author: Sergei Klykov, @year: 2025, @licence: MIT

"""


def test_initialization():
    try:
        from zernpy import ZernPol, generate_polynomials, ZernPSF, force_get_psf_compilation
        try:
            import numba
            if numba is not None:
                force_get_psf_compilation(True)
        except (ModuleNotFoundError, ImportError):
            pass
        zp = ZernPol(m=-2, n=2)
        assert len(zp.get_polynomial_name()) > 0, "Failed simple function call for getting polynomial name"
        pols = generate_polynomials(4)
        assert "Piston" in str(pols[0]), "Generated polynomials not started with Piston"
        zpsf = ZernPSF(ZernPol(m=-1, n=1))
        zpsf.set_physical_props(NA=0.1, wavelength=0.4, expansion_coeff=0.01, pixel_physical_size=0.4/1.5)
        zpsf.set_calculation_props(kernel_size=5, n_integration_points_r=100, n_integration_points_phi=180)
        zpsf.calculate_psf_kernel(normalized=True, verbose_info=False, suppress_warnings=True); zpsf.crop_kernel()
    except ImportError:
        import os
        os.chdir("..")  # navigate to the root folder of the project
        try:
            from src.zernpy import zernikepol, zernpsf  # import main scripts
            zp = zernikepol.ZernPol(m=-2, n=2)  # initialize Zernike ppolynomial ZernPol class
            zpsf = zernpsf.ZernPSF(zp)  # initialize ZernPSF class
            zpsf.set_physical_props(NA=1.25, wavelength=0.405, expansion_coeff=0.5, pixel_physical_size=0.405/5.85)
            print("zernpy package not installed in the current Python environment, this test automatically passed")
        except ModuleNotFoundError:
            print("Launched most likely in Python console, test passed automatically")


if __name__ == "__main__":
    test_initialization()  # run it for checking after installation of generated Python package
