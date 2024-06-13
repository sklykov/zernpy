# -*- coding: utf-8 -*-
"""
Test the imports of "zernpy" package.

@author: Sergei Klykov, @year: 2024, @licence: MIT

"""


def test_initialization():
    try:
        from zernpy import ZernPol, generate_polynomials, ZernPSF
        zp = ZernPol(m=-2, n=2)
        assert len(zp.get_polynomial_name()) > 0, "Failed simple function call for getting polynomial name"
        pols = generate_polynomials(4)
        assert "Piston" in str(pols[0]), "Generated polynomials not started with Piston"
        zpsf = ZernPSF(zp); zpsf.set_physical_props(NA=1.25, wavelength=0.405, expansion_coeff=0.5, pixel_physical_size=0.405/5.85)
    except ImportError:
        import os
        os.chdir("..")  # navigate to the root folder of the project
        from src.zernpy import zernikepol, zernpsf  # import main scripts
        zp = zernikepol.ZernPol(m=-2, n=2)  # initialize Zernike ppolynomial ZernPol class
        zpsf = zernpsf.ZernPSF(zp)  # initialize ZernPSF class
        zpsf.set_physical_props(NA=1.25, wavelength=0.405, expansion_coeff=0.5, pixel_physical_size=0.405/5.85)
        print("zernpy package not installed, this test automatically passed")


if __name__ == "__main__":
    test_initialization()
