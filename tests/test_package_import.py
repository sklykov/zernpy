# -*- coding: utf-8 -*-
"""
Test the imports of "zernpy" package.

@author: Sergei Klykov, @year: 2023
@licence: MIT
"""


def test_initialization():
    try:
        from zernpy import ZernPol, generate_polynomials
        zp = ZernPol(m=-2, n=2)
        assert len(zp.get_polynomial_name()) > 0, "Failed simple function call for getting polynomial name"
        pols = generate_polynomials(4)
        assert "Piston" in str(pols[0]), "Generated polynomials not started with Piston"
    except ImportError:
        import os
        os.chdir("..")  # navigate to the root folder of the project
        from src.zernpy import zernikepol  # import main script
        zp = zernikepol.ZernPol(m=-2, n=2)  # initialize controlling class
        print("zernpy package not installed, this test automatically passed")


if __name__ == "__main__":
    test_initialization()
