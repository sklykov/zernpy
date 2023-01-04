# -*- coding: utf-8 -*-
"""
Test the implemented calculation functions for module calc_zernike_pol and ZernPol static methods by using pytest library.

pytest library avalaible on: https://docs.pytest.org/en/latest/contents.html
For running this test, it's enough to run the command "pytest" from the repository location.

@author: Sergei Klykov
@licence: MIT

"""
import math

# Importing the written in the modules test functions for letting pytest library their automatic exploration
if __name__ != "__main__":
    from ..zernikepol import check_conformity, ZernPol


# Testing initialization of Zernike polynomials, for details, see the zernikepol module
def test_polynomials_initialization():
    check_conformity()


# Explicit testing initialization of Zernike polynomials
def test_explicit_initialization():
    m = 0; n = 2; zp = ZernPol(l=m, n=n)
    assert abs(zp.radial_dr(0.25) - 1.0) < 1E-9, f"Radial derivative calculated with error for Z{(m, n)}"
    m = 0; n = 6; zp = ZernPol(l=m, n=n)
    assert abs(zp.triangular_dtheta(math.pi)) < 1E-9, f"Triangular derivative calculated with error for Z{(m, n)}"
    m = -1; n = 1; zp = ZernPol(azimuthal_order=m, radial_order=1)
    assert abs(zp.polynomial_value(0.5, math.pi/2) - 1.0) < 1E-9, f"Pol. value Z{(m, n)} for r=0.5, theta=pi/2 calculated with error"
