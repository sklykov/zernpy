# -*- coding: utf-8 -*-
"""
Test the implemented calculation functions for module calc_zernike_pol and ZernPol static methods by using pytest library.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in the command line.

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
    # Testing the ordinary, normal initialization of polynomials
    m = 0; n = 2; zp = ZernPol(l=m, n=n)
    assert abs(zp.radial_dr(0.25) - 1.0) < 1E-9, f"Radial derivative calculated with error for Z{(m, n)}"
    m = 0; n = 6; zp = ZernPol(n=n, l=m)
    assert abs(zp.triangular_dtheta(math.pi)) < 1E-9, f"Triangular derivative calculated with error for Z{(m, n)}"
    m = -1; n = 1; zp = ZernPol(azimuthal_order=m, radial_order=n)
    assert abs(zp.polynomial_value(0.5, math.pi/2) - 1.0) < 1E-9, f"Pol. value Z{(m, n)} for r=0.5, theta=pi/2 calculated with error"
    m = 2; n = 2; zp = ZernPol(azimuthal_order=m, radial_order=n)
    assert abs(zp.radial(0.2) - 0.04) < 1E-9, f"Radial func. R{(m, n)} for r=0.2 calculated with error"
    assert abs(zp.triangular(math.pi/6) - 0.5) < 1E-9, f"Triangular func. for Z{(m, n)} for theta=pi/6 calculated with error"
    assert abs(zp.triangular_dtheta(math.pi/12) + 1.0) < 1E-9, f"Derivative from triangular func. for Z{(m, n)}, theta=pi/12"
    assert abs(zp.radial_dr(0.25) - 0.5) < 1E-9, f"Radial derivative calculated with error for Z{(m, n)}"
    assert abs(zp.normf() - math.sqrt(6)) < 1E-9, f"Normalization factor for Z{(m, n)} calculated with error"
    zp = ZernPol(fringe_index=11); (m, n), osa, noll, fringe = zp.get_indices()
    assert (osa == 6 and noll == 9 and fringe == 11
            and m == -3 and n == 3), f"Some error in definition of indices: {(m, n), osa, noll, fringe} for ZernPol(fringe = 11)"
    zp = ZernPol(noll=1); assert abs(zp.normf() - 1) < 1E-9, "Normalization factor for Z(noll=1) calculated with error"

    # Testing the initialization and getting names for various combination of parameters
    zernpol = ZernPol(n=7, l=-5); m, n = zernpol.get_mn_orders()
    assert zernpol.get_polynomial_name() == "Vertical secondary pentafoil", f"Returned wrong name for Z{(m, n)}"
    zernpol = ZernPol(osa=9); m, n = zernpol.get_mn_orders()
    assert zernpol.get_polynomial_name() == "Oblique trefoil", f"Returned wrong name for Z{(m, n)}"
    zernpol = ZernPol(noll=15); m, n = zernpol.get_mn_orders()
    assert zernpol.get_polynomial_name(True) == "Obliq. 4foil", f"Returned wrong short name for Z{(m, n)}"
    zernpol = ZernPol(fringe=60); m, n = zernpol.get_mn_orders()
    assert len(zernpol.get_polynomial_name()) == 0, f"Returned some name for Z{(m, n)}, but it's not defined"

    # Testing wrong initialization parameters - for checking that they are not passed through
    # OSA
    try:
        ZernPol(osa=-1); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(osa=-1)"
    try:
        ZernPol(osa=1600); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(osa=1600)"

    # Noll
    try:
        ZernPol(noll=0); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(noll=0)"
    try:
        ZernPol(noll=-2); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(noll=-2)"
    try:
        ZernPol(noll=1580); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(noll=1580)"

    # Fringe
    try:
        ZernPol(fringe=0); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(fringe=0)"
    try:
        ZernPol(fringe=0.4); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(fringe=0.4)"
    try:
        ZernPol(fringe=2981); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(fringe=2981)"

    # Orders radial, azimuthal
    try:
        ZernPol(l=2, n=3); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(l=2, n=3)"
    try:
        ZernPol(m=4, n=3); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(m=4, n=3)"
    try:
        ZernPol(n=-2, l=2); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(n=-2, l=2)"
    try:
        ZernPol(n=55, l=-3); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(n=55, l=-3)"

    # Wrong mix of orders
    try:
        ZernPol(osa=2, noll=9); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(osa=2, noll=9)"
    try:
        ZernPol(m=2, osa=3); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(m=2, osa=3)"
    try:
        ZernPol(); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol()"
    try:
        ZernPol(fringe=5, l=2); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(fringe=5, l=2)"
    try:
        ZernPol(m=2, l=2); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(m=2, l=2)"

    # Wrong orders specification
    try:
        ZernPol(n=2, m=1.01); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(n=2, m=1.01)"
    try:
        ZernPol(fringe='1'); assert_flag = False
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong initialization parameter passed: ZernPol(fringe='1')"

    # Testing some implemented methods for the ZernPol class
    zp1 = ZernPol(osa=4); zp2 = ZernPol(osa=5)
    assert zp2 > zp1, "Implemented method '>' isn't correct"

    zp1 = ZernPol(m=0, n=2); zp2 = ZernPol(osa=4)
    assert zp1 == zp2, "Implemented method '==' isn't correct"

    zp1 = ZernPol(fringe=21); zp2 = ZernPol(noll=8)
    assert not zp1 == zp2, "Implemented method '==' isn't correct"

    zp1 = ZernPol(fringe=17); zp2 = ZernPol(osa=14)
    assert zp1 == zp2, "Implemented method '==' isn't correct"
