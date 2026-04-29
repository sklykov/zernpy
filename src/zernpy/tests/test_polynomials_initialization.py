# -*- coding: utf-8 -*-
"""
Test the implemented calculation functions for module calc_zernike_pol and ZernPol static methods by using pytest library.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in the command line.

@author: Sergei Klykov
@licence: MIT

"""
import math

import numpy as np

from ..zernikepol import ZernPol


def test_polynomials_initialization():
    """
    Test various initialization ways of Zernike polynomials, for details, see the imported function.

    Returns
    -------
    None
    """
    zp = ZernPol(m=-2, n=2)  # Initialization with orders
    (m1, n1), osa_i, noll_i, fringe_i = zp.get_indices()
    assert (osa_i == 3 and noll_i == 5 and fringe_i == 6), (f"Check consistency of Z{(m1, n1)} indices: "
                                                            + f"OSA: {osa_i}, Noll: {noll_i}, Fringe: {fringe_i}")
    zp = ZernPol(l=-3, n=5)
    (m2, n2), osa_i, noll_i, fringe_i = zp.get_indices()
    assert (osa_i == 16 and noll_i == 19 and fringe_i == 20), (f"Check consistency of Z{(m2, n2)} indices: "
                                                               + f"OSA: {osa_i}, Noll: {noll_i}, Fringe: {fringe_i}")
    assert len(zp.get_polynomial_name(short=True)) > 0, f"Short name for Z{(m2, n2)} is zero length"
    zp = ZernPol(azimuthal_order=-1, radial_order=5)
    (m3, n3), osa_i, noll_i, fringe_i = zp.get_indices()
    assert (osa_i == 17 and noll_i == 17 and fringe_i == 15), (f"Check consistency of Z{(m3, n3)} indices: "
                                                               + f"OSA: {osa_i}, Noll: {noll_i}, Fringe: {fringe_i}")
    assert len(zp.get_polynomial_name()) > 0, f"Name for Z{(m3, n3)} is zero length"
    m4, n4 = zp.get_mn_orders()
    assert m4 == m3 and n3 == n4, f"Check method get_mn_orders() for Z{(m3, n3)}"
    print(f"Initialization of polynomials Z{(m1, n1)}, Z{(m2, n2)}, Z{(m3, n3)} tested")
    osa_i = 12; zp = ZernPol(osa_index=osa_i)  # Initialization with OSA index
    m, n = zp.get_mn_orders()
    assert (m == 0 and n == 4), f"Check consistency of Z[OSA index = {osa_i}] orders {m, n}"
    assert zp.get_fringe_index(m, n) == 9, f"Check consistency of Z[OSA index = {osa_i}] Fringe index"
    assert zp.get_noll_index(m, n) == 11, f"Check consistency of Z[OSA index = {osa_i}] Noll index"
    print(f"Initialization of polynomial Z[OSA index = {osa_i}] tested")
    noll_i = 10  # Testing static methods
    assert ZernPol.noll2osa(noll_i) == 9, f"Check consistency of Noll index {noll_i} conversion to OSA index"
    assert ZernPol.osa2fringe(ZernPol.noll2osa(noll_i)) == 10, ("Check consistency of Noll "
                                                                + f"index {noll_i} conversion to OSA index")
    print(f"Conversion of Noll index {noll_i} to OSA and Fringe indices tested")
    # Test for not proper initialization
    try:
        m_f = 2; n_f = -2
        zp = ZernPol(m=m_f, n=n_f)
        asserting_value = False
    except ValueError:
        print(f"Polynomial Z{(m_f, n_f)} haven't been initialized, test passed")
        asserting_value = True
    assert asserting_value, f"Polynomial Z{(m_f, n_f)} initialized with wrong orders assignment"
    # Testing input parameters for calculation
    zp = ZernPol(m=0, n=2); r = 0.0; theta = math.pi
    assert abs(zp.polynomial_value(r, theta) + math.sqrt(3)) < 1E-6, f"Check value of Z[{m}, {n}]({r}, {theta})"
    zp = ZernPol(m=-1, n=1); r = 0.5; theta = math.pi/2
    assert abs(zp.polynomial_value(r, theta) - 1.0) < 1E-6, f"Check value of Z[{m}, {n}]({r}, {theta})"
    print("Simple values of Zernike polynomials tested successfully")
    try:
        r = 'd'; theta = [1, 2]
        zp.polynomial_value(r, theta)
        asserting_value = False
    except ValueError:
        print("Input as string is not allowed for calculation of polynomial value, tested successfully")
        asserting_value = True
    assert asserting_value, "Wrong parameter passed (string) for calculation of polynomial value"
    try:
        r = [0.1, 0.2, 1.0+1E-9]; theta = math.pi
        zp.polynomial_value(r, theta)
        asserting_value = False
    except ValueError:
        print("Radius more than 1.0 is not allowed, tested successfully")
        asserting_value = True
    assert asserting_value, "Wrong parameter passed (r > 1.0) for calculation of polynomial value"
    # Compare two implementations of Zernike pol-s sum calculation: direct and using meshgrid
    pols = [ZernPol(osa=2), ZernPol(osa=4), ZernPol(osa=7), ZernPol(osa=10), ZernPol(osa=15),
            ZernPol(osa=3), ZernPol(osa=9), ZernPol(osa=12), ZernPol(osa=16), ZernPol(osa=19)]
    ampls = [-0.85, 0.85, 0.24, -0.37, 1.0, 0.1, -1.0, -0.05, 1.1, 0.41]
    radii = np.arange(start=0.0, stop=1.0 + 0.001, step=0.001); thetas = np.arange(start=0.0, stop=2.0*np.pi + np.pi/180, step=np.pi/180)
    ZernPol.sum_zernikes(ampls, pols, radii, thetas, get_surface=True)
    ZernPol._sum_zernikes_meshgrid(ampls, pols, radii, thetas)


def test_explicit_initialization():
    """
    Test particular initialization scenarios of Zernike polynomials.

    Returns
    -------
    None
    """
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
    assert zp1 != zp2, "Implemented method '==' isn't correct"

    zp1 = ZernPol(fringe=17); zp2 = ZernPol(osa=14)
    assert zp1 == zp2, "Implemented method '==' isn't correct"
