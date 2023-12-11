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

# Importing the written in the modules test functions for letting pytest library their automatic exploration
if __name__ != "__main__":
    from ..calculations.calc_zernike_pol import (compare_radial_calculations, compare_radial_derivatives,
                                                 compare_recursive_coeffs_radials, compare_recursive_coeffs_radials_dr,
                                                 check_high_orders_recursion)
    from ..zernikepol import ZernPol


# Testing implemented equations for tabular R(m, n) from References by comparing with the exact ones (with factorials)
def test_tabular_orders():
    compare_radial_calculations(max_order=10)


# Testing implemented tabular and recursive equations for R(m, n) by comparing with the exact ones (with factorials)
# Also, testing implemented recursive scheme for finding the polynomials coefficients for each order and comparison
# with the exact equations (with factorials)
def test_recursive_orders():
    compare_radial_calculations(max_order=17)
    compare_recursive_coeffs_radials()
    check_high_orders_recursion()


# Testing derived equations for derivatives dR(m, n)/dr by comparing with the exact ones (with factorials)
def test_tabular_derivatives():
    compare_radial_derivatives(max_order=10)


# Testing recursive and derived equations for derivatives dR(m, n)/dr by comparing with the exact ones (with factorials)
# Also, testing implemented recursive scheme for finding the polynomials coefficients for each order and comparison
# with the exact equations (with factorials) - derivative cases
def test_recursive_derivatives():
    compare_radial_derivatives(max_order=17)
    compare_recursive_coeffs_radials_dr()


# Testing sum of Zernike polynomials
def test_sum_zernikes():
    zp1 = ZernPol(osa_index=1); zp2 = ZernPol(noll_index=2); ampls = [-0.5, 0.5]; theta = math.pi/3; r = 1.0
    sum_pols = ZernPol.sum_zernikes(coefficients=ampls, polynomials=[zp1, zp2], r=r, theta=theta)
    sum_pols_manual = math.cos(theta) - math.sin(theta)  # manual calculation of specified above sum
    assert abs(sum_pols - sum_pols_manual) < 1E-6, (f"Sum of Zernikes {zp1.get_polynomial_name()} and "
                                                    + f" {zp2.get_polynomial_name()} for r={r}, theta={theta},"
                                                    + f" amplitudes {ampls} calculated with some mistake")
    zern_surface = ZernPol.gen_zernikes_surface(coefficients=ampls, polynomials=[zp1, zp2])
    assert isinstance(zern_surface, tuple) and isinstance(zern_surface.ZernSurf, np.ndarray), ("Check gen_zernikes_surface()"
                                                                                               + " method output (tuple len=3)")
    assert len(zern_surface.ZernSurf.shape) == 2, "Check gen_zernikes_surface() method for output matrix shape"
    try:
        from matplotlib.figure import Figure
        fig = Figure()
        plotted_fig = ZernPol.plot_sum_zernikes_on_fig(coefficients=ampls, polynomials=[zp1, zp2],
                                                       figure=fig, zernikes_sum_surface=zern_surface)
        assert isinstance(plotted_fig, Figure) and plotted_fig.tight_layout, "Something wrong with the plotting function"
    except ModuleNotFoundError:
        assert False, "Install matplotlib for passing the test"
    # Test difference between direct (naive) implementation and using meshgrids implementation of sum of Zernikes
    pols = [ZernPol(osa=2), ZernPol(osa=4), ZernPol(osa=7), ZernPol(osa=10), ZernPol(osa=15)]
    ampls = [-0.85, 0.85, 0.24, -0.37, 1.0]; radii = np.arange(start=0.0, stop=1.0 + 0.05, step=0.05)
    thetas = np.arange(start=0.0, stop=2.0*np.pi + np.pi/10, step=np.pi/10)
    sum_pols_d = ZernPol.sum_zernikes(ampls, pols, radii, thetas, get_surface=True)
    sum_pols = ZernPol._sum_zernikes_meshgrid(ampls, pols, radii, thetas)
    assert abs(np.max(sum_pols_d - sum_pols)) < 1E-6, ("Sum of Zernikes are different between implementations "
                                                       + f" and have abs(max) = {abs(np.max(sum_pols_d - sum_pols))}")
    assert abs(np.min(sum_pols_d - sum_pols)) < 1E-6, ("Sum of Zernikes are different between implementations "
                                                       + f" and have abs(min) = {abs(np.min(sum_pols_d - sum_pols))}")


# Testing the calculation of polynomial values for edge cases
def test_pol_values_edge_cases():
    zp = ZernPol(osa=8)  # test polynomial, could be any
    # Combination list + float
    r = [0, 0.1, 0.2, 0.5]; theta = 0.2; values = zp.polynomial_value(r, theta)
    assert_flag = True
    if not isinstance(values, np.ndarray):
        assert_flag = False
    assert assert_flag, "Wrong calculation of combination list (r) and float (theta)"
    # Combination float + float
    r = 0.2; theta = 0.2; values = zp.polynomial_value(r, theta)
    assert_flag = True
    if not isinstance(values, float):
        assert_flag = False
    assert assert_flag, "Wrong calculation of combination float (r) and float (theta)"
    # Combination list + tuple, equal sizes
    r = [0, 0.1, 0.2, 0.5]; theta = (0, 0.4, 0.2, 0.1); values = zp.polynomial_value(r, theta)
    assert_flag = True
    if not isinstance(values, np.ndarray):
        assert_flag = False
    assert assert_flag, "Wrong calculation of combination list (r) and tuple (theta)"
    # Combination of lists with different sizes
    r = [0, 0.1, 0.2, 0.5]; theta = [0, 0.1, 0.2, 0.5, 0.8, 1.0]
    try:
        assert_flag = False; zp.polynomial_value(r, theta)
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong calculation of combination list (r) and list (theta) with not equal sizes"
    # Combination 1D and 2D arrays
    r = [0, 0.1, 0.2, 0.5]; theta = [[0, 0.3, 0.6, 0.9]]
    try:
        assert_flag = False; zp.polynomial_value(r, theta)
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong calculation of combination list (r) and list (theta) with not equal dimensions"
    # Tests for out of range r value
    r = [0, 0.1, 0.2, 1.00000001]; theta = [0, 0.3, 0.6, 0.9]
    try:
        assert_flag = False; zp.polynomial_value(r, theta)
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong calculation of using r > 1.0"
    r = -0.2
    try:
        assert_flag = False; zp.polynomial_value(r, theta)
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong calculation of using r < 0.0"
    # Testing piston calculation (edge case - starting polynomial, should return constant value = 1.0)
    zp = ZernPol(n=0, m=0)
    r = 0.25; theta = 1.0; value = zp.polynomial_value(r, theta); value_ex = zp.polynomial_value(r, theta, use_exact_eq=True)
    assert abs(1.0 - value) < 1E-6, f"Wrong calculated piston value (constant phase): {value}"
    assert abs(value_ex - value) < 1E-6, f"Wrong difference between tabular and exact equations for piston value: {value}, {value_ex}"
