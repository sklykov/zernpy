# -*- coding: utf-8 -*-
"""
Test the implemented calculation functions for module calc_zernike_pol and ZernPol static methods by using pytest library.

pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running this test, it's enough to run the command "pytest" from the repository location.

@author: Sergei Klykov
@licence: MIT

"""
import math
import time
import numpy as np

# Importing the written in the modules test functions for letting pytest library their automatic exploration
if __name__ != "__main__":
    from ..calculations.calc_zernike_pol import compare_radial_calculations, compare_radial_derivatives
    from ..zernikepol import ZernPol


# Testing implemented equations for tabular R(m, n) from References by comparing with the exact ones (with factorials)
def test_tabular_orders():
    compare_radial_calculations(max_order=10)


# Testing implemented tabular and recursive equations for R(m, n) by comparing with the exact ones (with factorials)
def test_recursive_orders():
    compare_radial_calculations(max_order=21)


# Testing derived equations for derivatives dR(m, n)/dr by comparing with the exact ones (with factorials)
def test_tabular_derivatives():
    compare_radial_derivatives(max_order=10)


# Testing recursive and derived equations for derivatives dR(m, n)/dr by comparing with the exact ones (with factorials)
def test_recursive_derivatives():
    compare_radial_derivatives(max_order=19)


# Testing sum of Zernike polynomials
def test_sum_zernikes():
    zp1 = ZernPol(osa_index=1); zp2 = ZernPol(noll_index=2); ampls = [-0.5, 0.5]; theta = math.pi/3; r = 1.0
    sum_pols = ZernPol.sum_zernikes(coefficients=ampls, polynomials=[zp1, zp2], r=r, theta=theta)
    sum_pols_manual = math.cos(theta) - math.sin(theta)  # manual calculation of specified above sum
    assert abs(sum_pols - sum_pols_manual) < 1E-6, (f"Sum of Zernikes {zp1.get_polynomial_name()} and "
                                                    + f" {zp2.get_polynomial_name()} for r={r}, theta={theta},"
                                                    + f" amplitudes {ampls} calculated with some mistake")
    zern_surface = ZernPol.gen_zernikes_surface(coefficients=ampls, polynomials=[zp1, zp2])
    try:
        import numpy as np
    except ModuleNotFoundError:
        assert False, "Install numpy for passing the test"
    assert isinstance(zern_surface, tuple) and isinstance(zern_surface.ZernSurf, np.ndarray), ("Check gen_zernikes_surface()"
                                                                                               + " method output (tuple len=3)")
    assert len(zern_surface.ZernSurf.shape) == 2, "Check gen_zernikes_surface() method for output matrix shape"
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ZernPol.plot_sum_zernikes_on_fig(coefficients=ampls, polynomials=[zp1, zp2],
                                         figure=fig, zernikes_sum_surface=zern_surface)
        plt.show(block=False); time.sleep(1.05)  # show the figure for 1.2 sec. during test run
    except ModuleNotFoundError:
        assert False, "Install matplotlib for passing the test"


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
        assert_flag = False; values = zp.polynomial_value(r, theta)
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong calculation of combination list (r) and list (theta) with not equal sizes"
    # Combination 1D and 2D arrays
    r = [0, 0.1, 0.2, 0.5]; theta = [[0, 0.3, 0.6, 0.9]]
    try:
        assert_flag = False; values = zp.polynomial_value(r, theta)
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong calculation of combination list (r) and list (theta) with not equal dimensions"
    # Out of range value for r
    r = [0, 0.1, 0.2, 1.00000001]; theta = [0, 0.3, 0.6, 0.9]
    try:
        assert_flag = False; values = zp.polynomial_value(r, theta)
    except ValueError:
        assert_flag = True
    assert assert_flag, "Wrong calculation of using r > 1.0"
