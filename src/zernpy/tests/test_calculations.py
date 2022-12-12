# -*- coding: utf-8 -*-
"""
Test the implemented classes and methods by using pytest library.

pytest library avalaible on: https://docs.pytest.org/en/latest/contents.html
Note that for the running test without module installation, run this script by pytest module itself:
in command line navigate to this folder and run command pytest.

@author: Sergei Klykov
@licence: MIT

"""
import math
import time

# Importing the written in the modules test functions for letting pytest library automatic exploration
if __name__ != "__main__":
    from ..calculations.calc_zernike_pol import compare_radial_calculations
    from ..zernikepol import check_conformity, ZernPol


# Testing implemented equations for tabular R(m, n) from References by comparing with the exact ones
def test_tabular_orders():
    compare_radial_calculations(max_order=7)


# Testing implemented recursive equations for R(m, n) from References by comparing with the exact ones
def test_recursive_orders():
    compare_radial_calculations(max_order=21)


# Testing initialization of Zernike polynomials
def test_polynomials_initialization():
    check_conformity()


# Testing sum of Zernike polynomials
def test_sum_zernikes():
    zp1 = ZernPol(osa_index=1); zp2 = ZernPol(noll_index=2); ampls = [-0.5, 0.5]; theta = math.pi/3; r = 1.0
    sum_pols = ZernPol.sum_zernikes(coefficients=ampls, polynomials=[zp1, zp2], r=r, theta=theta)
    sum_pols_manual = math.cos(theta) - math.sin(theta)  # manual calculation of specified above sum
    assert abs(sum_pols - sum_pols_manual) < 1E-6, (f"Sum of Zernikes {zp1.get_polynomial()} and "
                                                    + " {zp2.get_polynomial()} for r={r}, theta={theta},"
                                                    + " amplitudes {ampls} calculated with some mistake")
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
        fig = ZernPol.plot_sum_zernikes_on_fig(coefficients=ampls, polynomials=[zp1, zp2],
                                               figure=fig, zernikes_sum_surface=zern_surface)
        plt.show(block=False); time.sleep(1.2)  # show the figure for 1.2 sec. during test run
    except ModuleNotFoundError:
        assert False, "Install matplotlib for passing the test"
