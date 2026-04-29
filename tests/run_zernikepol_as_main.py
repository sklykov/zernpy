# -*- coding: utf-8 -*-
"""
Run a few methods from 'zernikepol' as the main one for performing tests in IDE.

@author: Sergei Klykov, '@sklykov' on GitHub

@licence: MIT, @year: 2026

"""
# %% Imports and checking its validity
import importlib
import time
from contextlib import suppress
from typing import Tuple

import matplotlib
import numpy as np

# Explicit backend assignment for matplotlib - for compatibility between running configurations in Spyder and PyCharm IDEs
with suppress(ImportError):   # will be thrown in the environment doesn't contain Qt-like library
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Import of developed version of package, install in the editable mode: pip install -e .
import zernpy.zernikepol

importlib.reload(zernpy.zernikepol)  # trick to guarantee local changes to be always used for run
from zernpy.zernikepol import ZernPol, fit_polynomials, generate_phases_image, generate_random_phases, zernikes_surface


# %% Test functions for the external call
def compare_performances(min_order: int, max_order: int) -> Tuple[int, int, str]:
    """
    Compare performances of radial polynomials calculation by using recursive and exact equations.

    Comparison achieved by simple measuring of time in msec needed for calculation of all radial
    polynomials from minimal radial order up to maximum radial order returned as tuple.

    Parameters
    ----------
    min_order : int
        Minimum radial order of used polynomials (n).
    max_order : int
        Maximum radial order of used polynomials (n).

    Returns
    -------
    tuple
        Composed by time for recursive calculation and time for exact calculation.

    """
    # Generation of orders in OSA/ANSI indexing scheme for initializing Zernike polynomials
    zernpols = []
    for order in range(min_order, max_order+1):
        m = -order; n = order
        zernpols.append(ZernPol(m=m, n=n))
        for _ in range(0, order):
            m += 2; zernpols.append(ZernPol(m=m, n=n))
    # Generation numpy array with radii
    n_points = 251
    test_r = np.zeros(shape=(n_points, ))
    for i in range(n_points):
        test_r[i] = i/(n_points-1)
    test_r = np.round(test_r, 6)
    # Measuring performance of radial polynomials calculations using recursive implementation
    t1 = time.perf_counter()
    for polynomial in zernpols:
        polynomial.radial(test_r)  # calculate radial polynomials over vector of radii
    t2 = time.perf_counter()
    t_recursive_ms = round(1000*(t2-t1), 3)
    # Measuring performance of radial polynomials calculations using exact implementation
    t1 = time.perf_counter()
    for polynomial in zernpols:
        polynomial.radial(test_r, use_exact_eq=True)  # calculate radial polynomials over vector of radii
    t2 = time.perf_counter()
    t_exact_ms = round(1000*(t2-t1), 3)
    return t_recursive_ms, t_exact_ms, f"Used polynomials: {i}", f"Radii: {n_points}"


def _estimate_high_order_calc_times():
    """
    Estimate slowing down of radial polynomial calculation with increasing of radial order.

    Returns
    -------
    None.

    """
    r = 0.55
    high_order_pols = [ZernPol(m=-2, n=46), ZernPol(m=1, n=47), ZernPol(m=2, n=48), ZernPol(m=-1, n=49),
                       ZernPol(m=2, n=50), ZernPol(m=1, n=51), ZernPol(m=-2, n=52)]
    times = []
    # Single polynomials values
    for pol in high_order_pols:
        # calculate radial polynomials over vector of radii
        t1 = time.perf_counter(); pol.radial(r); t2 = time.perf_counter()
        times.append(f"{pol.get_mn_orders()}: {int(round(1000*(t2-t1), 0))} ms")
    # Delete polynomials with too high orders for derivatives calculates
    high_order_pols.pop(len(high_order_pols)-1); high_order_pols.pop(len(high_order_pols)-1)
    high_order_pols.pop(len(high_order_pols)-1)
    print(times); times = []
    # Single derivative polynomials values
    for pol in high_order_pols:
        # calculate derivatives of radial polynomials over vector of radii
        t1 = time.perf_counter(); pol.radial_dr(r); t2 = time.perf_counter()
        times.append(f"Deriv. {pol.get_mn_orders()}: {int(round(1000*(t2-t1), 0))} ms")
    print(times)


# %% Tests
if __name__ == "__main__":
    _test_plots = True  # regulates testing of plotting various plots
    _test_calculations = False  # regulates tests below concerning calculations

    # Testing plotting, the plots will be opened in the additional pop-up windows
    if _test_plots:
        plt.close("all")  # close all previously opened plots
        t1 = time.perf_counter(); zp = ZernPol(m=0, n=2); ZernPol.plot_profile(zp, color_map="jet", show_title=True)  # basic plot
        t2 = time.perf_counter(); print("Plotting of 1 non-zero polynomial takes ms: ", int(round(1000*(t2-t1), 0)))
        coordinates = ZernPol.gen_polar_coordinates(r_step=0.005)
        zp = ZernPol(m=-10, n=30); ZernPol.plot_profile(zp, color_map="jet", show_title=False, polar_coordinates=coordinates)  # high order plot
        zp = ZernPol(m=0, n=0); ZernPol.plot_profile(zp, color_map="turbo", show_title=True)  # plot of piston polynomial

        # Testing 3D surface plotting
        ZernPol.plot_profile(ZernPol(m=0, n=2), color_map="viridis", projection="3d")

        # Testing 3D figure plotting on the externally initialized Figure class
        fig3d = plt.figure(figsize=(6.8, 6.8))
        zern_surface = ZernPol.gen_zernikes_surface([1.0], [ZernPol(m=0, n=2)], equal_n_coordinates=True, n_points=400)
        ZernPol.plot_sum_zernikes_on_fig(figure=fig3d, use_defaults=False, zernikes_sum_surface=zern_surface,
                                         show_range=True, color_map="magma", projection="3D")
        fig3d2 = plt.figure(figsize=(5.8, 5.8))
        ZernPol.plot_sum_zernikes_on_fig(figure=fig3d2, coefficients=[1.0], polynomials=[ZernPol(m=0, n=2)],
                                         show_range=True, color_map="bwr", projection="3D")

        # Testing accelerated plotting / sum calculation
        fig3 = plt.figure(figsize=(3, 3))
        t1 = time.perf_counter(); n_pols = 31; polynomials = []; coefficients = [0.0]*n_pols
        for i in range(n_pols):
            polynomials.append(ZernPol(osa=58+i))
        coefficients[0] = 1.0  # only 1st polynomial will be plotted
        fig3 = ZernPol.plot_sum_zernikes_on_fig(figure=fig3, coefficients=coefficients, polynomials=polynomials,
                                                show_range=False, color_map="turbo")
        fig3.subplots_adjust(0, 0, 1, 1)
        t2 = time.perf_counter(); print("Plotting of 1 non-zero and 30 zero pol-s takes ms: ", int(round(1000*(t2-t1), 0)))

        # Tests with generation / restoring Zernike profiles (phases images)
        phases_image, polynomials_ampls, polynomials = generate_random_phases(img_height=301, img_width=321)
        plt.figure(); plt.axis("off"); plt.imshow(phases_image, cmap="jet"); plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)
        polynomials_amplitudes, cropped_img = fit_polynomials(phases_image, polynomials, return_cropped_image=True,
                                                              strict_circle_border=False, crop_radius=1.0)
        plt.figure(); plt.axis("off"); plt.imshow(cropped_img, cmap="jet")
        plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)

        # Updated test of fitting including piston polynomial
        height = 500; width = 481; crop_r = 1.0; strict_border = True; pols_coeffs = [-0.75, 0.86, 0.41]; fig4 = plt.figure(figsize=(4, 4))
        polynomials = [ZernPol(osa=0), ZernPol(m=0, n=2), ZernPol(m=-3, n=3)]; rs, angles = ZernPol.gen_polar_coordinates()
        phase_profile = ZernPol.sum_zernikes(coefficients=pols_coeffs, polynomials=polynomials, r=rs, theta=angles, get_surface=True)
        # Below - plotting specified polynomials on the polar coordinates
        ZernPol.plot_sum_zernikes_on_fig(figure=fig4, use_defaults=False, zernikes_sum_surface=zernikes_surface(phase_profile, rs, angles),
                                         color_map="jet")
        # Below - generate phases image with the cartesian coordinates
        phases_image2 = generate_phases_image(polynomials=tuple(polynomials), polynomials_amplitudes=tuple(pols_coeffs),
                                              img_height=height, img_width=width)
        plt.figure(); plt.axis("off"); im = plt.imshow(phases_image2, cmap="jet"); plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)
        plt.colorbar(mappable=im)
        # Below - fitting procedure on the provided phases image
        polynomials_amplitudes2, cropped_img2 = fit_polynomials(phases_image2, polynomials, return_cropped_image=True,
                                                                strict_circle_border=strict_border, crop_radius=crop_r)
        print("Difference between used amplitudes and fitted ones:", np.asarray(pols_coeffs) - polynomials_amplitudes2)
        plt.figure(); plt.axis("off"); im = plt.imshow(cropped_img2, cmap="jet"); plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)
        plt.colorbar(mappable=im)

        plt.show()  # show all images created by plt.figure() calls

    # Testing calculations and their performance comparison
    if _test_calculations:
        # Simple test of two concepts of calculations - exact and recursive equations
        z = ZernPol(n=30, m=-2); print("Diff. between recursive and exact equations:",
                                       round(z.radial(0.85) - z.radial(0.85, use_exact_eq=True), 9))
        z = ZernPol(n=32, l=0); print("Diff. between recursive and exact equations:",
                                      round(z.radial(0.35) - z.radial(0.35, use_exact_eq=True), 9))
        r = 0.955; theta = np.pi/8; z = ZernPol(osa=55)
        print("Diff. between recursive and exact equations:",
              round(z.polynomial_value(r, theta) - z.polynomial_value(r, theta, use_exact_eq=True), 9))
        z = ZernPol(n=35, l=-1); print("Diff. between recursive & exact eq-s for derivatives:",
                                       round(z.radial_dr(0.78) - z.radial_dr(0.78, use_exact_eq=True), 9))
        z = ZernPol(n=38, m=-2); print("Diff. between recursive & exact eq-s for derivatives:",
                                       round(z.radial_dr(0.9) - z.radial_dr(0.9, use_exact_eq=True), 9))
        # Compare performances
        print("Tabular (10th order) / exact calc. times:", compare_performances(1, 10))
        print("Recursive / exact calc. times for high orders:", compare_performances(12, 40))
        # Statement below producing expected warnings, it's used for performance estimation
        _estimate_high_order_calc_times()
