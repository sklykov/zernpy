# -*- coding: utf-8 -*-
"""
Run a few methods from 'zernpsf' as the main one for performing tests in IDE.

@author: Sergei Klykov, '@sklykov' on GitHub

@licence: MIT, @year: 2026

"""
import importlib
from contextlib import suppress
from pathlib import Path

import matplotlib
import numpy as np

# Explicit backend assignment for matplotlib - for compatibility between running configurations in Spyder and PyCharm IDEs
with suppress(ImportError):
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Import of developed version of package, install in the editable mode: pip install -e .
import zernpy.zernikepol
import zernpy.zernpsf

importlib.reload(zernpy.zernikepol)  # trick to guarantee local changes to be always used for run
importlib.reload(zernpy.zernpsf)

from zernpy.zernikepol import ZernPol
from zernpy.zernpsf import ZernPSF, force_get_psf_compilation

# %% Test as the main script
if __name__ == "__main__":
    plt.close("all")  # close all opened before figures
    wavelength_um = 0.55  # used below in a several calls
    check_other_pols = False; check_small_na_wl = False  # flag for checking some other polynomials PSFs
    check_airy = False; check_common_psf = False; check_io_kernel = False; check_parallel_calculation = False; check_test = False
    check_faster_airy = False; check_test_conditions = False; check_test_conditions2 = False; check_several_pols = False
    check_edge_conditions = False; test_acceleration_single_pol = False; test_acceleration_few_pol = False
    prepare_pic_readme = False  # for plotting the sum of polynomials produced profile
    test_io_few_pols = False; standard_path = Path.home().joinpath("Desktop")  # for saving json on the Desktop
    check_acceleration_flag = False  # for testing fallback calculation with the wrong flag for calculate PSF kernel
    check_init_several_pols = False  # check calculation several pol-s: Airy (Piston) + some of polynomial
    check_airy_patterns = False  # check the difference between calculated by equation for Airy pattern and diffraction integral
    check_precompilation = False  # checks precompilation
    check_cropping = False  # checks how kernel is cropped
    check_ver012 = True  # check used in README physical values for tuning empirical kernel size estimation + recalculate the results

    # Common PSF for testing
    if check_common_psf:
        force_get_psf_compilation(True)
        zpsf1 = ZernPSF(ZernPol(m=1, n=1)); zpsf2 = ZernPSF(ZernPol(m=-3, n=3)); ampl = -0.43
        zpsf2.set_physical_props(NA=0.95, wavelength=wavelength_um, expansion_coeff=ampl, pixel_physical_size=wavelength_um/5.05)
        zpsf1.set_physical_props(NA=0.5, wavelength=0.6, expansion_coeff=0.25, pixel_physical_size=wavelength_um/5.5)
        zpsf2.set_calculation_props(kernel_size=zpsf2.kernel_size, n_integration_points_r=250, n_integration_points_phi=320)
        zpsf1.set_calculation_props(kernel_size=zpsf2.kernel_size, n_integration_points_r=220, n_integration_points_phi=360)
        kernel3 = zpsf2.calculate_psf_kernel(suppress_warnings=False, verbose_info=True); zpsf2.plot_kernel("for loop")
        zpsf2.visualize_convolution()  # Visualize convolution on the disk image
        if check_io_kernel:
            # default_path = Path(__file__).parent.joinpath("saved_psfs").absolute()  # default path for storing JSON files
            zpsf2.save_json(overwrite=True, abs_path=standard_path)
            zpsf1.read_json(zpsf2.json_file_path)  # for testing reading and assigning values to ZernPSF class (substitution)
        if check_parallel_calculation:
            zpsf2.initialize_parallel_workers(); kernel2 = zpsf2.get_kernel_parallel()
            zpsf2.plot_kernel("parallel"); zpsf2.deinitialize_workers()

    # Another Zernike polynomial, big NA
    if check_other_pols:
        zpsf3 = ZernPSF(ZernPol(m=0, n=4))
        zpsf3.set_physical_props(NA=1.25, wavelength=wavelength_um, expansion_coeff=0.4, pixel_physical_size=wavelength_um/5.25)
        zpsf3.calculate_psf_kernel(suppress_warnings=False, verbose_info=True); zpsf3.plot_kernel()

    # Another Zernike polynomial, average to small NA and wavelength
    if check_small_na_wl:
        NA = 0.45; wavelength = 0.4; pixel_size = wavelength*0.2; ampl = -0.2
        zp2 = ZernPol(m=1, n=3); zpsf2 = ZernPSF(zp2)  # horizontal coma
        zpsf2.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf2.calculate_psf_kernel(normalized=True); zpsf2.plot_kernel()

    if check_airy:
        NA = 0.12; wavelength = 0.8; pixel_size = wavelength / 4.0; ampl = 0.1
        zp4 = ZernPol(m=0, n=0); zpsf4 = ZernPSF(zp4)  # piston for the Airy pattern
        zpsf4.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf4.calculate_psf_kernel(normalized=True); zpsf4.plot_kernel("Plus")
        zpsf4.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=-ampl, pixel_physical_size=pixel_size)
        zpsf4.calculate_psf_kernel(normalized=True); zpsf4.plot_kernel("Minus")

    if check_faster_airy:  # For set the test for pytest library
        NA = 0.35; wavelength = 0.55; pixel_size = wavelength / 3.05; ampl = -0.4
        zp4 = ZernPol(m=0, n=0); zpsf4 = ZernPSF(zp4)  # piston for the Airy pattern
        zpsf4.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf4.calculate_psf_kernel(normalized=True); zpsf4.plot_kernel()

    if check_test_conditions:
        NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 5.0; ampl = 0.55
        zp6 = ZernPol(m=0, n=2); zpsf6 = ZernPSF(zp6)  # defocus
        zpsf6.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)  # normal assignment
        zpsf6.set_calculation_props(kernel_size=zpsf6.kernel_size, n_integration_points_r=250, n_integration_points_phi=300)
        zpsf6.calculate_psf_kernel(normalized=True); zpsf6.plot_kernel()
    if check_test_conditions2:
        zp7 = ZernPol(m=1, n=3); zpsf7 = ZernPSF(zp7)  # horizontal coma
        NA = 0.4; wavelength = 0.4; pixel_size = wavelength / 3.2; ampl = 0.185  # Common physical properties
        zpsf7.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf7.calculate_psf_kernel(normalized=True); zpsf7.plot_kernel()

    # Test calculation of a PSF for several polynomials and I/O operations (see flags)
    if check_several_pols:
        zp1 = ZernPol(m=-2, n=2); zp2 = ZernPol(m=0, n=2); zp3 = ZernPol(m=2, n=2); pols = (zp1, zp2, zp3); coeffs = (-0.36, 0.25, 0.4)
        zpsf8 = ZernPSF(pols); zpsf8.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=coeffs, pixel_physical_size=0.5/4.5)
        composed_kernel = zpsf8.calculate_psf_kernel(normalized=True, verbose_info=True); zpsf8.plot_kernel()
        if test_io_few_pols:
            zpsf14 = ZernPSF(ZernPol(osa=19)); zpsf14.set_physical_props(NA=0.1, wavelength=0.4, expansion_coeff=0.82,
                                                                         pixel_physical_size=0.05)
            zpsf8.save_json(overwrite=True, abs_path=standard_path); zpsf14.read_json(zpsf8.json_file_path)  # save / read calculated kernel

    # Test some edge conditions - e.g., specifying 1 polynomial in a list with huge coefficient
    if check_edge_conditions:
        zp4 = ZernPol(m=3, n=3); pols2 = [zp4]; coeff = 5.1
        zpsf9 = ZernPSF(pols2); zpsf9.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=coeff, pixel_physical_size=0.5/4.75)
        zpsf9.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=(-0.67), pixel_physical_size=0.5/4.75)
        zpsf9.calculate_psf_kernel(normalized=True, verbose_info=True); zpsf9.plot_kernel()
    # Test acceleration by using numba library utilities
    if test_acceleration_single_pol:
        force_get_psf_compilation(); NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 4.6; ampl = -0.16
        zp6 = ZernPol(m=0, n=2); zpsf6 = ZernPSF(zp6)  # defocus
        zpsf6.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        kernel_acc = zpsf6.calculate_psf_kernel(normalized=True, accelerated=True, verbose_info=True); zpsf6.plot_kernel("Accelerated")
        kernel_norm = zpsf6.calculate_psf_kernel(normalized=True); zpsf6.plot_kernel("Normal")
    if test_acceleration_few_pol:
        force_get_psf_compilation()
        zp1 = ZernPol(m=-2, n=2); zp2 = ZernPol(m=0, n=2); zp3 = ZernPol(m=2, n=2); pols = (zp1, zp2, zp3); coeffs = (-0.12, 0.15, 0.1)
        zpsf8 = ZernPSF(pols); zpsf8.set_physical_props(NA=0.35, wavelength=0.5, expansion_coeff=coeffs, pixel_physical_size=0.5/1.5)
        composed_kernel = zpsf8.calculate_psf_kernel(normalized=True, verbose_info=True); zpsf8.plot_kernel("Normal")
        composed_kernel_acc = zpsf8.calculate_psf_kernel(normalized=True, verbose_info=True, accelerated=True)
        zpsf8.plot_kernel("Accelerated")
    if prepare_pic_readme:
        force_get_psf_compilation(verbose_report=True)
        zp1 = ZernPol(m=-1, n=3); zp2 = ZernPol(m=2, n=4); zp3 = ZernPol(m=0, n=4); pols = (zp1, zp2, zp3); coeffs = (0.5, 0.21, 0.15)
        zpsf_pic = ZernPSF(pols); zpsf_pic.set_physical_props(NA=0.65, wavelength=0.6, expansion_coeff=coeffs, pixel_physical_size=0.6/5.0)
        zpsf_pic.calculate_psf_kernel(normalized=True, verbose_info=True, accelerated=True)
        zpsf_pic.plot_kernel("Vert. Coma Vert. 2nd Astigmatism Spherical")
    if check_acceleration_flag:
        zp16 = ZernPol(m=-1, n=3); zp18 = ZernPol(m=2, n=4); pols10 = (zp16, zp18); coeffs10 = (-0.1, 0.13); zpsf_acc = ZernPSF(pols10)
        zpsf_norm = ZernPSF(pols10); zpsf_acc.set_physical_props(NA=0.43, wavelength=0.6, expansion_coeff=coeffs10,
                                                                 pixel_physical_size=0.6/3.0)
        zpsf_norm.set_physical_props(NA=0.43, wavelength=0.6, expansion_coeff=coeffs10, pixel_physical_size=0.6/3.0)
        kern_acc = zpsf_acc.calculate_psf_kernel(normalized=True, accelerated=True)
        kern_norm = zpsf_norm.calculate_psf_kernel(normalized=True)
        kern_diff = np.round(kern_acc - kern_norm, 9)  # for checking the difference in calculations

    if check_test:
        pols = (ZernPol(osa=10), ZernPol(osa=15)); coeffs = (0.28, -0.33); NA = 0.35; wavelength = 0.55
        zpsf = ZernPSF(pols); zpsf.set_physical_props(NA, wavelength, expansion_coeff=coeffs, pixel_physical_size=wavelength / 3.5)
        zpsf.set_calculation_props(kernel_size=25, n_integration_points_r=200, n_integration_points_phi=180)
        psf_kernel = zpsf.calculate_psf_kernel(normalized=False); zpsf.plot_kernel()

    if check_init_several_pols:
        force_get_psf_compilation(True)
        zpsf30 = ZernPSF(zernpol=(ZernPol(osa=1), ZernPol(osa=5)))
        try:
            zpsf30.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=0.5, pixel_physical_size=0.5/5.0)
        except ValueError:
            print("Check for 1 ampl. and 2 pol-s passed")  # as expected, transfer to test function
        zpsf31 = ZernPSF(zernpol=(ZernPol(osa=0), ZernPol(m=-1, n=3)))
        zpsf31.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=[1.5, 0.24], pixel_physical_size=0.5/3.8)
        # zpsf31.calculate_psf_kernel(verbose_info=True, accelerated=False); zpsf31.plot_kernel("Not accelerated 2 pol-s")
        # kernel_not_acc = np.copy(zpsf31.kernel)
        zpsf31.calculate_psf_kernel(verbose_info=True, accelerated=True); zpsf31.plot_kernel("Airy 1.5")
        kernel_acc = np.copy(zpsf31.kernel)
        # zpsf31.kernel = np.abs(kernel_acc - kernel_not_acc); zpsf31.plot_kernel("Diff. 2 pol-s")
        zpsf31.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=[-1.5, 0.24], pixel_physical_size=0.5/3.8)
        zpsf31.calculate_psf_kernel(verbose_info=True, accelerated=True); zpsf31.plot_kernel("Airy -1.5")
        kernel_neg_acc = np.copy(zpsf31.kernel)
        zpsf31.kernel = np.abs(kernel_acc - kernel_neg_acc); zpsf31.plot_kernel("Diff. neg. pos. Airy + Coma")

    if check_airy_patterns:
        force_get_psf_compilation(verbose_report=True)
        zpsf40 = ZernPSF(zernpol=ZernPol(osa=0))
        zpsf40.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=3.0, pixel_physical_size=0.5/4.0)
        zpsf40.calculate_psf_kernel(verbose_info=True, accelerated=False, normalized=False); zpsf40.plot_kernel("Not accelerated Airy")
        kernel_not_acc = np.copy(zpsf40.kernel)
        zpsf40.calculate_psf_kernel(verbose_info=True, accelerated=True, normalized=False); zpsf40.plot_kernel("Accelerated Airy")
        kernel_acc = np.copy(zpsf40.kernel)
        zpsf40.kernel = np.abs(kernel_acc - kernel_not_acc); zpsf40.plot_kernel("Diff. Airy")

    if check_precompilation:
        force_get_psf_compilation(True)
        zpsf50 = ZernPSF(zernpol=(ZernPol(m=1, n=3)))
        zpsf50.set_physical_props(NA=1.25, wavelength=0.52, expansion_coeff=0.5, pixel_physical_size=0.5/4.85)
        zpsf50.calculate_psf_kernel(verbose_info=True, accelerated=True, normalized=True); zpsf50.plot_kernel()

    if check_cropping:
        zpsf60 = ZernPSF(zernpol=(ZernPol(m=0, n=4)))
        zpsf60.set_physical_props(NA=1.25, wavelength=0.5, expansion_coeff=0.47, pixel_physical_size=0.5/5.0)
        zpsf60.calculate_psf_kernel(accelerated=True, verbose_info=True); zpsf60.plot_kernel("Not Cropped")
        original_kernel = np.copy(zpsf60.kernel)
        zpsf60.crop_kernel(min_part_of_max=0.025); zpsf60.plot_kernel("Cropped"); cropped_kernel = np.copy(zpsf60.kernel)
        print("Original kernel shape:", original_kernel.shape, "\nCropped kernel shape:", cropped_kernel.shape)

    if check_ver012:
        zpsf = ZernPSF(ZernPol(m=0, n=0))  # Airy - baseline, should be calculated using exact equation
        NA = 0.4; wavelength = 0.55; pixel_physical_size = 0.24*wavelength; expansion_coeff = -1.0  # example of physical properties
        zpsf.set_physical_props(NA, wavelength, expansion_coeff, pixel_physical_size)  # provide physical properties of the system
        zpsf.set_calculation_props(kernel_size=21, n_integration_points_r=200, n_integration_points_phi=360)
        kernel = zpsf.calculate_psf_kernel(normalized=True); zpsf.plot_kernel()
        zpsf = ZernPSF(ZernPol(m=1, n=3))  # horizontal coma - 0.0 - coincidence with the Airy pattern
        NA = 0.4; wavelength = 0.55; pixel_physical_size = 0.24*wavelength; expansion_coeff = -0.0  # example of physical properties
        zpsf.set_physical_props(NA, wavelength, expansion_coeff, pixel_physical_size)  # provide physical properties of the system
        kernel = zpsf.calculate_psf_kernel(normalized=True);  zpsf.plot_kernel()
        zpsf = ZernPSF(ZernPol(m=1, n=3))  # horizontal coma
        NA = 0.4; wavelength = 0.55; pixel_physical_size = 0.24*wavelength; expansion_coeff = -0.1  # example of physical properties
        zpsf.set_physical_props(NA, wavelength, expansion_coeff, pixel_physical_size)  # provide physical properties of the system
        kernel = zpsf.calculate_psf_kernel(normalized=True); zpsf.plot_kernel()
        # Set of Zernike polynomials
        zp1 = ZernPol(m=1, n=3); zp2 = ZernPol(m=2, n=4); zp3 = ZernPol(m=0, n=4); pols = (zp1, zp2, zp3); coeffs = (-0.1, 0.0, 0.0)
        NA = 0.4; wavelength = 0.55; pixel_physical_size = 0.24*wavelength
        zpsf_pic = ZernPSF(pols); zpsf_pic.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=coeffs,
                                                              pixel_physical_size=pixel_physical_size)
        zpsf_pic.calculate_psf_kernel(); zpsf_pic.plot_kernel("Sum of Polynomials Profile")
        # Cropping section
        zpsf = ZernPSF(zernpol=(ZernPol(m=0, n=4)))  # Spherical aberration
        zpsf.set_physical_props(NA=1.25, wavelength=0.5, expansion_coeff=0.25, pixel_physical_size=0.5/5.0)
        zpsf.calculate_psf_kernel(accelerated=True, verbose_info=True); zpsf.plot_kernel("Not Cropped")
        zpsf.crop_kernel(min_part_of_max=0.025)  # rows and columns containing less than 2.5% of kernel max are cropped out
        zpsf.plot_kernel("Cropped")

    # Cleaning up used flags for preventing of Variable Explorer flooding
    del check_other_pols, check_small_na_wl, check_airy, check_common_psf, check_io_kernel, check_parallel_calculation
    del check_test, check_faster_airy, check_test_conditions, check_test_conditions2, check_several_pols, check_edge_conditions
    del test_acceleration_single_pol, test_acceleration_few_pol, prepare_pic_readme, test_io_few_pols, check_acceleration_flag
    del check_init_several_pols, check_airy_patterns, check_precompilation, check_cropping, check_ver012
