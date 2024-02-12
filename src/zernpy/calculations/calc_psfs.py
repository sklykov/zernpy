# -*- coding: utf-8 -*-
"""
Calculation and plotting of associated with polynomials PSFs.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from pathlib import Path
import warnings
from math import sqrt, pi
from zernpy import ZernPol
import time
from skimage.morphology import disk  # only for tests, won't be used as part as export functions

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calc_zernike_pol import define_orders
    from calc_psfs_check import save_psf, read_psf, convolute_img_psf
else:
    from .calc_zernike_pol import define_orders
    from .calc_psfs_check import save_psf, read_psf, convolute_img_psf


# %% Module parameters
__docformat__ = "numpydoc"
um_char = "\u00B5"  # Unicode char code for micrometers
lambda_char = "\u03BB"  # Unicode char code for lambda (wavelength)
pi_char = "\u03C0"  # Unicode char code for pi
# print(um_char, lambda_char, pi_char)  # check the codes for characters from above


# %% Reference
def airy_ref_pattern(r: float):
    """
    Return Airy pattern radial function J1(r)/r.

    Parameters
    ----------
    r : float
        Radial distance (for an objective should be k*NA*r).

    Returns
    -------
    float
        J1(r)/r function.

    """
    r = round(r, 12)
    if r == 0.0:
        ratio = jv(1, 1E-11)/1E-11
    else:
        ratio = jv(1, r)/r
    return 4.0*pow(ratio, 2)


# %% PSF pixel value calc.
def diffraction_integral_r(zernike_pol: ZernPol, alpha: float, phi: float, p, theta: float, r: float) -> np.array:
    """
    Diffraction integral function for the formed image point (see the references as the sources of the equation).

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    alpha : float
        Amplitude of the polynomial (RMS).
    phi : float
        Angle on the pupil (entrance pupil of microobjective) coordinates (for integration).
    p : np.array or float
        Integration interval on the pupil (entrance pupil of microobjective) radius or radius as float number.
    theta : floats
        Angle on the image coordinates.
    r : float
        Radius on the image coordinates.

    References
    ----------
    [1] Principles of Optics, by M. Born and E. Wolf, 4 ed., 1968
    [2] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf

    Returns
    -------
    numpy.ndarray
        Values of the diffraction integral.

    """
    phase_arg = (alpha*zernike_pol.polynomial_value(p, phi) - r*p*np.cos(phi - theta))*1j
    return np.exp(phase_arg)*p


def radial_integral(zernike_pol: ZernPol, r: float, theta: float, phi: float, alpha: float, n_int_r_points: int) -> complex:
    """
    Make integration of the diffraction integral on the radius of the entrance pupil.

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    phi : float
        Angle on the pupil coordinates.
    alpha : float
        Amplitude of the polynomial.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).

    Returns
    -------
    complex
        Complex amplitude of the field as result of integration on the pupil radius coordinate.

    """
    # Integration on the pupil angle. Vectorized form of the trapezoidal rule
    h_p = 1.0/n_int_r_points; p = np.arange(start=h_p, stop=1.0, step=h_p)
    fa = diffraction_integral_r(zernike_pol, alpha, phi, 0.0, theta, r)
    fb = diffraction_integral_r(zernike_pol, alpha, phi, 1.0, theta, r)
    ang_int = np.sum(diffraction_integral_r(zernike_pol, alpha, phi, p, theta, r)) + 0.5*(fa + fb)
    return h_p*ang_int


def get_psf_point_r(zernike_pol: ZernPol, r: float, theta: float, alpha: float, n_int_r_points: int, n_int_phi_points: int) -> float:
    """
    Get the point for calculation of PSF depending on the image polar coordinates.

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    alpha : float, optional
        Amplitude of the polynomial.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).
    n_int_phi_points : int
        Number of integration points used for integration on the polar angle of the entrance pupil (from the range [0.0, 2pi]).

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    # Integration on the pupil radius using Simpson equation
    h_phi = 2.0*pi/n_int_phi_points; even_sum = 0.0j; odd_sum = 0.0j
    for i in range(2, n_int_phi_points-2, 2):
        phi = i*h_phi; even_sum += radial_integral(zernike_pol, r, theta, phi, alpha, n_int_r_points)
    for i in range(1, n_int_phi_points-1, 2):
        phi = i*h_phi; odd_sum += radial_integral(zernike_pol, r, theta, phi, alpha, n_int_r_points)
    yA = radial_integral(zernike_pol, r, theta, 0.0, alpha, n_int_r_points)
    yB = radial_integral(zernike_pol, r, theta, 2.0*pi, alpha, n_int_r_points)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


# %% PSF kernel calc.
def get_psf_kernel(zernike_pol, len2pixels: float, alpha: float, wavelength: float, NA: float, n_int_r_points: int = 320,
                   n_int_phi_points: int = 300, show_kernel: bool = False, fig_title: str = None, normalize_values: bool = False,
                   airy_pattern: bool = False, kernel_size: int = 0) -> np.ndarray:
    f"""
    Calculate centralized matrix with the PSF mask values.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or isntance of the ZernPol(...) class.
        Required for calculation Zernike polynomial.
    len2pixels : float
        Relation between length in physical units (the same as the provided wavelength) and pixels.
    alpha : float
        Zernike amplitude (the expansion coefficient) in physical units used for the wavelength specification (e.g., {um_char}).
        Note that the normalized Zernike polynomials are used, so its coefficient is normalized to the specified wavelength.
    wavelength : float
        Wavelength ({lambda_char}) in physical units (e.g., {um_char}) of the light used for calculations (in imaging).
    NA : float
        Objective property.
    n_int_r_points : int, optional
        Number of points used for integration on the unit pupli radius from the range [0.0, 1.0]. The default is 320.
    n_int_phi_points : int, optional
        Number of points used for integration on the unit pupli angle from the range [0.0, 2{pi_char}]. The default is 300.
    show_kernel : bool, optional
        Plot the calculated kernel interactively. The default is True.
    fig_title : str, optional
        Custom figure title. The default is None.
    normalize_values : bool, optional
        Normalize all values in the sense that the max kernel value = 1.0. The default is False.
    airy_pattern : bool, optional
        Plot the Airy pattern for the provided parameters. The default is False.
    kernel_size : int, optional
        Custom kernel size, if not provided, then the the size will be estimated based on the parameters. The default is 0.

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.

    """
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    # Check and initialize Zernike polynomial if provided only orders
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    if not isinstance(zernike_pol, ZernPol):
        zernike_pol = ZernPol(m=m, n=n)
    # Convert provided absolute value of Zernike expansion coefficient (in um) into fraction of wavelength
    alpha /= wavelength
    k = 2.0*pi/wavelength  # Calculate angular frequency (k)
    # Estimation of the kernel size, empirical estimation of the sufficient size for the kernel
    if kernel_size < 3:
        if m == 0 and n == 0:
            multiplier = 1.0
        else:
            multiplier = 1.25*sqrt(n)
        if alpha > 1.0:
            multiplier *= sqrt(alpha)
        size = int(round(multiplier/len2pixels, 0)) + 1  # Note that with the amplitude growth, it requires to grow as well
    else:
        size = kernel_size
    # Auto definition of the required PSF size is complicated for the different PSFs forms (e.g., vertical coma with different amplitudes)
    # Make kernel with odd sizes for precisely centering the kernel (in the center of an image)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros(shape=(size, size)); i_center = size//2; j_center = size//2
    # Print note about calculation length
    if size > 21:
        if abs(n_int_phi_points - 300) < 40 and abs(n_int_r_points - 320) < 50:
            print(f"Note that the calculated kernel size: {size}x{size}. Estimated calculation time: {int(round(size*size*40/1000, 0))} sec.")
        else:
            print(f"Note that the calculated kernel size: {size}x{size}. Calculation may take from several dozends of seconds to minutes")
    # Check that the calibration coefficient is sufficient for calculation
    pixel_size_nyquist = 0.5*0.61*wavelength/NA
    if len2pixels > pixel_size_nyquist:
        __warn_message = f"Provided calibration coefficient {len2pixels} {um_char}/pixels isn't sufficient enough"
        __warn_message += f" (defined by the relation between Nyquist freq. and the optical resolution: 0.61{lambda_char}/NA)"
        warnings.warn(__warn_message)
    # Calculate the PSF kernel for usage in convolution operation
    # mean_time_integration = 0.0; n = 0
    for i in range(size):
        for j in range(size):
            t1 = time.perf_counter(); n += 1
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))  # in pixels
            # Convert pixel distance in the required k*NA*pixel_dist*calibration coefficient
            distance = k*NA*len2pixels*pixel_dist  # conversion from pixel distance into phase multiplier in the diffraction integral
            # The PSF also has the angular dependency, not only the radial one
            theta = np.arctan2((i - i_center), (j - j_center))
            theta += np.pi  # shift angles to the range [0, 2pi]
            # The scaling below is not needed because the Zernike polynomial is scaled as the RMS values
            if not airy_pattern:
                kernel[i, j] = get_psf_point_r(zernike_pol, distance, theta, alpha, n_int_r_points, n_int_phi_points)
            else:
                kernel[i, j] = airy_ref_pattern(distance)
            # t2 = time.perf_counter(); mean_time_integration += round(1000.0*(t2-t1), 0)
    # print("Mean integration time, ms:", round(mean_time_integration/n, 0))
    # Normalize all values in kernel to bring the max value to 1.0
    if normalize_values:
        kernel /= np.max(kernel)
    # Provide warning in the case if kernel size not sufficient for representation of the calculated kernel
    if kernel[0, 0] > np.max(kernel)/100:
        __warn_message = f"The calculated size for PSF ({size}) isn't sufficient for its proper representation"
        warnings.warn(__warn_message)
    # Plotting the calculated kernel
    if show_kernel:
        if airy_pattern:
            fig_title = "Airy pattern"
        if fig_title is not None and len(fig_title) > 0:
            plt.figure(fig_title, figsize=(6, 6))
        else:
            plt.figure(f"{(m, n)} {zernike_pol.get_polynomial_name(True)}: {round(alpha, 2)}*wavelength", figsize=(6, 6))
        plt.imshow(kernel, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()
    return kernel


# %% Define standard exports from this module
__all__ = ['get_psf_kernel', 'save_psf', 'read_psf', 'convolute_img_psf']

# %% Tests
if __name__ == '__main__':
    plt.ion(); plt.close('all')  # close all plots before plotting new ones
    # Physical parameters of a system (an objective)
    wavelength = 0.55  # in micrometers
    NA = 0.95  # microobjective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length for an image)
    # Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
    resolution = 0.61*wavelength/NA  # ultimate theoretical physical resolution of an objective
    pixel_size_nyquist = 0.5*resolution  # Nyquist resolution needed for using theoretical physical resolution above
    pixel_size = 0.65*pixel_size_nyquist  # the relation between um / pixels for calculating the coordinate in physical units for each pixel
    # Note that pixel_size is only estimated here to be sufficient. It should be exchanged to the physical one as the input for the function

    # Flags for performing tests
    check_zero_case = False  # checking that integral equation is corresponding to the Airy pattern (zero case)
    check_sign_coeff = False  # checking the same amplitude applied for the same polynomial (trefoil)
    check_various_pols = False  # checking the shape of some Zernike polynomials for comparing with the link below
    # PSF shapes: https://en.wikipedia.org/wiki/Zernike_polynomials#/media/File:ZernikeAiryImage.jpg
    check_warnings = False  # flag for checking the warning producing
    check_io = False  # check save / read kernels
    show_convolution_results = True  # check result of convolution of several kernel with the disk

    # Definition of some Zernike polynomials
    pol1 = (0, 0); pol2 = (-1, 1); pol3 = (0, 2); pol4 = (-2, 2); pol5 = (-3, 3); pol6 = (2, 2); pol7 = (-1, 3); pol8 = (0, 4); pol9 = (-4, 4)

    if check_zero_case:
        kern_zc = get_psf_kernel(pol1, pixel_size, 0.5, wavelength, NA)
        kern_zc_ref = get_psf_kernel(pol1, pixel_size, 0.5, wavelength, NA, airy_pattern=True)
        diff = kern_zc_ref - kern_zc; plt.figure("Difference Airy and Piston", figsize=(6, 6)); plt.imshow(diff, cmap=plt.cm.viridis, origin='upper')

    if check_sign_coeff:
        kern_sign = get_psf_kernel(pol5, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=False)
        kern_sign_n = get_psf_kernel(pol5, len2pixels=pixel_size, alpha=-0.5, wavelength=wavelength, NA=NA, normalize_values=False)

    if check_various_pols:
        kern_def = get_psf_kernel(pol3, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True, show_kernel=True)
        kern_ast = get_psf_kernel(pol6, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True, show_kernel=True)
        kern_coma = get_psf_kernel(pol7, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True, show_kernel=True)
        kern_spher = get_psf_kernel(pol8, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True, show_kernel=True)
        kern_4foil = get_psf_kernel(pol9, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True, show_kernel=True)

    if check_warnings:
        kern_def = get_psf_kernel(pol3, len2pixels=1.0, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True)

    if check_io:
        amplitude = 0.5  # in micrometers
        kern_spher = get_psf_kernel(pol8, len2pixels=pixel_size, alpha=amplitude, wavelength=wavelength, NA=NA, normalize_values=True)
        used_path = save_psf("test", psf_kernel=kern_spher, NA=NA, wavelength=wavelength,
                             calibration_coefficient=pixel_size, amplitude=amplitude, polynomial_orders=pol8)
        kern_spher_read = read_psf(used_path)['PSF Kernel']  # read the saved values
        diff = kern_spher - kern_spher_read  # for checking that saved / loaded kernels are consistent
        plt.figure("Loaded Kernel"); plt.imshow(kern_spher_read, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()

    if show_convolution_results:
        pass
