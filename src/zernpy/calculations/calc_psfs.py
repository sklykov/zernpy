# -*- coding: utf-8 -*-
"""
Calculate and plot point spread functions associated with Zernike wavefronts.

The diffraction integral is evaluated as U(r, theta) = integral of exp(1j*(alpha*Z(p, phi) - r*p*cos(phi - theta)))*p dp dphi
over the unit pupil, with the intensity |U|**2/pi**2 (equal to 1.0 at the center of an unaberrated PSF).
Z is the OSA-normalized Zernike polynomial (unit RMS over the pupil), so the wavefront is a*Z with a being its RMS value.
All functions in this module expect the phase coefficient alpha = 2*pi*a/wavelength in radians, where a and wavelength
share the same length unit (e.g., micrometres). The conversion from a to alpha is performed by the caller.
The image-plane radius is used in the dimensionless form r = k*NA*pixel_size*pixel_distance, where k = 2*pi/wavelength.
The model is scalar: polarization (vectorial) effects, significant for high NA objectives (approximately NA > 0.7), are ignored.

@author: Sergei Klykov

@licence: MIT, @year: 2024

"""
# %% Global imports
import json
import time
import warnings
from math import e, pi, sqrt
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from scipy.special import jv

# %% Local (package-scoped) imports
from .calc_zernike_pol import define_orders

# %% Module parameters
__docformat__ = "numpydoc"
um_char = "\u00B5"  # Unicode char code for micrometers
lambda_char = "\u03BB"  # Unicode char code for lambda (wavelength)
pi_char = "\u03C0"  # Unicode char code for pi


# %% Sampling (Nyquist) check
def nyquist_pixel_size(wavelength: float, NA: float) -> float:
    """
    Return the largest pixel size satisfying Nyquist sampling: half of the Abbe resolution limit, 0.25*wavelength/NA.

    Parameters
    ----------
    wavelength : float
        Wavelength in physical units (e.g., micrometres).
    NA : float
        Numerical aperture of the objective.

    Returns
    -------
    float
        Pixel size in the same length unit as wavelength.

    """
    return 0.25*wavelength/NA


def nyquist_warning_message(pixel_size: float, wavelength: float, NA: float) -> str:
    """
    Compose the warning message if the pixel size is too large for Nyquist sampling of the PSF.

    Parameters
    ----------
    pixel_size : float
        Pixel size in the same length unit as wavelength.
    wavelength : float
        Wavelength in physical units (e.g., micrometres).
    NA : float
        Numerical aperture of the objective.

    Returns
    -------
    str
        Warning message, or empty string if the sampling is sufficient.

    """
    max_pixel_size = nyquist_pixel_size(wavelength, NA)
    if pixel_size > max_pixel_size:
        return (f"\nPixel size {pixel_size} is larger than the Nyquist sampling limit {max_pixel_size:.3e} (the same length unit as "
                + f"the wavelength), which is half of the Abbe resolution limit 0.5*{lambda_char}/NA. The PSF will be undersampled")
    return ""


# %% Reference - Airy profile for Z(0, 0)
def airy_ref_pattern(r: float) -> float:
    """
    Return the normalized Airy intensity, 4*(J1(r)/r)**2.

    Parameters
    ----------
    r : float
        Dimensionless radial argument, k*NA times the physical radial distance, where k = 2*pi/wavelength.

    Returns
    -------
    float
        Airy intensity with a limiting value of 1 at r = 0.

    """
    r = round(r, 12)
    ratio = jv(1, 1E-11)/1E-11 if r == 0.0 else jv(1, r)/r
    return 4.0*pow(ratio, 2)


# %% PSF pixel value calc.
def diffraction_integral_r(zernike_pol, alpha: float, phi: float, p: Union[float, np.ndarray], theta: float,
                           r: float) -> Union[complex, np.ndarray]:
    """
    Diffraction integral function for the formed image point (see the references as the sources of the equation).

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    alpha : float
        Converted phase coefficient 2*pi*a/wavelength, where a is the supplied physical wavefront coefficient.
    phi : float
        Angle on the pupil (entrance pupil of micro-objective) coordinates (for integration).
    p : numpy.array or float
        Integration interval on the pupil (entrance pupil of micro-objective) radius or radius as float number.
    theta : float
        Angle on the image coordinates.
    r : float
        Scaled image-plane radius, k*NA*pixel_size*pixel_distance, where k = 2*pi/wavelength.

    References
    ----------
    [1] Principles of Optics, by M. Born and E. Wolf, 4 ed., 1968
    [2] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf

    Returns
    -------
    complex or numpy.ndarray
        Complex integrand values, including the pupil-radius factor p.

    """
    phase_arg = (alpha*zernike_pol.polynomial_value(p, phi) - r*p*np.cos(phi - theta))*1j
    return np.exp(phase_arg)*p


def radial_integral(zernike_pol, r: float, theta: float, phi: float, alpha: float, n_int_r_points: int) -> complex:
    """
    Make integration of the diffraction integral on the radius of the entrance pupil.

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    r : float
        Scaled image-plane radius, k*NA*pixel_size*pixel_distance, where k = 2*pi/wavelength.
    theta : float
        Angle on the image coordinates.
    phi : float
        Angle on the pupil coordinates.
    alpha : float
        Converted phase coefficient 2*pi*a/wavelength, where a is the supplied physical wavefront coefficient.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).

    Returns
    -------
    complex
        Complex amplitude of the field as result of integration on the pupil radius coordinate.

    """
    # Integrate over the pupil radius using the vectorized trapezoidal rule
    h_p = 1.0/n_int_r_points; p = np.arange(h_p, 1.0, h_p)
    fa = diffraction_integral_r(zernike_pol, alpha, phi, 0.0, theta, r)
    fb = diffraction_integral_r(zernike_pol, alpha, phi, 1.0, theta, r)
    ang_int = np.sum(diffraction_integral_r(zernike_pol, alpha, phi, p, theta, r)) + 0.5*(fa + fb)
    return h_p*ang_int


def get_psf_point_r(zernike_pol, r: float, theta: float, alpha: float, n_int_r_points: int, n_int_phi_points: int) -> float:
    """
    Get the point for calculation of PSF depending on the image polar coordinates.

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    r : float
        Scaled image-plane radius, k*NA*pixel_size*pixel_distance, where k = 2*pi/wavelength.
    theta : float
        Angle on the image coordinates.
    alpha : float
        Converted phase coefficient 2*pi*a/wavelength, where a is the supplied physical wavefront coefficient.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).
    n_int_phi_points : int
        Even number of Simpson subintervals on the polar angle of the entrance pupil, range [0, 2*pi].
        The rule evaluates n_int_phi_points + 1 nodes.

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    # Composite Simpson's rule on n_int_phi_points (even) subintervals of [0, 2*pi]:
    # interior even nodes 2, 4, ..., n-2 get weight 2, odd nodes 1, 3, ..., n-1 get weight 4
    h_phi = 2.0*pi/n_int_phi_points; even_sum = 0.0j; odd_sum = 0.0j
    for i in range(2, n_int_phi_points-1, 2):
        phi = i*h_phi; even_sum += radial_integral(zernike_pol, r, theta, phi, alpha, n_int_r_points)
    for i in range(1, n_int_phi_points, 2):
        phi = i*h_phi; odd_sum += radial_integral(zernike_pol, r, theta, phi, alpha, n_int_r_points)
    yA = radial_integral(zernike_pol, r, theta, 0.0, alpha, n_int_r_points)
    yB = radial_integral(zernike_pol, r, theta, 2.0*pi, alpha, n_int_r_points)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


# %% Vectorized PSF point calculation
def get_psf_point_r_vectorized(zernike_pol, r: float, theta: float, alpha: float, n_int_r_points: int, n_int_phi_points: int) -> float:
    """
    Calculate PSF intensity using array-based summation of radial integrals.

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    r : float
        Scaled image-plane radius, k*NA*pixel_size*pixel_distance, where k = 2*pi/wavelength.
    theta : float
        Angle on the image coordinates.
    alpha : float
        Converted phase coefficient 2*pi*a/wavelength, where a is the supplied physical wavefront coefficient.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).
    n_int_phi_points : int
        Even number of Simpson subintervals on the polar angle of the entrance pupil, range [0, 2*pi].
        The rule evaluates n_int_phi_points + 1 nodes.

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    # Composite Simpson's rule on n_int_phi_points (even) subintervals of [0, 2*pi]:
    # interior even nodes 2, 4, ..., n-2 get weight 2, odd nodes 1, 3, ..., n-1 get weight 4
    h_phi = 2.0*pi/n_int_phi_points; even_sum = 0.0j; odd_sum = 0.0j
    # Vectorized form of for loop for even and odd phi-s
    even_sums = np.asarray([radial_integral(zernike_pol, r, theta, i*h_phi, alpha, n_int_r_points)
                            for i in range(2, n_int_phi_points-1, 2)])
    odd_sums = np.asarray([radial_integral(zernike_pol, r, theta, i*h_phi, alpha, n_int_r_points)
                           for i in range(1, n_int_phi_points, 2)])
    even_sum = np.sum(even_sums); odd_sum = np.sum(odd_sums)
    # Simpson integration rule implementation
    yA = radial_integral(zernike_pol, r, theta, 0.0, alpha, n_int_r_points)
    yB = radial_integral(zernike_pol, r, theta, 2.0*pi, alpha, n_int_r_points)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


# %% PSF kernel calc.
def get_kernel_size(zernike_pol, len2pixels: float, alpha: float, wavelength: float, NA: float) -> int:
    """
    Estimate empirically the kernel size.

    Parameters
    ----------
    zernike_pol : ZernPol
        The instance of ZernPol() class required for calculation of Zernike polynomial values.
    len2pixels : float
        Pixel size in the same length unit as wavelength.
    alpha : float
        Converted phase coefficient 2*pi*a/wavelength, where a and wavelength share the same length unit.
    wavelength : float
        Monochromatic wavelength.
    NA : float
        Numerical Aperture of the objective.

    Returns
    -------
    int
        Estimated kernel size.

    """
    m, n = define_orders(zernike_pol)  # get polynomial orders
    size_ext = 0   # additional size depending on some parameters below
    abs_alpha = abs(alpha)
    if m == 0 and n == 0:  # Airy profile
        size = 11 + round(16.0*NA)
    else:
        if n <= 3:
            multiplier = n + 2
        else:
            multiplier = n
        if n - abs(m) <= (n + 1) // 2:  # Enlarge kernel size for the not symmetrical orders
            size_ext += int(round(sqrt(n+abs(m)))) + 1
        if abs_alpha >= 0.5:
            size_ext += 13  # enlarge kernel size additionally for high amplitude
        elif abs_alpha >= 0.25:
            size_ext += 11  # add one more line for kernel (prevent automatic warnings)
        elif abs_alpha >= 0.1:
            size_ext += 7
        elif abs_alpha >= 0.05:
            size_ext += 5
        # Tuning size for autoestimation of a kernel size
        if 1.0 <= abs_alpha  <= 1.5:
            size = int(round((multiplier*abs(1.25*alpha)*wavelength)/len2pixels, 0)) + 1 + size_ext
        elif abs_alpha > 1.5:
            size = int(round((multiplier*abs(alpha)*wavelength)/len2pixels, 0)) + 1 + size_ext
        elif abs_alpha >= 0.001:
            size = int(round(((multiplier+1)*wavelength)/len2pixels, 0)) + 1 + size_ext
        else:
            size = int(round(((multiplier-1)*wavelength)/len2pixels, 0)) + 1 + size_ext
    # Correct the size of a kernel to the odd integer below
    if size % 2 == 0:
        size += 1
    return size


def get_psf_kernel(zernike_pol, len2pixels: float, alpha: float, wavelength: float, NA: float, n_int_r_points: int = 320,
                   n_int_phi_points: int = 300, show_kernel: bool = False, fig_title: Optional[str] = None,
                   normalize_values: bool = False, airy_pattern: bool = False, kernel_size: int = 0, fig_id: str = "",
                   use_vectorized: bool = False, suppress_warns: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Calculate centralized matrix (kernel) with the PSF mask values.

    Parameters
    ----------
    zernike_pol : ZernPol
        The instance of ZernPol() class required for calculation of Zernike polynomial values.
    len2pixels : float
        Pixel size in the same length unit as wavelength.
    alpha : float
        Phase coefficient in radians, 2*pi*a/wavelength, where a is the physical wavefront coefficient (RMS)
        in the same length unit as wavelength. The conversion is performed by the caller.
    wavelength : float
        Wavelength (λ) in physical units (e.g., µm) of the light used for calculations (in imaging).
    NA : float
        Objective property.
    n_int_r_points : int, optional
        Number of points used for integration on the unit pupil radius from the range [0.0, 1.0]. The default is 320.
    n_int_phi_points : int, optional
        Number of Simpson subintervals on the pupil angle, range [0, 2*pi]. Odd values are increased by one,
        because Simpson's rule requires an even number of subintervals. The default is 300.
    show_kernel : bool, optional
        Plot the calculated kernel interactively. The default is False.
    fig_title : str, optional
        Custom figure title. The default is None.
    normalize_values : bool, optional
        Normalize all values in the sense that the max kernel value = 1.0. The default is False.
    airy_pattern : bool, optional
        Calculate the analytic Airy intensity instead of the aberrated diffraction integral. The default is False.
    kernel_size : int, optional
        Custom kernel size, if not provided, then the size will be estimated based on the parameters. The default is 0.
    fig_id : str, optional
        Some string id for the figure title. The default is "".
    use_vectorized : bool, optional
        Sum angular integration terms using NumPy arrays instead of accumulating them in loops. The default is False.
    suppress_warns : bool, optional
        Flag for suppressing any thrown warnings. The default is False.
    verbose : bool, optional
        Flag for printing explicitly # of points calculated on each run and measure how long it takes to calculate it. The default is False.

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.

    """
    m, n = define_orders(zernike_pol)  # get polynomial orders
    k = 2.0*pi/wavelength  # wave number in radians per unit length
    # Empirical estimation of the sufficient size for the kernel
    if kernel_size < 3:
        size = get_kernel_size(zernike_pol, len2pixels, alpha, wavelength, NA)
    else:
        size = kernel_size
    # Auto definition of the required PSF size is complicated for the different PSFs forms (e.g., vertical coma with different amplitudes)
    # Make kernel with odd sizes for precisely centering the kernel (in the center of an image)
    if size % 2 == 0:
        size += 1
    if n_int_phi_points % 2 == 1:
        n_int_phi_points += 1  # Simpson's rule on the pupil angle requires an even number of subintervals
    kernel = np.zeros(shape=(size, size)); i_center = size//2; j_center = size//2
    # Print note about calculation duration
    if size > 25 and not suppress_warns and not airy_pattern:
        if abs(n_int_phi_points - 300) < 40 and abs(n_int_r_points - 320) < 50:
            print(f"Note that the estimated kernel size: {size}x{size} for {(m, n)}. "
                  + f"Estimated calc. time: {int(round(size*size*38.5/1000, 0))} sec.")  # estimated based on tests only
        else:
            print(f"Note that the estimated kernel size: {size}x{size} for {(m, n)}. "
                  + "Calculation may take from several dozens of seconds to minutes")
    # Check that the calibration coefficient is sufficient for calculation
    __warn_message = nyquist_warning_message(len2pixels, wavelength, NA)
    if len(__warn_message) > 0 and not suppress_warns:
        warnings.warn(__warn_message, stacklevel=2)
    # Calculate the PSF kernel for using in the convolution
    if verbose:
        calculated_points = 0  # for explicit showing of performance
        show_each_tenth_point = False; checking_point = 1  # flag and value for shortening print output
        if 100 < size*size < 301:
            show_each_tenth_point = True; checking_point = 10
    for i in range(size):
        for j in range(size):
            if verbose:
                t1 = time.perf_counter()  # for explicit showing of performance
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))  # in pixels
            # Convert pixel distance in the required k*NA*pixel_dist*calibration coefficient
            distance = k*NA*len2pixels*pixel_dist  # conversion from pixel distance into phase multiplier in the diffraction integral
            # The PSF also has the angular dependency, not only the radial one
            theta = np.arctan2((i - i_center), (j - j_center))
            theta += np.pi  # shift angles to the range [0, 2pi]
            # The scaling below is not needed because the Zernike polynomial is scaled as the RMS values
            if not airy_pattern:
                if not use_vectorized:
                    kernel[i, j] = get_psf_point_r(zernike_pol, distance, theta, alpha, n_int_r_points, n_int_phi_points)
                else:
                    kernel[i, j] = get_psf_point_r_vectorized(zernike_pol, distance, theta, alpha, n_int_r_points, n_int_phi_points)
            else:
                kernel[i, j] = airy_ref_pattern(distance)
            # Report progress for both integration variants and the Airy pattern
            if verbose:
                calculated_points += 1; passed_time_ms = int(round(1000*(time.perf_counter() - t1), 0))
                if show_each_tenth_point and (calculated_points == 1 or calculated_points == checking_point):
                    print(f"Calculated point #{calculated_points} from {size*size}, took: {passed_time_ms} ms")
                    if calculated_points == checking_point:
                        checking_point += 10
                elif (not show_each_tenth_point and not size*size >= 301):
                    print(f"Calculated point #{calculated_points} from {size*size}, took: {passed_time_ms} ms")
    # Normalize all values in kernel to bring the max value to 1.0
    if normalize_values:
        kernel /= np.max(kernel)
    # Provide warning in the case if kernel size not sufficient for representation of the calculated kernel
    k_size, k_size = kernel.shape
    kernel_max_zero_col = np.max(kernel[:, 0]); kernel_max_zero_row = np.max(kernel[0, :])
    kernel_max_end_col = np.max(kernel[:, k_size-1]); kernel_max_end_row = np.max(kernel[k_size-1, :])
    kernel_border_max = np.max([kernel_max_zero_col, kernel_max_zero_row, kernel_max_end_col, kernel_max_end_row])
    if kernel_border_max > np.max(kernel)/20.0 and not suppress_warns:
        __warn_message = (f"\nThe calculated size for PSF ({size}) isn't sufficient for its proper representation, "
                          + "because the maximum value on the kernel border is bigger than 5% of maximum overall kernel")
        warnings.warn(__warn_message, stacklevel=2)
    # Plotting the calculated kernel
    if show_kernel:
        if airy_pattern:
            fig_title = "Airy pattern"
        if fig_title is not None and len(fig_title) > 0:
            plt.figure(fig_title, figsize=(6, 6))
        else:
            plt.figure(f"{(m, n)} {zernike_pol.get_polynomial_name(True)}: {round(alpha/(2.0*pi), 2)}*wavelength {fig_id}", figsize=(6, 6))
        plt.imshow(kernel, cmap=plt.colormaps["viridis"], origin='upper'); plt.tight_layout()
    return kernel


# %% PSF calculation for several polynomials
def diffraction_integral_r_pols(polynomials, amplitudes: np.ndarray, phi: float, p: Union[float, np.ndarray],
                                theta: float, r: float) -> np.ndarray:
    """
    Diffraction integral function for the formed image point (see the references as the sources of the equation).

    Parameters
    ----------
    polynomials : Sequence[ZernPol]
        Instances of ZernPol() class required for calculation of Zernike polynomials values.
    amplitudes : np.ndarray
        Phase coefficients in radians, 2*pi*a/wavelength, where a are the physical wavefront coefficients
        in the same length unit as wavelength. The conversion is performed by the caller.
    phi : float
        Angle on the pupil (entrance pupil of micro-objective) coordinates (for integration).
    p : float or np.ndarray
        Integration interval on the pupil (entrance pupil of micro-objective) radius or radius as float number.
    theta : float
        Angle on the image coordinates.
    r : float
        Scaled image-plane radius, k*NA*pixel_size*pixel_distance, where k = 2*pi/wavelength.

    References
    ----------
    [1] Principles of Optics, by M. Born and E. Wolf, 4 ed., 1968
    [2] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf

    Returns
    -------
    complex or numpy.ndarray
        Complex integrand values, including the pupil-radius factor p.

    """
    # NOTE: the manual for loop should be implemented, because sum can be calculated for float 'p' and for np.ndarray 'p'
    for i, zernike_pol in enumerate(polynomials):
        if i == 0:
            sum_polynomial_values = amplitudes[i]*zernike_pol.polynomial_value(p, phi)
        else:
            sum_polynomial_values += amplitudes[i]*zernike_pol.polynomial_value(p, phi)
    phase_arg = (sum_polynomial_values - r*p*np.cos(phi - theta))*1j
    return np.exp(phase_arg)*p


def radial_integral_pols(polynomials, amplitudes: np.ndarray, r: float, theta: float, phi: float, n_int_r_points: int) -> complex:
    """
    Make integration of the diffraction integral on the radius of the entrance pupil.

    Parameters
    ----------
    polynomials : Sequence[ZernPol]
        Instances of ZernPol() class required for calculation of Zernike polynomials values.
    amplitudes : np.ndarray
        Phase coefficients in radians, 2*pi*a/wavelength, where a are the physical wavefront coefficients
        in the same length unit as wavelength. The conversion is performed by the caller.
    r : float
        Scaled image-plane radius, k*NA*pixel_size*pixel_distance, where k = 2*pi/wavelength.
    theta : float
        Angle on the image coordinates.
    phi : float
        Angle on the pupil coordinates.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).

    Returns
    -------
    complex
        Complex amplitude of the field as result of integration on the pupil radius coordinate.

    """
    # Integrate over the pupil radius using the vectorized trapezoidal rule
    h_p = 1.0/n_int_r_points; p = np.arange(h_p, 1.0, h_p)
    fa = diffraction_integral_r_pols(polynomials, amplitudes, phi, 0.0, theta, r)
    fb = diffraction_integral_r_pols(polynomials, amplitudes, phi, 1.0, theta, r)
    ang_int = np.sum(diffraction_integral_r_pols(polynomials, amplitudes, phi, p, theta, r)) + 0.5*(fa + fb)
    return h_p*ang_int


def get_psf_point_r_pols(polynomials, amplitudes: np.ndarray, r: float, theta: float, n_int_r_points: int, n_int_phi_points: int) -> float:
    """
    Calculate PSF point for provided polynomials.

    Parameters
    ----------
    polynomials : Sequence[ZernPol]
        Instances of ZernPol() class required for calculation of Zernike polynomials values.
    amplitudes : np.ndarray
        Phase coefficients in radians, 2*pi*a/wavelength, where a are the physical wavefront coefficients
        in the same length unit as wavelength. The conversion is performed by the caller.
    r : float
        Scaled image-plane radius, k*NA*pixel_size*pixel_distance, where k = 2*pi/wavelength.
    theta : float
        Angle on the image coordinates.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).
    n_int_phi_points : int
        Even number of Simpson subintervals on the polar angle of the entrance pupil, range [0, 2*pi].
        The rule evaluates n_int_phi_points + 1 nodes.

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    # Composite Simpson's rule on n_int_phi_points (even) subintervals of [0, 2*pi]:
    # interior even nodes 2, 4, ..., n-2 get weight 2, odd nodes 1, 3, ..., n-1 get weight 4
    h_phi = 2.0*pi/n_int_phi_points; even_sum = 0.0j; odd_sum = 0.0j
    # Vectorized or parallelized form of for loop for even and odd phi-s
    even_sums = np.asarray([radial_integral_pols(polynomials, amplitudes, r, theta, i*h_phi, n_int_r_points)
                            for i in range(2, n_int_phi_points-1, 2)])
    odd_sums = np.asarray([radial_integral_pols(polynomials, amplitudes, r, theta, i*h_phi, n_int_r_points)
                           for i in range(1, n_int_phi_points, 2)])
    even_sum = np.sum(even_sums); odd_sum = np.sum(odd_sums)
    # Simpson integration rule implementation
    yA = radial_integral_pols(polynomials, amplitudes, r, theta, 0.0, n_int_r_points)
    yB = radial_integral_pols(polynomials, amplitudes, r, theta, 2.0*pi, n_int_r_points)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


def get_psf_kernel_zerns(polynomials, amplitudes: np.ndarray, len2pixels: float, wavelength: float, NA: float, kernel_size: int,
                         n_int_r_points: int = 320, n_int_phi_points: int = 300, normalize_values: bool = False,
                         suppress_warns: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Calculate centralized matrix with the PSF mask values.

    Parameters
    ----------
    polynomials : Sequence[ZernPol]
        Instances of ZernPol() class required for calculation of Zernike polynomials values.
    amplitudes : np.ndarray
        Phase coefficients in radians, 2*pi*a/wavelength, where a are the physical wavefront coefficients
        in the same length unit as wavelength. The conversion is performed by the caller.
    len2pixels : float
        Pixel size in the same length unit as wavelength.
    wavelength : float
        Wavelength used for the PSF calculation (in physical units).
    NA : float
        NA used for the PSF calculation.
    kernel_size : int
        Size of a kernel - square matrix.
    n_int_r_points : int, optional
        Number of points used for integration on the unit pupil radius from the range [0.0, 1.0]. The default is 320.
    n_int_phi_points : int, optional
        Number of Simpson subintervals on the pupil angle, range [0, 2*pi]. Odd values are increased by one,
        because Simpson's rule requires an even number of subintervals. The default is 300.
    normalize_values : bool, optional
        Normalize all values in the sense that the max kernel value = 1.0. The default is False.
    suppress_warns : bool, optional
        Flag for suppressing any UserWarnings. The default is False.
    verbose : bool, optional
        Print calculation progress and elapsed time. The default is False.

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.

    """
    k = 2.0*pi/wavelength; size = kernel_size  # calculate the wave number and copy the kernel size
    # Make kernel with odd sizes for precisely centering the kernel (in the center of an image)
    if size % 2 == 0:
        size += 1
    if n_int_phi_points % 2 == 1:
        n_int_phi_points += 1  # Simpson's rule on the pupil angle requires an even number of subintervals
    kernel = np.zeros(shape=(size, size)); i_center = size//2; j_center = size//2
    # Print note about calculation duration
    if size > 25 and not suppress_warns:
        if abs(n_int_phi_points - 300) < 40 and abs(n_int_r_points - 320) < 50:
            print(f"Estimated calc. time: {int(round(size*size*38.5/1000, 0))} sec.")  # estimated based on tests only
        else:
            print("Calculation may take from several dozens of seconds to minutes")
    # Check that the calibration coefficient is sufficient for calculation
    __warn_message = nyquist_warning_message(len2pixels, wavelength, NA)
    if len(__warn_message) > 0 and not suppress_warns:
        warnings.warn(__warn_message, stacklevel=2)
    # Verbose info preparation
    if verbose:
        calculated_points = 0  # for explicit showing of performance
        show_each_tenth_point = False; checking_point = 1  # flag and value for shortening print output
        if 100 < size*size < 301:
            show_each_tenth_point = True; checking_point = 10
    # Do not perform check for provided Airy pattern case for making an agreement between accelerated and not accelerated cases
    # Calculate the PSF kernel for usage in convolution operation
    for i in range(size):
        for j in range(size):
            if verbose:
                t1 = time.perf_counter()  # for explicit showing of performance
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))  # in pixels
            # Convert pixel distance in the required k*NA*pixel_dist*calibration coefficient
            distance = k*NA*len2pixels*pixel_dist  # conversion from pixel distance into phase multiplier in the diffraction integral
            # The PSF also has the angular dependency, not only the radial one
            theta = np.arctan2((i - i_center), (j - j_center))
            theta += np.pi  # shift angles to the range [0, 2pi]
            # The scaling below is not needed because the Zernike polynomial is scaled as the RMS values
            kernel[i, j] = get_psf_point_r_pols(polynomials, amplitudes, distance, theta, n_int_r_points, n_int_phi_points)
            if verbose:
                calculated_points += 1; passed_time_ms = int(round(1000.0*(time.perf_counter() - t1), 0))
                if show_each_tenth_point and (calculated_points == 1 or calculated_points == checking_point):
                    print(f"Calculated point #{calculated_points} from {size*size}, took: {passed_time_ms} ms")
                    if calculated_points == checking_point:
                        checking_point += 10
                elif (not show_each_tenth_point and not size*size >= 301):
                    print(f"Calculated point #{calculated_points} from {size*size}, took: {passed_time_ms} ms")
    # Normalize all values in kernel to bring the max value to 1.0
    if normalize_values:
        kernel /= np.max(kernel)
    # Provide warning in the case if kernel size not sufficient for representation of the calculated kernel
    k_size, k_size = kernel.shape
    kernel_max_zero_col = np.max(kernel[:, 0]); kernel_max_zero_row = np.max(kernel[0, :])
    kernel_max_end_col = np.max(kernel[:, k_size-1]); kernel_max_end_row = np.max(kernel[k_size-1, :])
    kernel_border_max = np.max([kernel_max_zero_col, kernel_max_zero_row, kernel_max_end_col, kernel_max_end_row])
    if kernel_border_max > np.max(kernel)/20.0 and not suppress_warns:
        __warn_message = (f"\nThe calculated size for PSF ({size}) isn't sufficient for its proper representation, "
                          + "because the maximum value on the kernel border is bigger than 5% of maximum overall kernel")
        warnings.warn(__warn_message, stacklevel=2)
    return kernel


# %% Convolute a PSF kernel with an image
def convolute_img_psf(img: np.ndarray, psf_kernel: np.ndarray, scale2original: bool = False) -> np.ndarray:
    """
    Convolute the provided image with PSF kernel as 2D arrays and return the convolved image with the same type as the original one.

    Parameters
    ----------
    img : numpy.ndarray
        Sample image, not colour.
    psf_kernel : numpy.ndarray
        Calculated PSF kernel.
    scale2original : bool, optional
        Rescale convolved image to the maximum level of the input image. The default is False.

    Returns
    -------
    convolved_img : numpy.ndarray
        Convolution with reflected boundaries, divided by the kernel sum when positive, and cast back to the input dtype.

    """
    img_type = img.dtype; img = np.copy(img).astype(np.float64)  # get the image type and copy its content to a new container
    convolved_img: np.ndarray
    convolved_img = convolve(img, psf_kernel, mode='reflect'); conv_coeff = np.sum(psf_kernel)  # convolution using scipy
    if conv_coeff > 0.0:
        convolved_img /= conv_coeff  # correct the convolution result by dividing to the kernel sum
    if scale2original:
        max_original_intensity = np.max(img); max_conv_pixel = np.max(convolved_img)
        scaling_factor = max_original_intensity / max_conv_pixel; convolved_img *= scaling_factor
    convolved_img = convolved_img.astype(dtype=img_type)  # converting convolved image to the initial image
    return convolved_img


# %% Save and read the calculated PSF matrices
def save_psf(psf_kernel: np.ndarray, NA: float, wavelength: float, pixel_size: float, expansion_coefficient: Union[float, np.ndarray],
             kernel_size: int, n_int_points_r: int, n_int_points_phi: int, zernike_pol, folder_path: str,
             overwrite: bool = True, additional_file_name: Optional[str] = None) -> str:
    """
    Save the calculated PSF kernel along with the used for the calculation parameters.

    Parameters
    ----------
    psf_kernel : np.ndarray
        Calculated by using get_psf_kernel.
    NA : float
        NA used for the PSF calculation.
    wavelength : float
        Wavelength used for the PSF calculation (in physical units).
    pixel_size : float
        Pixel size (in physical units same to wavelength) used for the PSF calculation.
    expansion_coefficient : float or np.ndarray
        Polynomial coefficients in the same physical length unit as wavelength, before conversion to radians.
    kernel_size : int
        Size of the PSF kernel.
    n_int_points_r : int
        Number of the used integration points on r.
    n_int_points_phi : int
        Number of the used integration points on phi.
    zernike_pol : ZernPol instance or Sequence[ZernPol]
        Zernike polynomial(-s).
    folder_path : str
        Path to an existing destination folder, prepared by the caller.
    overwrite : bool, optional
        Flag for letting overwriting of the existing file. The default is True.
    additional_file_name : str, optional
        Additional to the composed file name string, e.g. unique addition. The default is None.

    Returns
    -------
    str
        Absolute path to the file.

    """
    single_pol_used = False; osa_indices = []  # flag for saving different parameters
    if isinstance(expansion_coefficient, np.ndarray):
        ampls = expansion_coefficient.copy().tolist()
    if not hasattr(zernike_pol, '__len__'):
        m, n = define_orders(zernike_pol); single_pol_used = True  # get polynomial orders
    else:
        for zernpol in zernike_pol:
            osa_indices.append(zernpol.get_indices()[1])
    saved_psfs_folder = Path(folder_path)  # folder selection and creation belong to the caller
    # Save provided PSF kernel with the provided parameters
    if additional_file_name is not None and len(additional_file_name) > 0:
        if single_pol_used:
            json_file_path = saved_psfs_folder.joinpath(f"psf_{(m, n)}_{additional_file_name}_{expansion_coefficient}.json")
        else:
            json_file_path = saved_psfs_folder.joinpath(f"psf_{osa_indices}_{additional_file_name}_{ampls}.json")
    else:
        if single_pol_used:
            json_file_path = saved_psfs_folder.joinpath(f"psf_{(m, n)}_{expansion_coefficient}.json")
        else:
            json_file_path = saved_psfs_folder.joinpath(f"psf_{osa_indices}_{ampls}.json")
    # Data composing for recording
    data4serialization = {}   # python dictionary is similar to the JSON file structure and can be dumped directly there
    data4serialization['PSF Kernel'] = psf_kernel.tolist(); data4serialization['NA'] = NA; data4serialization['Wavelength'] = wavelength
    data4serialization["Pixel Size"] = pixel_size; data4serialization["Kernel Size"] = kernel_size
    data4serialization["# of integration points R"] = n_int_points_r; data4serialization["# of integration points angle"] = n_int_points_phi
    if single_pol_used:
        data4serialization["Expansion Coefficient"] = expansion_coefficient; data4serialization["Polynomial"] = zernike_pol.get_indices()[1]
    else:
        data4serialization["Amplitudes"] = ampls; data4serialization["Polynomials"] = osa_indices
    # File presence check and recording
    if json_file_path.exists() and not overwrite:
        _warn_message = "\nThe file already exists, the content won't be overwritten."
        warnings.warn(_warn_message, stacklevel=2)
    if not json_file_path.exists() or (json_file_path.exists() and overwrite):
        with open(json_file_path, 'w') as json_write_file:
            json.dump(data4serialization, json_write_file)
    return str(json_file_path.absolute())


def read_psf(file_path: str) -> Optional[dict]:
    """
    Read the saved PSF data from the *json file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the JSON file.

    Returns
    -------
    dict or None
        Saved PSF data, or None if the path does not identify an existing file. Keys written by save_psf()
        include 'PSF Kernel', 'NA', 'Wavelength', 'Pixel Size', 'Kernel Size', integration settings,
        and either 'Polynomial' with 'Expansion Coefficient' or 'Polynomials' with 'Amplitudes'.

    """
    psf_file_path = Path(file_path); psf_data = None
    if psf_file_path.exists() and psf_file_path.is_file():
        with open(psf_file_path, 'r') as json_read_file:
            psf_data = json.load(json_read_file)
    return psf_data


# %% Sample generation
def get_bumped_circle(radius: float, max_intensity: int = 255) -> np.ndarray:
    """
    Get the circle using the 'bump' function.

    Parameters
    ----------
    radius : float
        Nominal disk radius in pixels, increased to 1.0 if smaller.
    max_intensity : int, optional
        Within the circle. The default is 255.

    Returns
    -------
    img : numpy.ndarray
        2D array (image) with the circle.

    """
    # Sizes calculation
    if radius < 1.0:
        radius = 1.0
    max_size = 4*int(round(radius, 0)) + 1; i_img_center = max_size // 2; j_img_center = max_size // 2
    # Defining image type
    if max_intensity <= 255:
        img = np.zeros(dtype='uint8', shape=(max_size, max_size))
    else:
        img = np.zeros(dtype='uint16', shape=(max_size, max_size))
    # Calculation pixel values
    q_rad = round(0.25*radius, 6)
    for i in range(max_size):
        for j in range(max_size):
            distance = np.round(np.sqrt(np.power(i - i_img_center, 2) + np.power(j - j_img_center, 2)), 6)
            pixel_value = 0.0  # meaning the intensity in the pixel
            r_exceed = 0.25; power = 8
            if distance < q_rad:
                pixel_value = 1.0  # entire pixel lays inside the circle
            # Continuous bump function
            elif distance < radius*(1.0 + r_exceed):
                x = distance/(radius + r_exceed); x_pow = pow(x, power); b_pow = pow(1.0 + r_exceed, power)
                pixel_value = e*np.exp(b_pow/(x_pow - b_pow))
            # Pixel value scaling according the provided image type
            pixel_value *= float(max_intensity)
            # Pixel value conversion to the image type
            if pixel_value > 0.0:
                pixel_value = int(round(pixel_value, 0)); img[i, j] = pixel_value
    return img


# %% Define standard exports from this module
__all__ = ['get_psf_kernel', 'save_psf', 'read_psf', 'convolute_img_psf', 'get_kernel_size', 'radial_integral',
           'um_char', 'lambda_char', 'get_bumped_circle', 'get_psf_kernel_zerns', 'nyquist_pixel_size', 'nyquist_warning_message']
