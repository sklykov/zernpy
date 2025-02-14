# -*- coding: utf-8 -*-
"""
Calculation and plotting of associated with polynomials PSFs.

@author: Sergei Klykov  \n
@licence: MIT, @year: 2024

"""
# %% Global imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from pathlib import Path
import warnings
from math import sqrt, pi, e
import time
from scipy.ndimage import convolve
import json
from matplotlib.patches import Circle
from functools import partial
from typing import Union
# import time

# Testing the parallelization with joblib. Native Pool.map() tested and results transferred to the collection_numCalc repo
try:
    from joblib import Parallel, delayed
    joblib_installed = True
except ModuleNotFoundError:
    joblib_installed = False

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calc_zernike_pol import define_orders
else:
    from .calc_zernike_pol import define_orders

# %% Module parameters
__docformat__ = "numpydoc"
um_char = "\u00B5"  # Unicode char code for micrometers
lambda_char = "\u03BB"  # Unicode char code for lambda (wavelength)
pi_char = "\u03C0"  # Unicode char code for pi
# print(um_char, lambda_char, pi_char)  # check the codes for characters from above


# %% Reference - Airy profile for Z(0, 0)
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
    # NOTE that the values produced by exact equation below is off with the direct computation of diffraction integral
    # apporoximate coefficient is 0.986711 for a central point. Most likely, the difference because of numerical integration
    return 4.0*pow(ratio, 2)


# %% PSF pixel value calc.
def diffraction_integral_r(zernike_pol, alpha: float, phi: float, p: Union[float, np.ndarray], theta: float, r: float) -> np.array:
    """
    Diffraction integral function for the formed image point (see the references as the sources of the equation).

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    alpha : float
        Amplitude of the polynomial (RMS).
    phi : float
        Angle on the pupil (entrance pupil of micro-objective) coordinates (for integration).
    p : numpy.array or float
        Integration interval on the pupil (entrance pupil of micro-objective) radius or radius as float number.
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


def radial_integral(zernike_pol, r: float, theta: float, phi: float, alpha: float, n_int_r_points: int) -> complex:
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


def radial_integral_args(phi: float, zernike_pol, r: float, theta: float, alpha: float, n_int_r_points: int) -> complex:
    """
    Make integration of the diffraction integral on the radius of the entrance pupil.

    The order of parameters are rearranged for the wrapping this function by using partial from functools.

    Parameters
    ----------
    phi : float
        Angle on the pupil coordinates.
    zernike_pol : ZernPol
        Zernike polynomial definition as the ZernPol() class.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
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


def radial_integral_s(params: tuple) -> complex:
    """
    Wrap function to accept input variables as tuple for parallelizing the computation.

    Parameters
    ----------
    params : tuple
        phi, constants. zernike_pol, r, theta, alpha, n_int_r_points = constants.

    Returns
    -------
    complex
        Phase integral value.

    """
    phi, constants = params; zernike_pol, r, theta, alpha, n_int_r_points = constants
    return radial_integral_args(phi, zernike_pol, r, theta, alpha, n_int_r_points)


def get_psf_point_r(zernike_pol, r: float, theta: float, alpha: float, n_int_r_points: int, n_int_phi_points: int) -> float:
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


# %% Testing various speeding up calculation approaches
def get_psf_point_r_parallel(zernike_pol, r: float, theta: float, alpha: float, n_int_r_points: int, n_int_phi_points: int,
                             paralleljobs=None) -> float:
    """
    Calculate PSF point for the kernel using Parallel class from the joblib library.

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
    paralleljobs : Parallel (from joblib import Parallel), optional
        Parallel class for parallelizing the computation jobs using joblib backend. The default is None.

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    h_phi = 2.0*pi/n_int_phi_points; even_sum = 0.0j; odd_sum = 0.0j
    # below - wrapping the callable function with the fixed arguments for using in the paralleled framework call
    radial_integral_fixed_args = partial(radial_integral_args, zernike_pol=zernike_pol, r=r, theta=theta,
                                         alpha=alpha, n_int_r_points=n_int_r_points)

    # Vectorized or parallelized form of for loop for even and odd phi-s
    if not joblib_installed or paralleljobs is None:
        even_sums = [radial_integral(zernike_pol, r, theta, i*h_phi, alpha, n_int_r_points) for i in range(2, n_int_phi_points-2, 2)]
        even_sums = np.asarray(even_sums); even_sum = np.sum(even_sums)
        odd_sums = [radial_integral(zernike_pol, r, theta, i*h_phi, alpha, n_int_r_points) for i in range(1, n_int_phi_points-1, 2)]
        odd_sums = np.asarray(odd_sums); odd_sum = np.sum(odd_sums)
    else:
        if paralleljobs is not None and isinstance(paralleljobs, Parallel):
            even_sums = paralleljobs(delayed(radial_integral_fixed_args)(i*h_phi) for i in range(2, n_int_phi_points-2, 2))
            odd_sums = paralleljobs(delayed(radial_integral_fixed_args)(i*h_phi) for i in range(1, n_int_phi_points-1, 2))
            # even_sum = sum(even_sums, start=even_sum); odd_sum = sum(even_sums, start=odd_sum)
            even_sums = np.asarray(even_sums); even_sum = np.sum(even_sums); odd_sums = np.asarray(odd_sums); odd_sum = np.sum(odd_sums)

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
        Relation between length in physical units (the same as the provided wavelength) and pixels.
    alpha : float
        Zernike amplitude (the expansion coefficient) in physical units used for the wavelength specification (e.g., \u00B5m).
        Note that the normalized Zernike polynomials are used, so its coefficient is normalized to the specified wavelength.
    wavelength : float
        Monochromatic wavelength.
    NA : float
        Numerical Aperture of the objective.

    Returns
    -------
    int
        Estimated kernel size.

    """
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    size_ext = 0   # additional size depending on some parameters below
    if m == 0 and n == 0:  # Airy
        if 0.25 < NA < 1.0:
            multiplier = 5.0*(1.0 - NA) + 1.5
        else:
            multiplier = 4.5 + 2.5*sqrt(1.0 / NA)
    else:
        multiplier = 1.25*sqrt(n)  # Enlarge kernel size according to the provided radial order n
        if abs(m) > 0 and n % 2 != 0:  # Enlarge kernel size for the not symmetrical orders
            multiplier += 0.5*sqrt(n - abs(m))
        elif abs(m) > 0 and n % 2 == 0:
            multiplier = 1.45*sqrt(n)
        if n - abs(m) <= (n + 1) // 2:  # Enlarge kernel size for the not symmetrical orders
            size_ext += int(round(sqrt(n+abs(m)))) + 1
    if abs(alpha) >= 0.5:
        multiplier *= sqrt(2.5*abs(alpha))  # Enlarge kernel size according to the provided amplitude, scaling with the coefficient
        size_ext += 2  # enlarge kernel size additionally for high amplitude
    elif abs(alpha) >= 0.25:
        size_ext += 1  # add one more line for kernel (prevent automatic warnings)
        multiplier *= sqrt(4.25*abs(alpha))  # Enlarge kernel size according to the provided amplitude, scaling with the coefficient
    # Estimation below based on the provided physical properties
    size = int(round((multiplier*wavelength)/len2pixels, 0)) + 1 + size_ext
    # Correct the size of a kernel to the odd integer below
    if size % 2 == 0:
        size += 1
    return size


def get_psf_kernel(zernike_pol, len2pixels: float, alpha: float, wavelength: float, NA: float, n_int_r_points: int = 320,
                   n_int_phi_points: int = 300, show_kernel: bool = False, fig_title: str = None, normalize_values: bool = False,
                   airy_pattern: bool = False, kernel_size: int = 0, test_parallel: bool = False, fig_id: str = "",
                   test_vectorized: bool = False, suppress_warns: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Calculate centralized matrix (kernel) with the PSF mask values.

    Parameters
    ----------
    zernike_pol : ZernPol
        The instance of ZernPol() class required for calculation of Zernike polynomial values.
    len2pixels : float
        Relation between length in physical units (the same as the provided wavelength) and pixels.
    alpha : float
        Zernike amplitude (the expansion coefficient) in physical units used for the wavelength specification (e.g., \u00B5m).
        Note that the normalized Zernike polynomials are used, so its coefficient is normalized to the specified wavelength.
    wavelength : float
        Wavelength (\u03BB) in physical units (e.g., \u00B5m) of the light used for calculations (in imaging).
    NA : float
        Objective property.
    n_int_r_points : int, optional
        Number of points used for integration on the unit pupil radius from the range [0.0, 1.0]. The default is 320.
    n_int_phi_points : int, optional
        Number of points used for integration on the unit pupil angle from the range [0.0, 2\u03C0]. The default is 300.
    show_kernel : bool, optional
        Plot the calculated kernel interactively. The default is False.
    fig_title : str, optional
        Custom figure title. The default is None.
    normalize_values : bool, optional
        Normalize all values in the sense that the max kernel value = 1.0. The default is False.
    airy_pattern : bool, optional
        Plot the Airy pattern for the provided parameters. The default is False.
    kernel_size : int, optional
        Custom kernel size, if not provided, then the size will be estimated based on the parameters. The default is 0.
    test_parallel : bool, optional
        Testing joblib library for speeding up calculations. The default is False.
    fig_id : str, optional
        Some string id for the figure title. The default is "".
    test_vectorized : bool, optional
        For using vectorized calculations instead of simple for loops. The default is False.
    suppress_warns : bool, optional
        Flag for suppressing any thrown warnings. The default is False.
    verbose: bool, optional
        Flag for printing explicitly # of points calculated on each run and measure how long it takes to calculate it. The default is False.

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.

    """
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    # Convert provided absolute value of Zernike expansion coefficient (in um) into fraction of wavelength
    alpha /= wavelength; k = 2.0*pi/wavelength  # Calculate angular frequency (k)
    # Empirical estimation of the sufficient size for the kernel
    if kernel_size < 3:
        size = get_kernel_size(zernike_pol, len2pixels, alpha, wavelength, NA)
    else:
        size = kernel_size
    # Auto definition of the required PSF size is complicated for the different PSFs forms (e.g., vertical coma with different amplitudes)
    # Make kernel with odd sizes for precisely centering the kernel (in the center of an image)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros(shape=(size, size)); i_center = size//2; j_center = size//2
    # Print note about calculation duration
    if size > 25 and not suppress_warns and not airy_pattern:
        if abs(n_int_phi_points - 300) < 40 and abs(n_int_r_points - 320) < 50:
            print(f"Note that the estimated kernel size: {size}x{size} for {(m, n)}."
                  + f"Estimated calc. time: {int(round(size*size*38.5/1000, 0))} sec.")  # estimated based on tests only
        else:
            print(f"Note that the estimated kernel size: {size}x{size} for {(m, n)}."
                  + "Calculation may take from several dozens of seconds to minutes")
    # Check that the calibration coefficient is sufficient for calculation
    pixel_size_nyquist = 0.5*0.61*wavelength/NA
    if len2pixels > pixel_size_nyquist and not suppress_warns:
        __warn_message = f"Provided calibration coefficient {len2pixels} {um_char}/pixels isn't sufficient enough"
        __warn_message += f" (defined by the relation between Nyquist freq. and the optical resolution: 0.61{lambda_char}/NA)"
        warnings.warn(__warn_message)
    # Calculate the PSF kernel for usage in convolution operation
    # mean_time_integration = 0.0; n = 0
    if not joblib_installed or not test_parallel:
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
                    if not test_vectorized:
                        kernel[i, j] = get_psf_point_r(zernike_pol, distance, theta, alpha, n_int_r_points, n_int_phi_points)
                    else:
                        kernel[i, j] = get_psf_point_r_parallel(zernike_pol, distance, theta, alpha, n_int_r_points, n_int_phi_points)
                        if verbose:
                            calculated_points += 1; passed_time_ms = int(round(1000*(time.perf_counter() - t1), 0))
                            if show_each_tenth_point and (calculated_points == 1 or calculated_points == checking_point):
                                print(f"Calculated point #{calculated_points} from {size*size}, takes: {passed_time_ms} ms")
                                if calculated_points == checking_point:
                                    checking_point += 10
                            elif (not show_each_tenth_point and not size*size >= 301):
                                print(f"Calculated point #{calculated_points} from {size*size}, takes: {passed_time_ms} ms")
                else:
                    kernel[i, j] = airy_ref_pattern(distance)
    elif joblib_installed and test_parallel:
        # NOTE: after several tests, it is clear that the parallelization using joblib not optimizing performance
        with Parallel(n_jobs=4, pre_dispatch=size*size*n_int_phi_points*n_int_phi_points+2, backend='multiprocessing') as paralleljobs:
            for i in range(size):
                for j in range(size):
                    pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))  # in pixels
                    # Convert pixel distance in the required k*NA*pixel_dist*calibration coefficient
                    distance = k*NA*len2pixels*pixel_dist  # conversion from pixel distance into phase multiplier in the diffraction integral
                    # The PSF also has the angular dependency, not only the radial one
                    theta = np.arctan2((i - i_center), (j - j_center))
                    theta += np.pi  # shift angles to the range [0, 2pi]
                    # The scaling below is not needed because the Zernike polynomial is scaled as the RMS values
                    if not airy_pattern:
                        kernel[i, j] = get_psf_point_r_parallel(zernike_pol, distance, theta, alpha, n_int_r_points,
                                                                n_int_phi_points, paralleljobs)
                    else:
                        kernel[i, j] = airy_ref_pattern(distance)
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
        warnings.warn(__warn_message)
    # Plotting the calculated kernel
    if show_kernel:
        if airy_pattern:
            fig_title = "Airy pattern"
        if fig_title is not None and len(fig_title) > 0:
            plt.figure(fig_title, figsize=(6, 6))
        else:
            plt.figure(f"{(m, n)} {zernike_pol.get_polynomial_name(True)}: {round(alpha, 2)}*wavelength {fig_id}", figsize=(6, 6))
        plt.imshow(kernel, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()
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
        Zernike amplitudes (the expansion coefficients) in physical units.
    phi : float
        Angle on the pupil (entrance pupil of micro-objective) coordinates (for integration).
    p : float or np.ndarray
        Integration interval on the pupil (entrance pupil of micro-objective) radius or radius as float number.
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
        Zernike amplitudes (the expansion coefficients) in physical units.
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
        Zernike amplitudes (the expansion coefficients) in physical units.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).
    n_int_phi_points : int
        Number of integration points used for integration on the polar angle of the entrance pupil (from the range [0.0, 2pi]).

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    h_phi = 2.0*pi/n_int_phi_points; even_sum = 0.0j; odd_sum = 0.0j
    # Vectorized or parallelized form of for loop for even and odd phi-s
    even_sums = [radial_integral_pols(polynomials, amplitudes, r, theta, i*h_phi, n_int_r_points) for i in range(2, n_int_phi_points-2, 2)]
    even_sums = np.asarray(even_sums); even_sum = np.sum(even_sums)
    odd_sums = [radial_integral_pols(polynomials, amplitudes, r, theta, i*h_phi, n_int_r_points) for i in range(1, n_int_phi_points-1, 2)]
    odd_sums = np.asarray(odd_sums); odd_sum = np.sum(odd_sums)
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
        Zernike amplitudes (the expansion coefficients) in physical units used for the wavelength specification (e.g., \u00B5m).
        Note that the normalized Zernike polynomials are used, so their coefficients are normalized to the specified wavelength.
    len2pixels : float
        Relation between length in physical units (the same as the provided wavelength) and pixels.
    wavelength : float
        Wavelength (\u03BB) in physical units (e.g., \u00B5m) of the light used for calculations (in imaging).
    NA : float
        Objective property.
    n_int_r_points : int, optional
        Number of points used for integration on the unit pupil radius from the range [0.0, 1.0]. The default is 320.
    n_int_phi_points : int, optional
        Number of points used for integration on the unit pupil angle from the range [0.0, 2\u03C0]. The default is 300.
    show_kernel : bool, optional
        Plot the calculated kernel interactively. The default is True.
    fig_title : str, optional
        Custom figure title. The default is None.
    normalize_values : bool, optional
        Normalize all values in the sense that the max kernel value = 1.0. The default is False.
    kernel_size : int
        Kernel size (2D matrix).
    fig_id : str, optional
        Some string id for the figure title. The default is "".
    verbose: bool, optional
        Flag for printing explicitly # of points calculated on each run and measure how long it takes to calculate it.

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.

    """
    k = 2.0*pi/wavelength; size = kernel_size  # Calculate angular frequency (k) and reassign size for further using
    # Make kernel with odd sizes for precisely centering the kernel (in the center of an image)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros(shape=(size, size)); i_center = size//2; j_center = size//2
    # Print note about calculation duration
    if size > 25 and not suppress_warns:
        if abs(n_int_phi_points - 300) < 40 and abs(n_int_r_points - 320) < 50:
            print(f"Estimated calc. time: {int(round(size*size*38.5/1000, 0))} sec.")  # estimated based on tests only
        else:
            print("Calculation may take from several dozens of seconds to minutes")
    # Check that the calibration coefficient is sufficient for calculation
    pixel_size_nyquist = 0.5*0.61*wavelength/NA
    if len2pixels > pixel_size_nyquist and not suppress_warns:
        __warn_message = f"\nProvided calibration coefficient {len2pixels} {um_char}/pixels isn't sufficient enough"
        __warn_message += f" (defined by the relation between Nyquist freq. and the optical resolution: 0.61{lambda_char}/NA)"
        warnings.warn(__warn_message)
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
        warnings.warn(__warn_message)
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

    Returns
    -------
    convolved_img : numpy.ndarray
        Result of convolution (used scipy.ndimage.convolve).

    """
    img_type = img.dtype; img = np.copy(img)  # get the image type and copy its content to a new container
    convolved_img = convolve(np.float64(img), psf_kernel, mode='reflect'); conv_coeff = np.sum(psf_kernel)  # convolution using scipy
    if conv_coeff > 0.0:
        convolved_img /= conv_coeff  # correct the convolution result by dividing to the kernel sum
    if scale2original:
        max_original_intensity = np.max(img); max_conv_pixel = np.max(convolved_img)
        scaling_factor = max_original_intensity / max_conv_pixel; convolved_img *= scaling_factor
    convolved_img = convolved_img.astype(dtype=img_type)  # converting convolved image to the initial image
    return convolved_img


# %% Save and read the calculated PSF matrices
def save_psf(psf_kernel: np.ndarray, NA: float, wavelength: float, pixel_size: float, expansion_coefficient: Union[float, np.ndarray],
             kernel_size: int, n_int_points_r: int, n_int_points_phi: int, zernike_pol, folder_path: str = None,
             overwrite: bool = True, additional_file_name: str = None) -> str:
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
        Amplitude(-s) (in other words) of the polynomial(-s).
    kernel_size : int
        Size of the PSF kernel.
    n_int_points_r : int
        Number of the used integration points on r.
    n_int_points_phi : int
        Number of the used integration points on phi.
    zernike_pol : ZernPol instance or Sequence[ZernPol]
        Zernike polynomial(-s).
    folder_path : str, optional
        Absolute path to the folder where the file will be saved. The default is None.
    overwrite : bool, optional
        Flag for letting overwriting of the existing file. The default is True.
    additional_file_name : str
        Additional to the composed file name string, e.g. unique addition. The default is None.

    Returns
    -------
    str
        Absolute path to the file.

    """
    single_pol_used = False; osa_indices = []  # flag for saving different parameters
    if not hasattr(zernike_pol, '__len__'):
        (m, n) = define_orders(zernike_pol); single_pol_used = True  # get polynomial orders
    else:
        for zernpol in zernike_pol:
            osa_indices.append(zernpol.get_indices()[1])
    # Checking the provided folder or creating the folder for storing files
    if folder_path is None or len(folder_path) == 0 or not Path(folder_path).is_dir():
        working_folder = Path(__file__).cwd(); saved_psfs_folder = Path(working_folder).joinpath("saved_psfs")
        if not saved_psfs_folder.is_dir():
            saved_psfs_folder.mkdir()
        print("Auto assigned folder for saving calculated PSF kernel:", saved_psfs_folder)
    else:
        if Path(folder_path).is_dir():
            saved_psfs_folder = Path(folder_path)
    # Save provided PSF kernel with the provided parameters
    if additional_file_name is not None and len(additional_file_name) > 0:
        if single_pol_used:
            json_file_path = saved_psfs_folder.joinpath(f"psf_{(m, n)}_{additional_file_name}_{expansion_coefficient}.json")
        else:
            json_file_path = saved_psfs_folder.joinpath(f"psf_{osa_indices}_{additional_file_name}_{expansion_coefficient}.json")
    else:
        if single_pol_used:
            json_file_path = saved_psfs_folder.joinpath(f"psf_{(m, n)}_{expansion_coefficient}.json")
        else:
            json_file_path = saved_psfs_folder.joinpath(f"psf_{osa_indices}_{expansion_coefficient}.json")
    # Data composing for recording
    data4serialization = {}   # python dictionary is similar to the JSON file structure and can be dumped directly there
    data4serialization['PSF Kernel'] = psf_kernel.tolist(); data4serialization['NA'] = NA; data4serialization['Wavelength'] = wavelength
    data4serialization["Pixel Size"] = pixel_size; data4serialization["Kernel Size"] = kernel_size
    data4serialization["# of integration points R"] = n_int_points_r; data4serialization["# of integration points angle"] = n_int_points_phi
    if single_pol_used:
        data4serialization["Expansion Coefficient"] = expansion_coefficient; data4serialization["Polynomial"] = zernike_pol.get_indices()[1]
    else:
        data4serialization["Amplitudes"] = expansion_coefficient.tolist(); data4serialization["Polynomials"] = osa_indices
    # File presence check and recording
    if json_file_path.exists() and not overwrite:
        _warn_message = "The file already exists, the content won't be overwritten."; warnings.warn(_warn_message)
    if not json_file_path.exists() or (json_file_path.exists() and overwrite):
        with open(json_file_path, 'w') as json_write_file:
            json.dump(data4serialization, json_write_file)
    return str(json_file_path.absolute())


def read_psf(file_path: str) -> dict:
    """
    Read the saved PSF data from the *json file.

    Parameters
    ----------
    file_path : str
        Absolute path to the *json file.

    Returns
    -------
    dict
        Data stored in the *json file, the used keys: 'PSF kernel', 'NA', 'Wavelength', 'Calibration (um/pixels)'.

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
        In pixels.
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
__all__ = ['get_psf_kernel', 'save_psf', 'read_psf', 'convolute_img_psf', 'radial_integral_s', 'get_kernel_size', 'radial_integral',
           'get_kernel_size', 'um_char', 'lambda_char', 'get_bumped_circle', 'get_psf_kernel_zerns']

# %% Tests
if __name__ == '__main__':
    from zernpy import ZernPol
    plt.ion(); plt.close('all')  # close all plots before plotting new ones
    # Physical parameters of a system (an objective)
    wavelength = 0.55  # in micrometers
    NA = 0.95  # objective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length for an image)
    # Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
    resolution = 0.61*wavelength/NA  # ultimate theoretical physical resolution of an objective
    pixel_size_nyquist = 0.5*resolution  # Nyquist's resolution needed for using theoretical physical resolution above
    pixel_size = 0.95*pixel_size_nyquist  # the relation between um / pixels for calculating the coordinate in physical units for each pixel

    # Flags for performing tests
    check_zero_case = True  # checking that integral equation is corresponding to the Airy pattern (zero case)
    check_sign_coeff = False  # checking the same amplitude applied for the same polynomial (trefoil)
    check_performance_optimizations = False  # checking optimization of calculations
    check_various_pols = False  # checking the shape of some Zernike polynomials for comparing with the link below
    check_warnings = False  # flag for checking the warning producing
    check_io = False  # check save / read kernels
    show_convolution_results = False  # check result of convolution of several kernel with the disk

    # Definition of some Zernike polynomials for further tests
    pol1 = (0, 0); pol2 = (-1, 1); pol3 = (0, 2); pol4 = (-2, 2); pol5 = (-3, 3); pol6 = (2, 2); pol7 = (-1, 3); pol8 = (0, 4)
    pol9 = (-4, 4); pol7z = ZernPol(m=pol7[0], n=pol7[1])
    pol1z = ZernPol(m=pol1[0], n=pol1[1]); pol2z = ZernPol(m=pol2[0], n=pol2[1]); pol3z = ZernPol(m=pol3[0], n=pol3[1])

    if check_zero_case:
        kern_zc = get_psf_kernel(pol1z, len2pixels=wavelength/5.5, alpha=-0.4, wavelength=0.55, NA=0.65,
                                 show_kernel=True, normalize_values=True)
        kern_zc_ref = get_psf_kernel(pol1z, len2pixels=wavelength/5.5, alpha=-0.4, wavelength=0.55, NA=0.65, airy_pattern=True,
                                     show_kernel=True, normalize_values=True)
        diff = kern_zc_ref - kern_zc; plt.figure("Difference Airy and Piston", figsize=(6, 6))
        plt.imshow(diff, cmap=plt.cm.viridis, origin='upper')

    if check_sign_coeff:
        kern_sign_n = get_psf_kernel(pol5, len2pixels=pixel_size, alpha=-0.5, wavelength=wavelength, NA=NA, normalize_values=False)
        kern_sign_p = get_psf_kernel(pol5, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=False)

    if check_performance_optimizations:
        t1 = time.perf_counter()
        kern_sign = get_psf_kernel(ZernPol(m=-3, n=3), len2pixels=pixel_size, alpha=0.2, wavelength=wavelength,
                                   NA=NA, normalize_values=True, show_kernel=False)
        print("Calc. time for 'for loops' form ms:", int(round(1000*(time.perf_counter() - t1), 0)))
        t1 = time.perf_counter()
        kern_sign2 = get_psf_kernel(ZernPol(m=-3, n=3), len2pixels=pixel_size, alpha=0.2, wavelength=wavelength,
                                    NA=NA, normalize_values=True, show_kernel=False, test_vectorized=True, fig_id="Vector. Form")
        print("Calc. time for 'vectorized' form ms:", int(round(1000*(time.perf_counter() - t1), 0)))
        t1 = time.perf_counter()
        kern_sign3 = get_psf_kernel(ZernPol(m=-3, n=3), len2pixels=pixel_size, alpha=0.2, wavelength=wavelength,
                                    NA=NA, normalize_values=True, show_kernel=False, test_parallel=True, fig_id="Parallel. Form")
        print("Calc. time for 'parallelized' form ms:", int(round(1000*(time.perf_counter() - t1), 0)))
        kern_diff1 = np.round(kern_sign - kern_sign2, 9); kern_diff2 = np.round(kern_sign - kern_sign3, 9)

    if check_various_pols:
        kern_def = get_psf_kernel(pol3z, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True,
                                  show_kernel=True)
        kern_ast = get_psf_kernel(pol6, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True,
                                  show_kernel=True)
        kern_coma = get_psf_kernel(pol7, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True,
                                   show_kernel=True)
        kern_spher = get_psf_kernel(pol8, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength,
                                    NA=NA, normalize_values=True, show_kernel=True)
        kern_4foil = get_psf_kernel(pol9, len2pixels=pixel_size, alpha=0.5, wavelength=wavelength,
                                    NA=NA, normalize_values=True, show_kernel=True)

    if check_warnings:
        kern_def = get_psf_kernel(pol3z, len2pixels=1.0, alpha=0.5, wavelength=wavelength, NA=NA, normalize_values=True)
    if show_convolution_results:
        # Generate the ideal centralized circle with the blurred edges
        radius = 4.0; sample = get_bumped_circle(radius); plt.figure("Sample disk"); m_center, n_center = sample.shape
        m_center = m_center // 2; n_center = n_center // 2; axes_img = plt.imshow(sample, cmap=plt.cm.viridis, origin='upper')
        plt.tight_layout(); axes_img.axes.add_patch(Circle((n_center, m_center), radius, edgecolor='red', facecolor='none'))
        # Visualize results of convolution with various PSFs
        kern_def = get_psf_kernel(pol3z, len2pixels=pixel_size, alpha=0.7, wavelength=wavelength,
                                  NA=NA, normalize_values=True, show_kernel=True)
        conv_def = convolute_img_psf(img=sample, psf_kernel=kern_def, scale2original=True)
        plt.figure("Defocused"); axes_img2 = plt.imshow(conv_def, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()
        axes_img2.axes.add_patch(Circle((n_center, m_center), radius, edgecolor='red', facecolor='none'))
        kern_coma = get_psf_kernel(pol7z, len2pixels=pixel_size, alpha=-0.45, wavelength=wavelength,
                                   NA=NA, normalize_values=True, show_kernel=True)
        conv_coma = convolute_img_psf(img=sample, psf_kernel=kern_coma, scale2original=True)
        plt.figure("*Coma"); axes_img3 = plt.imshow(conv_coma, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()
        axes_img3.axes.add_patch(Circle((n_center, m_center), radius, edgecolor='red', facecolor='none'))
