# -*- coding: utf-8 -*-
"""
Calculation accelerated by numba library compilation and plotting of associated with polynomials PSFs.

@author: Sergei Klykov  \n
@licence: MIT, @year: 2024

"""
# %% Global imports
import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import pi
import time
# import time

# %% Checking and import the numba library for speeding up the calculation
global numba_installed
try:
    from numba import jit
    numba_installed = True
except ModuleNotFoundError:
    numba_installed = False


# %% Module parameters
__docformat__ = "numpydoc"
um_char = "\u00B5"  # Unicode char code for micrometers
lambda_char = "\u03BB"  # Unicode char code for lambda (wavelength)
pi_char = "\u03C0"  # Unicode char code for pi


# %% Reference - Airy profile for Z(0, 0) ('airy_ref_pattern' cannot be compiled, deleted)

# %% PSF pixel value calc.
@jit  # gives TypingError (run for details)
def diffraction_integral_r(zernike_pol, alpha: float, phi: float, p, theta: float, r: float) -> np.array:
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
def get_psf_point_r_parallel(zernike_pol, r: float, theta: float, alpha: float, n_int_r_points: int, n_int_phi_points: int) -> float:
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
    even_sums = [radial_integral(zernike_pol, r, theta, i*h_phi, alpha, n_int_r_points) for i in range(2, n_int_phi_points-2, 2)]
    even_sums = np.asarray(even_sums); even_sum = np.sum(even_sums)
    odd_sums = [radial_integral(zernike_pol, r, theta, i*h_phi, alpha, n_int_r_points) for i in range(1, n_int_phi_points-1, 2)]
    odd_sums = np.asarray(odd_sums); odd_sum = np.sum(odd_sums)
    # Simpson integration rule implementation
    yA = radial_integral(zernike_pol, r, theta, 0.0, alpha, n_int_r_points)
    yB = radial_integral(zernike_pol, r, theta, 2.0*pi, alpha, n_int_r_points)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


# %% PSF kernel calc. (accelerated)
def get_psf_kernel_comp(zernike_pol, len2pixels: float, alpha: float, wavelength: float, NA: float, n_int_r_points: int = 320,
                        n_int_phi_points: int = 300, show_kernel: bool = False, fig_title: str = None, normalize_values: bool = False,
                        kernel_size: int = 3, fig_id: str = "",
                        test_vectorized: bool = False, suppress_warns: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Calculate centralized matrix with the PSF mask values.

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
        Plot the calculated kernel interactively. The default is True.
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
        Flag for printing explicitly # of points calculated on each run and measure how long it takes to calculate it.

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.

    """
    # Convert provided absolute value of Zernike expansion coefficient (in um) into fraction of wavelength
    alpha /= wavelength; k = 2.0*pi/wavelength  # Calculate angular frequency (k)
    size = kernel_size
    # Make kernel with odd sizes for precisely centering the kernel (in the center of an image)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros(shape=(size, size)); i_center = size//2; j_center = size//2
    # Print note about calculation duration
    if size > 30 and not suppress_warns:
        if abs(n_int_phi_points - 300) < 40 and abs(n_int_r_points - 320) < 50:
            print(f"Note that the estimated kernel size: {size}x{size}. Estimated calc. time: {int(round(size*size*38.5/1000, 0))} sec.")
        else:
            print(f"Note that the estimated kernel size: {size}x{size}, calculation may take from several dozens of seconds to minutes")
    # Check that the calibration coefficient is sufficient for calculation
    pixel_size_nyquist = 0.5*0.61*wavelength/NA
    if len2pixels > pixel_size_nyquist and not suppress_warns:
        __warn_message = f"Provided calibration coefficient {len2pixels} {um_char}/pixels isn't sufficient enough"
        __warn_message += f" (defined by the relation between Nyquist freq. and the optical resolution: 0.61{lambda_char}/NA)"
        warnings.warn(__warn_message)
    # Calculate the PSF kernel for usage in convolution operation
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
        if fig_title is not None and len(fig_title) > 0:
            plt.figure(fig_title, figsize=(6, 6))
        else:
            plt.figure(f"{zernike_pol.get_mn_orders()} {zernike_pol.get_polynomial_name(True)}: {round(alpha, 2)}*wavelength {fig_id}",
                       figsize=(6, 6))
        plt.imshow(kernel, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()
    return kernel


# %% Define standard exports from this module
__all__ = ['get_psf_kernel_comp']

# %% Tests
if __name__ == '__main__':
    from zernpy import ZernPol  # for polynomials initialization
    plt.ion(); plt.close('all')  # close all plots before plotting new ones
    # Physical parameters of a system (an objective)
    wavelength = 0.55  # in micrometers
    NA = 0.95  # objective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length for an image)
    # Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
    resolution = 0.61*wavelength/NA  # ultimate theoretical physical resolution of an objective
    pixel_size_nyquist = 0.5*resolution  # Nyquist's resolution needed for using theoretical physical resolution above
    pixel_size = 0.95*pixel_size_nyquist  # the relation between um / pixels for calculating the coordinate in physical units for each pixel

    # Flags for performing tests

    # Definition of some Zernike polynomials for further tests
    kernel0 = get_psf_kernel_comp(ZernPol(m=0, n=0), pixel_size*0.7, alpha=1.5, wavelength=wavelength, NA=NA,
                                  show_kernel=True, kernel_size=11)
