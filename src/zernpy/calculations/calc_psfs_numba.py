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
from typing import Union

# %% Checking and import the numba library for speeding up the calculation
methods_compiled = False  # flag for storing if the methods compiled
try:
    from numba import njit
except ModuleNotFoundError:
    pass


# %% Local (package-scoped) imports - result in TypingError (Cannot determine Numba type of <class 'function'>)

# %% Module parameters
__docformat__ = "numpydoc"
um_char = "\u00B5"  # Unicode char code for micrometers
lambda_char = "\u03BB"  # Unicode char code for lambda (wavelength)
pi_char = "\u03C0"  # Unicode char code for pi


# %% Airy profile for Z(0, 0) ('airy_ref_pattern' cannot be compiled, deleted)

# %% Exchange ZernPol class call to calculation functions
@njit
def zernpol_value(orders: tuple, r: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> np.ndarray:
    """
    Provide composed Zernike polynomial value calculation for compilation by numba.

    Parameters
    ----------
    orders : tuple
        Orders of a Zernike polynomial.
    r : Union[float, np.ndarray]
        Radial values from an unit circle (range [0.0, 1.0]).
    theta : Union[float, np.ndarray]
        Angle for polar coordinates.

    Returns
    -------
    np.ndarray
        Polynomial value(-s).

    """
    m, n = orders  # transfer definition of a polynomial
    # Normalization factor
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2*(n + 1))
    # Triangular function (cannot be used as the import from a module)
    if m >= 0:
        triangular = np.cos(m*theta)
    else:
        triangular = -np.sin(m*theta)
    # Radial function exact definition for compilation (cannot be used as the import from a module)
    # 0th order
    if (m == 0) and (n == 0):
        radial = np.power(r, 0)
    # 1st order
    elif ((m == -1) and (n == 1)) or ((m == 1) and (n == 1)):
        radial = r
    # 2nd order
    elif ((m == -2) and (n == 2)) or ((m == 2) and (n == 2)):
        radial = np.power(r, 2)  # r^2
    elif (m == 0) and (n == 2):
        radial = 2.0*np.power(r, 2) - 1.0  # 2r^2 - 1
    # 3rd order
    elif ((m == -3) and (n == 3)) or ((m == 3) and (n == 3)):
        radial = np.power(r, 3)  # r^3
    elif ((m == -1) and (n == 3)) or ((m == 1) and (n == 3)):
        radial = 3.0*np.power(r, 3) - 2.0*r  # 3r^3 - 2r
    # 4th order
    elif ((m == -4) and (n == 4)) or ((m == 4) and (n == 4)):
        radial = np.power(r, 4)  # r^4
    elif ((m == -2) and (n == 4)) or ((m == 2) and (n == 4)):
        radial = 4.0*np.power(r, 4) - 3.0*np.power(r, 2)  # 4r^4 - 3r^2
    elif (m == 0) and (n == 4):
        radial = 6.0*np.power(r, 4) - 6.0*np.power(r, 2) + 1.0  # 6r^4 - 6r^2 + 1
    # 5th order
    elif ((m == -5) and (n == 5)) or ((m == 5) and (n == 5)):
        radial = np.power(r, 5)  # r^5
    elif ((m == -3) and (n == 5)) or ((m == 3) and (n == 5)):
        radial = 5.0*np.power(r, 5) - 4.0*np.power(r, 3)  # 5r^5 - 4r^3
    elif ((m == -1) and (n == 5)) or ((m == 1) and (n == 5)):
        radial = 10.0*np.power(r, 5) - 12.0*np.power(r, 3) + 3.0*r  # 10r^5 - 12r^3 + 3r
    # 6th order
    elif ((m == -6) and (n == 6)) or ((m == 6) and (n == 6)):
        radial = np.power(r, 6)  # r^6
    elif ((m == -4) and (n == 6)) or ((m == 4) and (n == 6)):
        radial = 6.0*np.power(r, 6) - 5.0*np.power(r, 4)  # 6r^6 - 5r^4
    elif ((m == -2) and (n == 6)) or ((m == 2) and (n == 6)):
        radial = 15.0*np.power(r, 6) - 20.0*np.power(r, 4) + 6.0*np.power(r, 2)  # 15r^6 - 20r^4 + 6r^2
    elif (m == 0) and (n == 6):
        radial = 20.0*np.power(r, 6) - 30.0*np.power(r, 4) + 12.0*np.power(r, 2) - 1.0  # 20r^6 - 30r^4 + 12r^2 - 1
    # 7th order
    elif ((m == -7) and (n == 7)) or ((m == 7) and (n == 7)):
        radial = np.power(r, 7)  # r^7
    elif ((m == -5) and (n == 7)) or ((m == 5) and (n == 7)):
        radial = 7.0*np.power(r, 7) - 6.0*np.power(r, 5)  # 7r^7 - 6r^5
    elif ((m == -3) and (n == 7)) or ((m == 3) and (n == 7)):
        radial = 21.0*np.power(r, 7) - 30.0*np.power(r, 5) + 10.0*np.power(r, 3)  # 21r^7 - 30r^5 + 10r^3
    elif ((m == -1) and (n == 7)) or ((m == 1) and (n == 7)):
        radial = 35.0*np.power(r, 7) - 60.0*np.power(r, 5) + 30.0*np.power(r, 3) - 4.0*r  # 35r^7 - 60r^5 + 30r^3 - 4r
    # 8th order
    elif ((m == -6) and (n == 8)) or ((m == 6) and (n == 8)):
        radial = 8.0*np.power(r, 8) - 7.0*np.power(r, 6)  # 8r^8 - 7r^6
    elif ((m == -4) and (n == 8)) or ((m == 4) and (n == 8)):
        radial = 28.0*np.power(r, 8) - 42.0*np.power(r, 6) + 15.0*np.power(r, 4)  # 28r^8 - 42r^6 + 15r^4
    elif ((m == -2) and (n == 8)) or ((m == 2) and (n == 8)):
        # 56r^8 - 105r^6 + 60r^4 - 10r^2
        radial = 56.0*np.power(r, 8) - 105.0*np.power(r, 6) + 60.0*np.power(r, 4) - 10.0*np.power(r, 2)
    elif (m == 0) and (n == 8):
        # 70r^8 - 140r^6 + 90r^4 - 20r^2 + 1
        radial = 70.0*np.power(r, 8) - 140.0*np.power(r, 6) + 90.0*np.power(r, 4) - 20.0*np.power(r, 2) + 1.0
    # 9th order
    elif ((m == -7) and (n == 9)) or ((m == 7) and (n == 9)):
        radial = 9.0*np.power(r, 9) - 8.0*np.power(r, 7)  # 9r^9 - 8r^7
    elif ((m == -5) and (n == 9)) or ((m == 5) and (n == 9)):
        radial = 36.0*np.power(r, 9) - 56.0*np.power(r, 7) + 21.0*np.power(r, 5)  # 36r^9 - 56r^7 + 21r^5
    elif ((m == -3) and (n == 9)) or ((m == 3) and (n == 9)):
        # 84r^9 - 168r^7 + 105r^5 - 20r^3
        radial = 84.0*np.power(r, 9) - 168.0*np.power(r, 7) + 105.0*np.power(r, 5) - 20.0*np.power(r, 3)
    elif ((m == -1) and (n == 9)) or ((m == 1) and (n == 9)):
        # 126r^9 - 280r^7 + 210r^5 - 60r^3 + 5r
        radial = 126.0*np.power(r, 9) - 280.0*np.power(r, 7) + 210.0*np.power(r, 5) - 60.0*np.power(r, 3) + 5.0*r
    # 10th order
    elif ((m == -8) and (n == 10)) or ((m == 8) and (n == 10)):
        radial = 10.0*np.power(r, 10) - 9.0*np.power(r, 8)  # 10r^10 - 9r^8
    elif ((m == -6) and (n == 10)) or ((m == 6) and (n == 10)):
        radial = 45.0*np.power(r, 10) - 72.0*np.power(r, 8) + 28.0*np.power(r, 6)  # 45r^10 - 72r^8 + 28r^6
    elif ((m == -4) and (n == 10)) or ((m == 4) and (n == 10)):
        # 120r^10 - 252r^8 + 168r^6 - 35r^4
        radial = 120.0*np.power(r, 10) - 252.0*np.power(r, 8) + 168.0*np.power(r, 6) - 35.0*np.power(r, 4)
    elif ((m == -2) and (n == 10)) or ((m == 2) and (n == 10)):
        # 210r^10 - 504r^8 + 420r^6 - 140r^4 + 15r^2
        radial = 210.0*np.power(r, 10) - 504.0*np.power(r, 8) + 420.0*np.power(r, 6) - 140.0*np.power(r, 4) + 15.0*np.power(r, 2)
    elif (m == 0) and (n == 10):
        # 252r^10 - 630r^8 + 560r^6 - 210r^4 + 30r^2 - 1
        radial = 252.0*np.power(r, 10) - 630.0*np.power(r, 8) + 560.0*np.power(r, 6) - 210.0*np.power(r, 4) + 30.0*np.power(r, 2) - 1.0
    elif n > 7 and abs(m) == n:  # equation for high order polynomials (equal orders)
        return np.power(r, n)
    elif n > 10 and abs(m) == n-2:  # equation for high order polynomials (orders with abs(m) == n-2)
        return float(n)*np.power(r, n) - float(n-1)*np.power(r, n-2)
    # Polynomial value as the multiplication of calculated above components
    return norm*triangular*radial


# %% PSF pixel value calc.
@njit
def diffraction_integral_r_comp(orders: tuple, alpha: float, phi: float, p: Union[float, np.ndarray], theta: float, r: float) -> np.ndarray:
    """
    Diffraction integral function for the formed image point (see the references as the sources of the equation).

    Parameters
    ----------
    orders : (m, n)
        Orders of a Zernike polynomial.
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
    phase_arg = (alpha*zernpol_value(orders, p, phi) - r*p*np.cos(phi - theta))*1j
    return np.exp(phase_arg)*p


@njit
def radial_integral_comp(orders: tuple, r: float, theta: float, phi: float, alpha: float, n_int_r_points: int) -> complex:
    """
    Make integration of the diffraction integral on the radius of the entrance pupil.

    Parameters
    ----------
    orders : (m, n)
        Orders of a Zernike polynomial.
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
    h_p = 1.0/n_int_r_points; p = np.arange(h_p, 1.0, h_p)
    fa = diffraction_integral_r_comp(orders, alpha, phi, 0.0, theta, r)
    fb = diffraction_integral_r_comp(orders, alpha, phi, 1.0, theta, r)
    ang_int = np.sum(diffraction_integral_r_comp(orders, alpha, phi, p, theta, r)) + 0.5*(fa + fb)
    return h_p*ang_int


# %% Testing various speeding up calculation approaches
@njit
def get_psf_point_r_comp(orders: tuple, r: float, theta: float, alpha: float, n_int_r_points: int, n_int_phi_points: int) -> float:
    """
    Calculate PSF point for the kernel using Parallel class from the joblib library.

    Parameters
    ----------
    orders : (m, n)
        Orders of a Zernike polynomial.
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
    even_sums = [radial_integral_comp(orders, r, theta, i*h_phi, alpha, n_int_r_points) for i in range(2, n_int_phi_points-2, 2)]
    even_sums = np.asarray(even_sums); even_sum = np.sum(even_sums)
    odd_sums = [radial_integral_comp(orders, r, theta, i*h_phi, alpha, n_int_r_points) for i in range(1, n_int_phi_points-1, 2)]
    odd_sums = np.asarray(odd_sums); odd_sum = np.sum(odd_sums)
    # Simpson integration rule implementation
    yA = radial_integral_comp(orders, r, theta, 0.0, alpha, n_int_r_points)
    yB = radial_integral_comp(orders, r, theta, 2.0*pi, alpha, n_int_r_points)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


# %% PSF kernel calc. (accelerated)
def get_psf_kernel_comp(zernike_pol, len2pixels: float, alpha: Union[float, np.ndarray], wavelength: float, NA: float, n_int_r_points: int = 320,
                        n_int_phi_points: int = 300, show_kernel: bool = False, fig_title: str = None, normalize_values: bool = False,
                        kernel_size: int = 3, fig_id: str = "", suppress_warns: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Calculate centralized matrix (kernel) with the PSF mask values.

    Parameters
    ----------
    zernike_pol : ZernPol or Sequence[ZernPol]
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
    kernel_size : int, optional
        Custom kernel size, if not provided, then the size will be estimated based on the parameters. The default is 3.
    fig_id : str, optional
        Some string id for the figure title. The default is "".
    suppress_warns : bool, optional
        Flag for suppressing any thrown warnings. The default is False.
    verbose: bool, optional
        Flag for printing explicitly # of points calculated on each run and measure how long it takes to calculate it. The default is False.

    Raises
    ------
    ValueError
        If the Zernike polynomial has the radial order > 10 and (abs(m) != n or abs(m) != n-2).

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.

    """
    # Convert provided absolute value of Zernike expansion coefficient (in um) into fraction of wavelength
    size = kernel_size; k = 2.0*pi/wavelength  # Calculate angular frequency (k)
    # Make kernel with odd sizes for precisely centering the kernel (in the center of an image)
    if size % 2 == 0:
        size += 1
    # Provide performance tip if the provided kernel size is quite big for calculations
    if size > 85 and not suppress_warns:
        __warn_message = f"\nCalculation of provided kernel size ({size}x{size}) may take more than 20 seconds"
        warnings.warn(__warn_message); __warn_message = ""
    kernel = np.zeros(shape=(size, size)); i_center = size//2; j_center = size//2
    # Get the orders of polynomial and check if the equation for compilation was implemented
    single_polynomial_provided = False  # flag for using single polynomial functions
    if not hasattr(zernike_pol, "__len__") and isinstance(alpha, float):
        alpha /= wavelength  # normalize to wavelength (physical units)
        m, n = zernike_pol.get_mn_orders(); single_polynomial_provided = True
        if n > 10 and (abs(m) != n or abs(m) != n-2):
            raise ValueError(f"The calculation PSF function isn't implemented for these orders: {m, n}")
    else:
        if not isinstance(alpha, np.ndarray):
            alpha = np.asarray(alpha)
        polynomials_orders = []  # for checking and providing for further compilation polynomials in a tuple
        for pol in zernike_pol:
            m, n = pol.get_mn_orders(); polynomials_orders.append((m, n))
            if n > 10 and (abs(m) != n or abs(m) != n-2):
                raise ValueError(f"The calculation PSF function isn't implemented for these orders: {m, n}")
        polynomials_orders = tuple(polynomials_orders)  # convert list to tuple
    # Check that the calibration coefficient is sufficient for calculation
    pixel_size_nyquist = 0.5*0.61*wavelength/NA
    if len2pixels > pixel_size_nyquist and not suppress_warns:
        __warn_message = f"\nProvided calibration coefficient {len2pixels} {um_char}/pixels isn't sufficient enough"
        __warn_message += f" (defined by the relation between Nyquist freq. and the optical resolution: 0.61{lambda_char}/NA)"
        warnings.warn(__warn_message); __warn_message = ""
    # Calculate the PSF kernel for usage in convolution operation
    if verbose:
        calculated_points = 0  # for explicit showing of performance
        show_each_tenth_point = False; checking_point = 1  # flag and value for shortening print output
        if 100 < size*size < 651:
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
            if single_polynomial_provided:
                kernel[i, j] = get_psf_point_r_comp((m, n), distance, theta, alpha, n_int_r_points, n_int_phi_points)
            else:
                kernel[i, j] = get_psf_point_r_pols_comp(polynomials_orders, alpha, distance, theta, n_int_r_points, n_int_phi_points)
            if verbose:
                calculated_points += 1; passed_time_ms = int(round(1000*(time.perf_counter() - t1), 0))
                if show_each_tenth_point and (calculated_points == 1 or calculated_points == checking_point):
                    print(f"Calculated point #{calculated_points} from {size*size}, took: {passed_time_ms} ms", flush=True)
                    if calculated_points == checking_point:
                        checking_point += 10
                elif (not show_each_tenth_point and not size*size >= 651):
                    print(f"Calculated point #{calculated_points} from {size*size}, took: {passed_time_ms} ms", flush=True)
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
            if not hasattr(zernike_pol, "__len__"):
                plt.figure(f"{zernike_pol.get_mn_orders()} {zernike_pol.get_polynomial_name(True)}: {round(alpha, 2)}*wavelength {fig_id}",
                           figsize=(6, 6))
            else:
                plt.figure(f"Sum of provided #{len(zernike_pol)} of polynomials {fig_id}", figsize=(6, 6))
        plt.imshow(kernel, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()
    return kernel


# %% PSF calc. for several polynomials
@njit
def pol_sums(polynomials_orders: tuple, amplitudes: np.ndarray, p: float, phi: float) -> float:
    """
    Wrap calculation of polynomials values sum for compilation.

    Parameters
    ----------
    polynomials_orders : tuple
        DESCRIPTION.
    amplitudes : np.ndarray
        DESCRIPTION.
    p : float
        Value from the calling function.
    phi : float
        Value from the calling function.

    Returns
    -------
    sum_pols : float
        Sum of polynomials values in the polar coordinates point.

    """
    for i, orders in enumerate(polynomials_orders):
        if i == 0:
            sum_pols = amplitudes[i]*zernpol_value(orders, p, phi)
        else:
            sum_pols += amplitudes[i]*zernpol_value(orders, p, phi)
    return sum_pols


@njit
def diffraction_integral_r_pols_comp(polynomials_orders: tuple, amplitudes: np.ndarray, phi: float,
                                     p: float, theta: float, r: float) -> float:
    """
    Diffraction integral function for the formed image point (see the references as the sources of the equation).

    Parameters
    ----------
    polynomials_orders : tuple
        Tuple with several orders of Zernike polynomials.
    amplitudes : np.ndarray
        Amplitudes of polynomials (RMS).
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
    phase_arg = (pol_sums(polynomials_orders, amplitudes, p, phi) - r*p*np.cos(phi - theta))*1j
    return np.exp(phase_arg)*p


@njit
def radial_integral_pols_comp(polynomials_orders: tuple, amplitudes: np.ndarray, r: float, theta: float,
                              phi: float, n_int_r_points: int) -> complex:
    """
    Make integration of the diffraction integral on the radius of the entrance pupil.

    Parameters
    ----------
    polynomials_orders : tuple
        Tuple with several orders of Zernike polynomials.
    amplitudes : np.ndarray
        Amplitudes of polynomials (RMS).
    r : float
        Radius on the image coordinates.
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
    # Integration on the pupil angle. Vectorized form of the trapezoidal rule
    h_p = 1.0/n_int_r_points; p = np.arange(h_p, 1.0, h_p)
    fa = diffraction_integral_r_pols_comp(polynomials_orders, amplitudes, phi, 0.0, theta, r)
    fb = diffraction_integral_r_pols_comp(polynomials_orders, amplitudes, phi, 1.0, theta, r)
    ang_int = np.sum(diffraction_integral_r_pols_comp(polynomials_orders, amplitudes, phi, p, theta, r)) + 0.5*(fa + fb)
    return h_p*ang_int


@njit
def get_psf_point_r_pols_comp(polynomials_orders: tuple, amplitudes: np.ndarray, r: float, theta: float,
                              n_int_r_points: int, n_int_phi_points: int) -> float:
    """
    Calculate PSF point for the kernel using Parallel class from the joblib library.

    Parameters
    ----------
    polynomials_orders : tuple
        Tuple with several orders of Zernike polynomials.
    amplitudes : np.ndarray
        Amplitudes of polynomials (RMS).
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
    even_sums = [radial_integral_pols_comp(polynomials_orders, amplitudes, r, theta, i*h_phi, n_int_r_points)
                 for i in range(2, n_int_phi_points-2, 2)]
    even_sums = np.asarray(even_sums); even_sum = np.sum(even_sums)
    odd_sums = [radial_integral_pols_comp(polynomials_orders, amplitudes, r, theta, i*h_phi, n_int_r_points)
                for i in range(1, n_int_phi_points-1, 2)]
    odd_sums = np.asarray(odd_sums); odd_sum = np.sum(odd_sums)
    # Simpson integration rule implementation
    yA = radial_integral_pols_comp(polynomials_orders, amplitudes, r, theta, 0.0, n_int_r_points)
    yB = radial_integral_pols_comp(polynomials_orders, amplitudes, r, theta, 2.0*pi, n_int_r_points)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


# %% Utility functions
def set_methods_compiled():
    """
    Reset the flag by external call.

    Returns
    -------
    None.

    """
    global methods_compiled; methods_compiled = True


# %% Define standard exports from this module
__all__ = ['get_psf_kernel_comp', 'methods_compiled', 'set_methods_compiled']

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

    # Flags for testing various scenarios
    test_single_polynomial = False; test_few_polynomials = True

    # Testing implemented calculations for single polynomial
    if test_single_polynomial:
        get_psf_kernel_comp(ZernPol(m=0, n=0), pixel_size*0.7, alpha=1.5, wavelength=wavelength, NA=NA, normalize_values=True,
                            show_kernel=True, kernel_size=11, verbose=True)
        print("*******************************************")
        get_psf_kernel_comp(ZernPol(m=0, n=4), pixel_size*0.7, alpha=0.75, wavelength=wavelength, NA=NA, normalize_values=True,
                            show_kernel=True, kernel_size=21, verbose=True)
    # For several (calculated their sum)
    if test_few_polynomials:
        # First, checking below sequentially all functions to be compilable
        # pols = ((-2, 2), (1, 3)); ampls = np.asarray([-0.4, 0.6])
        # diffraction_integral_r_pols_comp(pols, ampls, phi=0.2, p=0.1, theta=0.3, r=0.5)
        # radial_integral_pols_comp(pols, ampls, r=0.5, theta=0.3, phi=1.01, n_int_r_points=300)
        # get_psf_point_r_pols_comp(pols, ampls, r=0.5, theta=0.3, n_int_r_points=250, n_int_phi_points=320)
        # Second, test all at once for calling the function
        zp1 = ZernPol(m=-2, n=2); zp2 = ZernPol(m=0, n=2); zp3 = ZernPol(m=2, n=2); pols = (zp1, zp2, zp3); coeffs = (-0.86, 0.4, 0.7)
        get_psf_kernel_comp(pols, pixel_size*0.7, alpha=coeffs, wavelength=wavelength, NA=NA, normalize_values=True,
                            show_kernel=True, kernel_size=24, verbose=True)
