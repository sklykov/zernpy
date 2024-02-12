# -*- coding: utf-8 -*-
"""
Check calculation and plotting of associated with polynomials PSFs.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from scipy.special import jv
import warnings
from scipy.ndimage import convolve
from math import cos, pi
from zernpy import ZernPol
import time
import os
from skimage import io
from skimage.util import img_as_ubyte
import json
from math import e

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calc_zernike_pol import define_orders
else:
    from .calc_zernike_pol import define_orders

# %% Module parameters
__docformat__ = "numpydoc"
n_phi_points = 300; n_p_points = 320
# Physical parameters
wavelength = 0.55  # in micrometers
k = 2.0*np.pi/wavelength  # angular frequency
NA = 0.95  # microobjective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length for an image)
# Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
pixel_size = 0.14  # in micrometers, physical length in pixels (um / pixels)
pixel2um_coeff = k*NA*pixel_size  # coefficient used for relate pixels to physical units
pixel2um_coeff_plot = k*NA*(pixel_size/10.0)  # coefficient used for better plotting with the reduced pixel size for preventing pixelated


# %% Integral Functions - integration by trapezoidal rule first on radius (normalized on the pupil), after - on angle
def diffraction_integral_r(zernike_pol: ZernPol, alpha: float, phi: float, p: np.array, theta: float, r: float) -> np.array:
    """
    Diffraction integral function (see the references for the source of the equation).

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition.
    alpha : float
        Amplitude of the polynomial.
    phi : float
        Angle on the pupil coordinates.
    p : np.array
        Integration interval on the pupil coordinates.
    theta : float
        Angle on the image coordinates.
    r : float
        Radius on the image coordinates.

    References
    ----------
    [1] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf
    [2] https://opg.optica.org/ao/abstract.cfm?uri=ao-52-10-2062 (DOI: 10.1364/AO.52.002062)

    Returns
    -------
    numpy.ndarray
        Values of the diffraction integral.

    """
    # Multiplication by phi (in r*p*np.cos(...) only guides to scaling of the resulting PSF
    phase_arg = (alpha*zernike_pol.polynomial_value(p, phi) - r*p*np.cos(phi - theta))*1j
    return np.exp(phase_arg)*p


def radial_integral(zernike_pol: ZernPol, r: float, theta: float, phi: float, alpha: float) -> complex:
    """
    Make integration of the diffraction integral on the radius.

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    phi : float
        Angle on the pupil coordinates.
    alpha : float
        Amplitude of the polynomial.

    Returns
    -------
    complex
        Complex amplitude of the field as result of integration on the pupil radius coordinate.

    """
    # Integration on the pupil angle. Vectorized form of the trapezoidal rule
    h_p = 1.0/n_p_points; p = np.arange(start=h_p, stop=1.0, step=h_p)
    fa = diffraction_integral_r(zernike_pol, alpha, phi, 0.0, theta, r); fb = diffraction_integral_r(zernike_pol, alpha, phi, 1.0, theta, r)
    ang_int = np.sum(diffraction_integral_r(zernike_pol, alpha, phi, p, theta, r)) + 0.5*(fa + fb)
    return h_p*ang_int


def get_psf_point_r(zernike_pol, r: float, theta: float, alpha: float = 1.0) -> float:
    """
    Get the point for calculation of PSF.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or ZernPol
        Zernike polynomial definition.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    alpha : float, optional
        Amplitude of the polynomial. The default is 1.0.

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    # Check and initialize Zernike polynomial if provided only orders
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    if not isinstance(zernike_pol, ZernPol):
        zernike_pol = ZernPol(m=m, n=n)
    # Integration on the pupil radius using Simpson equation
    n_integral_points = n_phi_points; h_phi = 2.0*pi/n_integral_points
    even_sum = 0.0j; odd_sum = 0.0j
    for i in range(2, n_integral_points-2, 2):
        phi = i*h_phi; even_sum += radial_integral(zernike_pol, r, theta, phi, alpha)
    for i in range(1, n_integral_points-1, 2):
        phi = i*h_phi; odd_sum += radial_integral(zernike_pol, r, theta, phi, alpha)
    yA = radial_integral(zernike_pol, r, theta, 0.0, alpha); yB = radial_integral(zernike_pol, r, theta, 2.0*pi, alpha)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


# %% Integral Functions - integration on the radial external, angular - internal by the trapezoidal rule
def diffraction_integral_ang(zernike_pol: ZernPol, alpha: float, phi: np.array, p: float, theta: float, r: float) -> np.array:
    """
    Diffraction integral function (see the references for the source of the equation).

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition.
    alpha : float
        Amplitude of the polynomial.
    phi : float
        Integration interval of angles on the pupil coordinates.
    p : float
        Radius on the pupil coordinates.
    theta : float
        Angle on the image coordinates.
    r : float
        Radius on the image coordinates.

    References
    ----------
    [1] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf
    [2] https://opg.optica.org/ao/abstract.cfm?uri=ao-52-10-2062 (DOI: 10.1364/AO.52.002062)

    Returns
    -------
    numpy.ndarray
        Values of the diffraction integral.

    """
    phase_arg = (alpha*zernike_pol.polynomial_value(p, phi) - r*p*np.cos(phi - theta))*1j
    return np.exp(phase_arg)*p


def angular_integral(zernike_pol: ZernPol, r: float, theta: float, p: float, alpha: float) -> complex:
    """
    Make integration of the diffraction integral on the angle.

    Parameters
    ----------
    zernike_pol : ZernPol
        Zernike polynomial definition.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    p : float
        Radius on the pupil coordinates.
    alpha : float
        Amplitude of the polynomial.

    Returns
    -------
    complex
        Complex amplitude of the field as result of integration on the pupil radius coordinate.

    """
    # Integration on the pupil angle. Vectorized form of the trapezoidal rule
    h_phi = pi/n_phi_points; phi = np.arange(start=h_phi, stop=2.0*pi, step=h_phi)
    fa = diffraction_integral_ang(zernike_pol, alpha, 0.0, p, theta, r); fb = diffraction_integral_ang(zernike_pol, alpha, 2.0*pi, p, theta, r)
    ang_int = np.sum(diffraction_integral_ang(zernike_pol, alpha, phi, p, theta, r)) + 0.5*(fa + fb)
    return h_phi*ang_int


def get_psf_point_ang(zernike_pol, r: float, theta: float, alpha: float = 1.0) -> float:
    """
    Get the point for calculation of PSF.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or ZernPol
        Zernike polynomial definition.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    alpha : float, optional
        Amplitude of the polynomial. The default is 1.0.

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    # Check and initialize Zernike polynomial if provided only orders
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    if not isinstance(zernike_pol, ZernPol):
        zernike_pol = ZernPol(m=m, n=n)
    # Integration on the pupil radius using Simpson equation
    n_integral_points = n_p_points; h_p = 1.0/n_integral_points
    even_sum = 0.0j; odd_sum = 0.0j
    for i in range(2, n_integral_points-2, 2):
        p = i*h_p; even_sum += angular_integral(zernike_pol, r, theta, p, alpha)
    for i in range(1, n_integral_points-1, 2):
        p = i*h_p; odd_sum += angular_integral(zernike_pol, r, theta, p, alpha)
    yA = angular_integral(zernike_pol, r, theta, 0.0, alpha); yB = angular_integral(zernike_pol, r, theta, 1.0, alpha)
    integral_sum = (h_p/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum)
    return np.power(np.abs(integral_sum), 2)/(4.0*pi*pi)


# %% Attempt to speed up calculations by using provided in the Born & Wolf's handbook equation
def diffraction_int_approx_sum(zernike_pol: ZernPol, s: int, alpha: float, theta, r):
    m, n = define_orders(zernike_pol)
    if s == 0:
        c = 2.0
    else:
        c = 4.0
    coeffiicent = c*pow(1j, (m-1)*s)*np.cos(m*s*theta)
    h_p = 1.0/n_p_points; p_arr = np.arange(start=h_p, stop=1.0, step=h_p)
    eps_zero = 1E-11  # substituion of the exact 0.0 value
    if r <= 1E-12:
        r = eps_zero
    if alpha <= 1E-12:
        alpha = 1E-11
    fa = jv(s, alpha*zernike_pol.radial(eps_zero))*jv(m*s, eps_zero*r)*eps_zero  # f(0.0)
    fb = jv(s, alpha*zernike_pol.radial(1.0))*jv(m*s, r); fAB = 0.5*(fa + fb)
    integral_sum = h_p*np.sum(jv(s, alpha*zernike_pol.radial(p_arr))*jv(m*s, r*p_arr)*p_arr)
    return coeffiicent*(fAB + integral_sum)


def get_psf_point_bwap(zernike_pol, r: float, theta: float, alpha: float = 1.0) -> float:
    m, n = define_orders(zernike_pol)
    if not isinstance(zernike_pol, ZernPol):
        zernike_pol = ZernPol(m=m, n=n)
    s_max = 30; phases_sum = 0.0j
    for s_i in range(s_max):
        phases_sum += diffraction_int_approx_sum(zernike_pol, s_i, alpha, theta, r)
    integral_normalization = 1.0/(pi*pi)
    return integral_normalization*np.power(np.abs(phases_sum), 2)


# %% Nijboer's Thesis Functions
def radial_func(n: int, r: float) -> float:
    """
    Define the radial function Jn(r)/r, there Jn - Bessel function of 1st kind with order n.

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    r : float
        Radial parameter r.

    Returns
    -------
    float
        Evaluated function Jn(r)/r.

    """
    # Defining pure radial function (angular indenpendent) - Jv(r)/r
    if isinstance(r, int):  # Convert int to float explicitly
        r = float(r)
    radial = 0.0  # default value
    # Calculate value only for input r as the float number
    if isinstance(r, float):
        if abs(round(r, 12)) == 0.0:  # check that the argument provided with 0 value
            radial = round(2.0*pow(jv(n, 1E-11)/1E-11, 2), 11)  # approximation of the limit for the special condition jv(x)/x, where x -> 0
        else:
            radial = round(2.0*pow(jv(n, r)/r, 2), 11)
    return radial


def radial_ampl_func(n: int, r: np.ndarray) -> np.ndarray:
    """
    Calculate the radial amplitude functions used in the Nijboer's thesis as Jv(r)/r.

    Parameters
    ----------
    n : int
        Radial order of the Zernike polynomial.
    r : np.ndarray
        Radii on the pupil coordinates.
        Note that the array can only start with the special zero point (r[0] = 0.0).

    Reference
    ---------
    [1] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf

    Returns
    -------
    radial_ampl_f : np.ndarray
        Jv(r)/r function values.

    """
    # Expanse the calculation for using the numpy array, assuming it starting with 0.0
    radial_ampl_f = None
    print(type(r))
    if isinstance(r, np.ndarray):
        r = np.round(r, 12)
        if r[0] == 0.0:
            radial_ampl_f = np.zeros(shape=(r.shape[0]))
            r1 = r[1:]; radial_ampl_f[1:] = jv(n, r1)/r1
            radial_ampl_f[0] = jv(n, 1E-11)/1E-11
    return radial_ampl_f


def get_aberrated_psf(zernike_pol, r: float, theta: float, alpha: float = 1.0) -> float:
    """
    Get the precalculated expansion of diffraction integral based on the Nijboer's thesis.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or ZernPol
        Zernike polynomial definition.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    alpha : float, optional
        Amplitude of the polynomial. The default is 1.0.

    Reference
    ---------
    [1] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf

    Returns
    -------
    float
        PSF point as |U|**2.

    """
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    # The analytical equation could be found in the Nijboer's thesis
    if m == 0 and n == 0:  # piston value
        x1 = radial_func(1, r)
        return x1*x1*(1.0 + alpha*alpha)
    # Approximation of the (-1, 1) polynomial (Y tilt). Note that each polynomial should be somehow evaluated or overall equation used
    elif m == -1 and n == 1:
        alpha = round(alpha, 4)  # rounding with extended precision, commonly alpha up to 0.001 useful
        x1 = radial_func(1, r); x2 = radial_func(2, r); x3 = radial_func(3, r); x4 = radial_func(4, r)
        c1 = -(alpha*alpha)/8.0; c2 = (alpha*alpha)/4.0; c4 = pow(alpha, 3)/24.0; c5 = pow(alpha, 3)/132.0
        u = x1 + alpha*x2*cos(theta) + c1*(x1 - x3) + c2*x3*cos(2.0*theta) + c4*x4*cos(3.0*theta) + c5*(x4 - 2.0*x1)
        if alpha > 1.0:
            c6 = pow(alpha, 4)/192.0; x5 = radial_func(5, r)
            u += c6*(x1 + -1.5*x3 + 0.5*x5 - cos(2.0*theta)*(3.0*x3 - x5) + cos(4.0*theta)*x5)
        if alpha > 2.0:
            __warn_message = f"For such big aberration amplitude ({alpha}) not precise equation was found, use it with caution!"
            warnings.warn(__warn_message)
        return u*u
    # Approximation of the (-2, 2) polynomial (defocus)
    elif m == 0 and n == 2:
        x1 = radial_func(1, r); x3 = radial_func(3, r); x5 = radial_func(5, r); x7 = radial_func(7, r)
        c1 = -(alpha*alpha)/2.0; c2 = pow(alpha, 3)/6.0
        a = x1 + c1*((4.0/3.0)*x1 - x3 + (2.0/3.0)*x5); b = alpha*x3 + c2*(-7.0*x1 + 4.8*x3 + 2.0*x5 - 0.8*x7)
        if alpha < 0.0:
            return a*a - b*b
        else:
            return b*b - a*a
    else:
        return 0.0


# %% Another attempt to use Nijboer's Thesis Functions
# Tested, this approximation somehow doens't work (maybe, the amplitudes of polynomial aren't small enough)
def radial_integral_nijboer(zernike_pol: ZernPol, r: float, order: int, power: int) -> float:
    """
    Calculate the radial integrals used in the Nijboer's thesis.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or ZernPol
        Zernike polynomial definition.
    r : float
        Radius on the image coordinates.
    order : int
        Of the Bessel function Jv(r).
    power : int
        Power of the polynomial radial component.

    Reference
    ---------
    [1] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf

    Returns
    -------
    float
        Result of integration.

    """
    integral_sum = 0.0; (m, n) = define_orders(zernike_pol)  # get polynomial orders
    # Integration on the pupil radius. Vectorized form of simple integration equation
    h_p = 1.0/1000; p = np.arange(start=h_p, stop=1.0-h_p, step=h_p)
    fa = 0.0; fb = np.power(zernike_pol.radial(1.0), power)*jv(order, r)
    integral_sum = h_p*(np.sum(p*np.power(zernike_pol.radial(p), power)*jv(order, p*r)) + 0.5*(fa + fb))
    return integral_sum


def psf_point_approx_sum(zernike_pol: ZernPol, r: float, theta: float, alpha: float):
    """
    Get the point for calculation of PSF as the expansion of the diffraction integral acquired in the Nijboer's thesis.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or ZernPol
        Zernike polynomial definition.
    r : float
        Radius on the image coordinates.
    theta : float
        Angle on the image coordinates.
    alpha : float, optional
        Amplitude of the polynomial. The default is 1.0.

    Reference
    ---------
    [1] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf


    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    if not isinstance(zernike_pol, ZernPol):
        zernike_pol = ZernPol(m=m, n=n)
    # Integration on the pupil radius. Vectorized form of simple integration equation
    r = round(r, 11)
    if r == 0.0:
        s1 = 2.0*(jv(1, 1E-11)/1E-11); s2 = -2.0*alpha*pow(1j, m+n+1)*np.cos(m*theta)*(jv(n+1, 1E-11)/1E-11)
    else:
        s1 = 2.0*(jv(1, r)/r); s2 = -2.0*alpha*pow(1j, m+n+1)*np.cos(m*theta)*(jv(n+1, r)/r)
    s3_1 = radial_integral_nijboer(zernike_pol, r, order=0, power=2); s3_2 = radial_integral_nijboer(zernike_pol, r, order=2*m, power=2)
    s3 = -(pow(alpha, 2)/2.0)*(s3_1 + pow(1j, 2*m)*np.cos(2*m*theta)*s3_2)
    s4_1 = radial_integral_nijboer(zernike_pol, r, order=m, power=3); s4_2 = radial_integral_nijboer(zernike_pol, r, order=3*m, power=3)
    s4 = (pow(alpha, 3)/12.0)*(3.0*pow(1j, m+1)*np.cos(m*theta)*s4_1 + pow(1j, 3*m+1)*np.cos(3*m*theta)*s4_2)
    s5_1 = 3.0*radial_integral_nijboer(zernike_pol, r, order=0, power=4)
    s5_2 = 4.0*pow(1j, 2*m)*np.cos(2*m*theta)*radial_integral_nijboer(zernike_pol, r, order=2*m, power=4)
    s5_3 = pow(1j, 4*m)*np.cos(4*m*theta)*radial_integral_nijboer(zernike_pol, r, order=4*m, power=4)
    s5 = (pow(alpha, 4)/96.0)*(s5_1 + s5_2 + s5_3)
    return np.power(np.abs(s1 + s2 + s3 + s4 + s5), 2)


# %% Save and read the calculated PSF matricies
def save_psf(additional_file_name: str, psf_kernel: np.ndarray, NA: float, wavelength: float, calibration_coefficient: float,
             amplitude: float, polynomial_orders: tuple, folder_path: str = None, overwrite: bool = True) -> str:
    """
    Save the calculated PSF kernel along with the used for the calculation parameters.

    Parameters
    ----------
    additional_file_name : str
        Additional to the composed file name string, e.g. unique addition.
    psf_kernel : np.ndarray
        Calculated by using get_psf_kernel.
    NA : float
        NA used for the PSF calculation.
    wavelength : float
        Wavelength used for the PSF calculation (in micrometers).
    calibration_coefficient : float
        Calibration (micrometers/pixels) used for the PSF calculation.
    amplitude : float
        Amplitude of the polynomial.
    polynomial_orders : tuple
        Tuple as the (m, n) orders.
    folder_path : str, optional
        Absolute path to the folder where the file will be saved. The default is None.
    overwrite : bool, optional
        Flag for letting overwriting of the existing file. The default is True.

    Returns
    -------
    str
        Absolute path to the file.

    """
    # Checking the provided folder or creating the folder for storing files
    if folder_path is None or len(folder_path) == 0 or not Path(folder_path).is_dir():
        working_folder = Path(__file__).cwd()
        saved_psfs_folder = Path(working_folder).joinpath("saved_psfs")
        if not saved_psfs_folder.is_dir():
            saved_psfs_folder.mkdir()
        print("Auto assigned folder for saving calculated PSF kernel:", saved_psfs_folder)
    else:
        if Path(folder_path).is_dir():
            saved_psfs_folder = Path(folder_path)
    # Save provided PSF kernel with the provided parameters
    json_file_path = saved_psfs_folder.joinpath(f"psf_{polynomial_orders}_{additional_file_name}_{amplitude}.json")
    data4serialization = {}   # python dictionary is similar to the JSON file structure and can be dumped directly there
    data4serialization['PSF Kernel'] = psf_kernel.tolist(); data4serialization['NA'] = NA; data4serialization['Wavelength'] = wavelength
    data4serialization["Calibration (wavelength physical units/pixels)"] = calibration_coefficient
    if json_file_path.exists() and overwrite:
        _warn_message = "The file already exists, the content will be overwritten!"
        warnings.warn(_warn_message)
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


# %% Calculation, plotting
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
    img_type = img.dtype
    convolved_img = convolve(np.float32(img), psf_kernel, mode='reflect')
    conv_coeff = np.sum(psf_kernel)
    if conv_coeff > 0.0:
        convolved_img /= conv_coeff  # correct the convolution result by dividing to the kernel sum
    if scale2original:
        max_original_intensity = np.max(img); max_conv_pixel = np.max(convolved_img)
        scaling_factor = max_original_intensity / max_conv_pixel
        convolved_img *= scaling_factor
    convolved_img = convolved_img.astype(dtype=img_type)  # converting convolved image to the initial image
    return convolved_img


def get_psf_kernel(zernike_pol, calibration_coefficient: float, alpha: float, unified_kernel_size: bool = False) -> np.ndarray:
    """
    Calculate centralized matrix with PSF mask.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or isntance of the ZernPol class.
        Required for calculation Zernike polynomial.

    calibration_coefficient : float
        Relation between pixels and distance (physical).

    unified_kernel_size : bool
        Flag for adjusting or not the kernel size to the provided absolute value of alpha. The default is False.

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.
    """
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    # Define the kernel size just empirically, based on made experimental plots
    if not unified_kernel_size:
        if abs(alpha) < 1.0:
            multiplier = 20.0
        elif abs(alpha) < 1.5:
            multiplier = 24.0
        elif abs(alpha) <= 2.0:
            multiplier = 26.0
        else:
            multiplier = 30.0
    else:
        multiplier = 26.0
    max_size = int(round(multiplier*(1.0/calibration_coefficient), 0)) + 1  # Note that with the amplitude growth, it requires to grow as well
    size = max_size  # Just use the maximum estimation
    # Auto definition of the required PSF size is failed for the different PSFs forms (e.g., vertical coma with different amplitudes)
    # Make kernel with odd sizes for precisely centering the kernel
    if size % 2 == 0:
        size += 1
    kernel = np.zeros(shape=(size, size))
    i_center = size//2; j_center = size//2
    # Calculate the PSF kernel for usage in convolution operation
    for i in range(size):
        for j in range(size):
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))
            # The PSF also has the angular dependency, not only the radial one
            theta = np.arctan2((i - i_center), (j - j_center))
            theta += np.pi  # shift angles to the range [0, 2pi]
            # The scaling below is not needed because the Zernike polynomial is scaled as the RMS values
            kernel[i, j] = get_psf_point_r(zernike_pol, pixel_dist*calibration_coefficient, theta, alpha)
    return kernel


def show_ideal_psf(zernike_pol, size: int, calibration_coefficient: float, alpha: float, title: str = None, test_alt: bool = False):
    """
    Plot the intensity distribution on the image with WxH: (size, size) and using coefficient between pixel and physical distance.

    Note the color map is viridis.

    Parameters
    ----------
    size : int
        Size of picture for plotting.
    calibration_coefficient : float
        Relation between distance in pixels and um (see parameters at the start lines of the script).
    title : str, optional
        Title for the plotted figure. The default is None.

    Returns
    -------
    None.

    """
    if size % 2 == 0:
        size += 1  # make the image with odd sizes
    img = np.zeros((size, size), dtype=float)
    i_center = size//2; j_center = size//2
    for i in range(size):
        for j in range(size):
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))
            # The PSF also has the angular dependency, not only the radial one
            theta = np.arctan2((i - i_center), (j - j_center))
            theta += np.pi  # shift angles to the range [0, 2pi]
            if not test_alt:
                img[i, j] = get_psf_point_r(zernike_pol, pixel_dist*calibration_coefficient, theta, alpha)
            else:
                # Implementation for testing and comparing with the general equation
                theta -= np.pi/2.0
                img[i, j] = get_psf_point_bwap(zernike_pol, pixel_dist*calibration_coefficient, theta, alpha)
    if img[0, 0] > np.max(img)/100:
        __warn_message = f"The provided size for plotting PSF ({size}) isn't sufficient for proper representation"
        warnings.warn(__warn_message)
    if title is not None and len(title) > 0:
        plt.figure(title, figsize=(6, 6))
    else:
        plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()
    return img


def plot_correlation(zernike_pol, size: int, calibration_coefficient: float, alpha: float, title: str = None,
                     show_original: bool = True, show_psf: bool = True, R: int = 1):
    """
    Plot result of correlation with the object shown as the pixelated circle.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or ZernPol
        Zernike polynomial definition.
    size : int
        Size of the picture with the object.
    calibration_coefficient : float
        Coefficient between physical values and pixels.
    alpha : float
        Amplitude of the polynomial.
    title : str, optional
        Title of plots. The default is None.
    show_original : bool, optional
        Flag for plotting the original object. The default is True.
    show_psf : bool, optional
        Flag for plotting the used PSF. The default is True.
    R : int, optional
        Radius of the circle. The default is 1.

    Returns
    -------
    None.

    """
    if size % 2 == 0:
        size += 1  # make the image with odd sizes
    img = np.zeros((size, size), dtype=float)
    i_center = size//2; j_center = size//2; R = 1
    for i in range(size):
        for j in range(size):
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))/R
            if pixel_dist < 1.0:
                img[i, j] = 1.0
            # Blurring edges effects
            # elif pixel_dist == 1.0:
            #     img[i, j] = round(1.0/pow(pixel_dist+0.2, 2.6), 3)
            # elif pixel_dist < 1.5:
            #     img[i, j] = round(1.0/pow(pixel_dist, 2.6), 3)
            else:
                continue
    if show_original:
        plt.figure("Original object", figsize=(6, 6)); plt.imshow(img, cmap=plt.cm.viridis, extent=(0, size, 0, size))
        plt.tight_layout()
    psf_kernel = get_psf_kernel(zernike_pol, calibration_coefficient, alpha)
    if show_psf:
        plt.figure(f"PSF for {title}", figsize=(6, 6)); plt.imshow(psf_kernel, cmap=plt.cm.viridis)
    conv_img = convolute_img_psf(img, psf_kernel, scale2original=True)
    plt.figure(f"Convolved with {title} image"); plt.imshow(conv_img, cmap=plt.cm.viridis, extent=(0, size, 0, size)); plt.tight_layout()


def plot_correlation_photo(zernike_pol, calibration_coefficient: float, alpha: float, title: str = None,
                           show_original: bool = True, show_psf: bool = False):
    """
    Plot result of convolution of PSF with the sample image.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or ZernPol
        Zernike polynomial definition.
    size : int
        Size of the picture with the object.
    calibration_coefficient : float
        Coefficient between physical values and pixels.
    alpha : float
        Amplitude of the polynomial.
    title : str, optional
        Title of plots. The default is None.
    show_original : bool, optional
        Flag for plotting the original object. The default is True.
    show_psf : bool, optional
        Flag for plotting the used PSF. The default is True.

    Returns
    -------
    None.

    """
    try:
        sample = img_as_ubyte(io.imread(os.path.join(os.getcwd(), "nesvizh_grey.jpg"), as_gray=True))
        if show_original:
            plt.figure("Original Photo"); plt.imshow(sample, cmap=plt.cm.gray); plt.axis('off'); plt.tight_layout()
        psf_kernel = get_psf_kernel(zernike_pol, calibration_coefficient, alpha)
        if show_psf:
            plt.figure(f"PSF for {title}", figsize=(6, 6)); plt.imshow(psf_kernel, cmap=plt.cm.viridis)
        conv_img = convolute_img_psf(sample, psf_kernel, scale2original=True)
        plt.figure(f"Convolved with {title} image"); plt.imshow(conv_img, cmap=plt.cm.gray); plt.axis('off'); plt.tight_layout()
    except FileNotFoundError:
        _warn_message = "The sample photo is ignored in the repository, load it to the folder with this script from repo 'collection_numCalc'"
        warnings.warn(_warn_message)


# %% Object generation
def distance_f(i_px, j_px, i_centre, j_centre):
    """
    Calculate the distances for pixels.

    Parameters
    ----------
    i_px : int or numpy.ndarray
        Pixel(-s) i of an image.
    j_px : int or numpy.ndarray
        Pixel(-s) i of an image.
    i_centre : int
        Center of an image.
    j_centre : int
        Center of an image.

    Returns
    -------
    float or numpy.ndarray
        Distances between provided pixels and the center of an image.

    """
    return np.round(np.sqrt(np.power(i_px - i_centre, 2) + np.power(j_px - j_centre, 2)), 6)


def make_sample(radius: float, center_shift: tuple, max_intensity=255, test_plots: bool = False) -> np.ndarray:
    if radius < 1.0:
        radius = 1.0
    max_size = 4*int(round(radius, 0)) + 1
    i_shift, j_shift = center_shift
    net_shift = round(0.5*np.sqrt(i_shift*i_shift + j_shift*j_shift), 6)
    i_img_center = max_size // 2; j_img_center = max_size // 2
    if abs(i_shift) <= 1.0 and abs(j_shift) <= 1.0:
        i_center = i_img_center + i_shift; j_center = j_img_center + j_shift
    else:
        i_center = i_img_center; j_center = j_img_center
    print("Center of a bead:", i_center, j_center)
    # Define image type
    if isinstance(max_intensity, int):
        if max_intensity <= 255:
            img_type = 'uint8'
        else:
            img_type = 'uint16'
    elif isinstance(max_intensity, float):
        if max_intensity > 1.0:
            max_intensity = 1.0
            img_type = 'float'
    else:
        raise ValueError("Specify Max Intencity for image type according to uint8, uint16, float")
    img = np.zeros(dtype=img_type, shape=(max_size, max_size))
    # Below - difficult to calculate the precise intersection of the circle and pixels
    # points = []
    q_rad = round(0.25*radius, 6)
    size_subareas = 1001; normalization = 0.001*size_subareas*size_subareas
    for i in range(max_size):
        for j in range(max_size):
            distance = distance_f(i, j, i_center, j_center)
            pixel_value = 0.0  # meaning the intensity in the pixel

            # Discrete function
            # if distance < 0.5*radius:
            #     pixel_value = max_intensity
            # elif distance < radius:
            #     pixel_value = float(max_intensity)*np.exp(pow(0.5, 1.25) - np.power(distance/radius, 1.25))
            # else:
            #     pixel_value = float(max_intensity)*np.exp(pow(0.5, 2.5) - np.power(distance/radius, 2.5))

            # # Continiuous bump function - too scaled result
            # r_exceed = 0.499; power = 4
            # if distance < radius*(1.0 + r_exceed):
            #     x = distance/(radius + r_exceed)
            #     x_pow = pow(x, power); b_pow = pow(1.0 + r_exceed, power)
            #     pixel_value = e*np.exp(b_pow/(x_pow - b_pow))

            # Discontinuous
            # x = distance / radius; ots = np.exp(-1.0/np.power(6.0, 2))
            # if distance < radius:
            #     pixel_value = np.exp(-np.power(x, 2)/np.power(6.0, 2))
            # else:
            #     x_shift = pow(x, 4); x_c = pow(0.95, 4)
            #     pixel_value = ots*np.exp(x_c - x_shift)

            # The center of bead lays always within single pixel
            # oversize = round(radius + 1 + net_shift, 6); bump_f_power = 16
            if distance < q_rad:
                pixel_value = 1.0  # entire pixel lays inside the circle
            # elif distance <= oversize - 0.5:
            #     x = pow(distance, bump_f_power); b = pow(oversize, bump_f_power)
            #     pixel_value = e*np.exp(b/(x - b))

            # !!! The scheme below - overcomplicated and requires better definition of intersections of pixels and circle curvature
            # Rough estimate of the potentially outside pixels - they should be checked for intersection with the circle

            elif q_rad <= distance <= radius + net_shift + 1.0:
                stop_checking = False  # flag for quitting this calculations
                # First, sort out the pixels that lay completely within the circle, but the distance is more than quarter of R:
                if i < i_center:
                    i_corner = i - 0.5
                else:
                    i_corner = i + 0.5
                if j < j_center:
                    j_corner = j - 0.5
                else:
                    j_corner = j + 0.5
                # Below - distance to the most distant point of the pixel
                distance_corner = distance_f(i_corner, j_corner, i_center, j_center)
                if distance_corner <= radius:
                    pixel_value = 1.0; stop_checking = True

                # So, the pixel's borders can potentially are intersected by the circle, calculate the estimated intersection area
                if not stop_checking:
                    i_m = i - 0.5; j_m = j - 0.5; i_p = i + 0.5; j_p = j + 0.5
                    # circle_arc_area = np.zeros(shape=(size_subareas, size_subareas))
                    # h_x = (i_p - i_m)/size_subareas; h_y = (j_p - j_m)/size_subareas
                    # x_row = np.round(np.arange(start=i_m, stop=i_p+h_x/2, step=h_x), 6)
                    # y_col = np.round(np.arange(start=j_m, stop=j_p+h_y/2, step=h_y), 6)
                    x_row = np.linspace(start=i_m, stop=i_p, num=size_subareas); y_col = np.linspace(start=j_m, stop=j_p, num=size_subareas)
                    # print(np.min(x_row), np.max(x_row), np.min(y_col), np.max(y_col))
                    coords = np.meshgrid(x_row, y_col); distances = distance_f(coords[0], coords[1], i_center, j_center)
                    circle_arc_area1 = np.where(distances <= radius, 0.001, 0.0)
                    # print(circle_arc_area1.shape)
                    if np.max(circle_arc_area1) > 0.0 and radius <= 2.0 and test_plots:
                        plt.figure(f"{i, j}"); plt.imshow(circle_arc_area1)
                    # print(np.max(circle_arc_area1), np.min(circle_arc_area1))
                    # for y in range(size_subareas):
                    #     for x in range(size_subareas):
                    #         i_c = i_m + y*((i_p - i_m)/size_subareas); j_c = j_m + x*((j_p - j_m)/size_subareas)
                    #         distance_px = distance_f(i_c, j_c, i_center, j_center)
                    #         if distance_px <= radius:
                    #             circle_arc_area[y, x] = 1.0
                    # S = round(np.sum(circle_arc_area)/np.sum(pixel_area), 6)
                    S1 = round(np.sum(circle_arc_area1)/normalization, 6)
                    if S1 > 1.0:
                        print(np.min(x_row), np.max(x_row), np.min(y_col), np.max(y_col))
                        print(circle_arc_area1.shape)
                        print("Overflowed value", S1, "sum of pixels inside of the intersection:", np.sum(circle_arc_area1), "norm.:", normalization)
                        if test_plots:
                            plt.figure(f"[{i, j}]"); plt.imshow(circle_arc_area1)
                        S1 = 1.0
                    print(f"Found ratio for the pixel [{i, j}]:", S1); pixel_value = S1
                    # print(f"Found ratio for the pixel [{i, j}]:", S, "diff for - vect. implementations:", round(abs(S-S1), 6))
                    # r_diff1 = round(r2 - np.power(i_m - i_center, 2), 6); r_diff2 = round(r2 - np.power(j_m - j_center, 2), 6)
                    # r_diff3 = round(r2 - np.power(i_p - i_center, 2), 6); r_diff4 = round(r2 - np.power(j_p - j_center, 2), 6)
                    # found_points = 0; this_pixel_points = []
                    # # calculation of the j index
                    # if r_diff1 > 0.0:
                    #     j1 = round(j_center - np.sqrt(r_diff1), 6); j2 = round(j_center + np.sqrt(r_diff1), 6)
                    #     if j1 > 0.0 and j1 <= j_p and j1 >= j_m:
                    #         point = (i_m, j1); points.append(point); this_pixel_points.append(point); found_points += 1
                    #     if j2 > 0.0 and j2 <= j_p and j2 >= j_m:
                    #         point = (i_m, j2); points.append(point); this_pixel_points.append(point); found_points += 1
                    # if r_diff3 > 0.0:
                    #     j1 = round(j_center - np.sqrt(r_diff3), 6); j2 = round(j_center + np.sqrt(r_diff3), 6)
                    #     if j1 > 0.0 and j1 <= j_p and j1 >= j_m:
                    #         point = (i_p, j1); points.append(point); this_pixel_points.append(point); found_points += 1
                    #     if j2 > 0.0 and j2 <= j_p and j2 >= j_m:
                    #         point = (i_p, j2); points.append(point); this_pixel_points.append(point); found_points += 1
                    # # calculation of the i index
                    # if r_diff2 > 0.0:
                    #     i1 = round(i_center - np.sqrt(r_diff2), 6); i2 = round(i_center + np.sqrt(r_diff2), 6)
                    #     if i1 > 0.0 and i1 <= i_p and i1 >= i_m:
                    #         point = (i1, j_m); points.append(point); this_pixel_points.append(point); found_points += 1
                    #     if i2 > 0.0 and i2 <= i_p and i2 >= i_m:
                    #         point = (i2, j_m); points.append(point); this_pixel_points.append(point); found_points += 1
                    # if r_diff4 > 0.0:
                    #     i1 = round(i_center - np.sqrt(r_diff4), 6); i2 = round(i_center + np.sqrt(r_diff4), 6)
                    #     if i1 > 0.0 and i1 <= i_p and i1 >= i_m:
                    #         point = (i1, j_p); points.append(point); this_pixel_points.append(point); found_points += 1
                    #     if i2 > 0.0 and i2 <= i_p and i2 >= i_m:
                    #         point = (i2, j_p); points.append(point); this_pixel_points.append(point); found_points += 1

                    # # Calculated intersected square
                    # if found_points == 2:
                    #     print(f"Found intersections for the pixel [{i, j}]:", this_pixel_points)
                    #     x1, y1 = this_pixel_points[0]; x2, y2 = this_pixel_points[1]; S = 0.0
                    #     # Define intersection type - too complex (triangle, trapezoid, etc.)
                    #     # A = (i_m, j_m); B = (i_m, j_p); C = (i_p, j_m); D = (i_p, j_p)
                    #     x_m = 0.5*(x1 + x2); y_m = 0.5*(y1 + y2)
                    # print("middle point:", x_m, y_m)
                    # distance_m = round(np.sqrt(np.power(x_m - i_center, 2) + np.power(y_m - j_center, 2)), 6)
                    # if distance_m > distance:
                    #     S = 1.0 - 0.5*(distance_m - distance)
                    # else:
                    #     S = 1.0 + 0.5*(distance_m - distance)
                    # pixel_value = S
                    # print(f"Found points for the single pixel [{i, j}]:", found_points)

            # Pixel value scaling according the the provided image type
            pixel_value *= float(max_intensity)
            # Pixel value conversion to the image type
            if pixel_value > 0.0:
                if 'uint' in img_type:
                    pixel_value = int(round(pixel_value, 0))
                img[i, j] = pixel_value
    # points = set(points)  # excluding repeated found in the loop coordinates
    # print("found # of points:", len(points), "\ncoordinates:", points)
    return img


# %% Radial profile testing for the object generation
def profile1(x, sigma: float = 1.0):
    return np.exp(-np.power(x, 2)/np.power(sigma, 2))


def profile2(x):
    y = np.zeros(shape=x.shape); ots = np.exp(-np.power(1.0, 2)/np.power(3*1.0, 2))
    for i, el in enumerate(x):
        # if el < 0.5:
        #     y[i] = 1.0
        # elif el < 1.0:
        #     y[i] = np.exp(pow(0.5, 1.25) - np.power(el, 1.25))
        # else:
        #     y[i] = np.exp(pow(0.5, 2.5) - np.power(el, 2.5))
        if el <= 1.0:
            y[i] = np.exp(-np.power(el, 2)/np.power(3*1.0, 2))
        else:
            xc = 1.0; el = pow(el, 8)
            y[i] = ots*np.exp(xc-el)
    return y


def profile3(x, gamma: float = 1.0):
    gamma2 = gamma*gamma
    return gamma2/(np.power(x, 2) + gamma2)


def profile4(x):
    return np.exp(x)/np.power(1.0 + np.exp(x), 2)


def profile5(x, b: float = 1.0):
    y = np.zeros(shape=x.shape)
    for i, el in enumerate(x):
        if el < b:
            el2 = el*el; b2 = b*b
            y[i] = np.exp(b2/(el2 - b2))
    return y*e


def profile6(x, b: float = 1.0):
    y = np.zeros(shape=x.shape)
    for i, el in enumerate(x):
        if el < b:
            el3 = el*el*el; b3 = b*b*b
            y[i] = np.exp(b3/(el3 - b3))
    return y*e


# %% Testing the bump function difference in the radial profiles
def bump_f(x: np.ndarray, b: float = 1.0, power: int = 2) -> np.ndarray:
    y = np.zeros(shape=x.shape)
    for i, el in enumerate(x):
        if el < b:
            el_pow = pow(el, power); b_pow = pow(b, power)
            y[i] = e*np.exp(b_pow/(el_pow - b_pow))
    return y


# %% Tests
if __name__ == '__main__':
    orders1 = (0, 2); orders2 = (0, 0); orders3 = (-1, 1); orders4 = (-3, 3); plot_pure_psfs = True; plot_photo_convolution = False
    plot_photo_convolution_row = False; figsizes = (6.5, 6.5); test_write_read_psf = False; test_disk_show = False

    # Plotting
    plt.ion(); plt.close('all'); conv_pic_size = 14; detailed_plots_sizes = 22; calibration_coeff = pixel2um_coeff; alpha = 2.0
    if plot_pure_psfs:
        t1 = time.time()
        p_img = show_ideal_psf(orders2, detailed_plots_sizes-10, calibration_coeff, alpha, "Piston")
        print(f"Calculation (steps on phi/p {n_phi_points}/{n_p_points}) of Piston takes s:", round((time.time() - t1), 3)); t1 = time.time()
        # p_img2 = show_ideal_psf(orders2, detailed_plots_sizes, calibration_coeff, alpha, "Piston2", test_alt=True)
        # print("Alternative calculation of Piston takes s:", round((time.time() - t1), 3)); t1 = time.time()
        # ytilt_img = show_ideal_psf(orders3, detailed_plots_sizes, calibration_coeff, alpha, "Y Tilt")
        # print(f"Calculation (steps on phi/p {n_phi_points}/{n_p_points}) of Y Tilt takes s:", round((time.time() - t1), 3)); t1 = time.time()
        # ytilt_img2 = show_ideal_psf(orders3, detailed_plots_sizes, calibration_coeff, alpha, "Y Tilt2", test_alt=True)
        # print("Alternative calculation of Y tilt takes s:", round((time.time() - t1), 3)); t1 = time.time()
        # # defocus_img = show_ideal_psf(orders1, detailed_plots_sizes, pixel2um_coeff/3.0, alpha, "Defocus")
        # t1 = time.time()
        # aberr4 = show_ideal_psf(orders4, detailed_plots_sizes+2, calibration_coeff, alpha, "Vertical Trefoil")
        # print(f"Calculation (steps on phi/p {n_phi_points}/{n_p_points}) of Vertical Trefoil takes s:", round((time.time() - t1), 3))
        # aberr4_2 = show_ideal_psf(orders4, detailed_plots_sizes, calibration_coeff, alpha, "Vertical Trefoil2", test_alt=True)
        # print("Alternative calculation of Vertical Trefoil takes s:", round((time.time() - t1), 3)); t1 = time.time()

        # Testing the kernel calculation
        v_coma_m = get_psf_kernel((-1, 3), calibration_coeff, -1.0)
        plt.figure(figsize=figsizes); plt.imshow(v_coma_m, cmap=plt.cm.viridis); plt.tight_layout()
        # v_coma_p = get_psf_kernel((-1, 3), calibration_coeff, 1.0)
        # plt.figure(figsize=figsizes); plt.imshow(v_coma_p, cmap=plt.cm.viridis); plt.tight_layout()

    # Compare positive and negative coefficients influence on the PSF
    # ytilt_img1 = show_ideal_psf(orders3, detailed_plots_sizes, calibration_coeff, alpha, "+ Y Tilt")
    # ytilt_img2 = show_ideal_psf(orders3, detailed_plots_sizes, calibration_coeff, -alpha, "- Y Tilt")
    # p_img1 = show_ideal_psf(orders1, detailed_plots_sizes, calibration_coeff, alpha, "Defocus +")
    # p_img2 = show_ideal_psf(orders1, detailed_plots_sizes, calibration_coeff, -alpha, "Defocus -")
    # show_ideal_psf(orders4, detailed_plots_sizes, calibration_coeff, alpha, "+ Vertical Trefoil")
    # show_ideal_psf(orders4, detailed_plots_sizes, calibration_coeff, -alpha, "- Vertical Trefoil")
    # show_ideal_psf((-2, 2), detailed_plots_sizes, calibration_coeff, alpha, "+ Obliq. Astigmatism")
    # show_ideal_psf((-2, 2), detailed_plots_sizes, calibration_coeff, -alpha, "- Obliq. Astigmatism")
    # show_ideal_psf((0, 4), detailed_plots_sizes, calibration_coeff, alpha, f"{alpha} Primary Spherical")
    # show_ideal_psf((0, 4), detailed_plots_sizes, calibration_coeff, -alpha, f"{-alpha} Primary Spherical")

    # Plot convolution of the sample photo with the various psfs
    if plot_photo_convolution:
        plot_correlation_photo(orders2, pixel2um_coeff/1.5, 1.0, "Piston", show_psf=True, show_original=True)
        plot_correlation_photo(orders3, pixel2um_coeff/1.5, 1.0, "Y Tilt", show_psf=True, show_original=True)
        plot_correlation_photo(orders1, pixel2um_coeff/1.5, 1.0, "Defocus", show_psf=True, show_original=True)
        plot_correlation_photo(orders4, pixel2um_coeff/1.5, 1.0, "Vertical Trefoil", show_psf=True, show_original=True)

    # Testing of the convolution results with different amplitudes and signs
    if plot_photo_convolution_row:
        amplitudes = [-2.0, 0.0, 2.0]; vert_coma = (-1, 3)
        for amplitude in amplitudes:
            plot_correlation_photo(vert_coma, pixel2um_coeff/1.5, amplitude, f"Y Coma {amplitude}", show_psf=False, show_original=False)

    # Plot convolution of point objet with the various psfs
    # plot_correlation(orders2, conv_pic_size, pixel2um_coeff, 0.85, "Piston", show_psf=True)
    # plot_correlation(orders3, conv_pic_size, pixel2um_coeff/1.75, 0.5, "Y Tilt", True, show_psf=True)
    # plot_correlation(orders1, conv_pic_size, pixel2um_coeff, 0.85, "Defocus", False, show_psf=True)

    # Testing saving / reading
    if test_write_read_psf:
        ampl = -1.0; psf_kernel = get_psf_kernel(orders4, calibration_coeff, ampl)
        file_path = save_psf("test", psf_kernel, NA, wavelength, calibration_coeff, ampl, orders4)
        psf_stored_data = read_psf(file_path)
        read_psf_kernel = np.asarray(psf_stored_data['PSF kernel'])
        plt.figure(figsize=figsizes); plt.imshow(read_psf_kernel, cmap=plt.cm.viridis); plt.tight_layout()

    # Testing disk representation
    if test_disk_show:
        i_shift = 0.23; j_shift = -0.591; disk_r = 6.0
        disk1 = make_sample(radius=disk_r, center_shift=(i_shift, j_shift), test_plots=False)
        plt.figure(figsize=figsizes); axes_img = plt.imshow(disk1, cmap=plt.cm.viridis); plt.tight_layout()
        m_center, n_center = disk1.shape; m_center = m_center // 2 + i_shift; n_center = n_center // 2 + j_shift
        axes_img.axes.add_patch(Circle((n_center, m_center), disk_r, edgecolor='red', facecolor='none'))
        r = np.arange(start=0.0, stop=1.3, step=0.02)
        profile1_f = profile1(r, 1.0)  # normal gaussian
        profile2_f = profile2(r)  # discontinuous function
        profile3_f = profile3(r, 1.0)  # lorentzian
        profile4_f = profile4(r); max_4 = np.max(profile4_f); profile4_f /= max_4  # derivative of logistic function
        profile5_f = profile5(r, 1.4)  # bump function
        profile6_f = profile6(r, 1.4)  # modified bump function
        # plt.figure("Profiles Comparison"); plt.plot(r, profile1_f, r, profile2_f, r, profile3_f, r, profile4_f, r, profile5_f, r, profile6_f)
        # plt.legend(['gaussian', 'discontinuous', 'lorentzian', 'd(logist.f)/dr', 'bump', 'mod. bump'])
        # Comparison of bump functions depending on the power of arguments
        size = 1.5; step_r = 0.01; r1 = np.arange(start=0.0, stop=size+step_r, step=step_r)
        bump2 = bump_f(r1, size, 2); bump4 = bump_f(r1, size, 4); bump3 = bump_f(r1, size, 3); bump64 = bump_f(r1, size, 64)
        bump8 = bump_f(r1, size, 8); bump16 = bump_f(r1, size, 16); bump32 = bump_f(r1, size, 32)
        # plt.figure("Bump() Comparison"); plt.plot(r1, bump2, r1, bump3, r1, bump4, r1, bump8, r1, bump16)
        # plt.legend(['^2', '^3', '^4', '^8', '^16']); plt.axvline(x=0.5); plt.axvline(x=1.0)
