# -*- coding: utf-8 -*-
"""
Calculation and plotting of associated with polynomials PSFs.

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
NA = 0.95  # microobjective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length)
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
    h_p = 1.0/n_p_points; p = np.arange(start=h_p, stop=1.0 - h_p, step=h_p)
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
    h_phi = pi/n_phi_points; phi = np.arange(start=h_phi, stop=2.0*pi - h_phi, step=h_phi)
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
        print("Folder for saving calculated PSFs:", saved_psfs_folder)
    else:
        if Path(folder_path).is_dir():
            saved_psfs_folder = Path(folder_path)
    # Save provided PSF kernel with the provided parameters
    json_file_path = saved_psfs_folder.joinpath(f"psf_{polynomial_orders}_{additional_file_name}_{amplitude}.json")
    data4serialization = {}   # python dictionary is similar to the JSON file structure and can be dumped directly there
    data4serialization['PSF kernel'] = psf_kernel.tolist(); data4serialization['NA'] = NA
    data4serialization['Wavelength'] = wavelength; data4serialization['Calibration (um/pixels)'] = calibration_coefficient
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


def show_ideal_psf(zernike_pol, size: int, calibration_coefficient: float, alpha: float, title: str = None):
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
            img[i, j] = get_psf_point_r(zernike_pol, pixel_dist*calibration_coefficient, theta, alpha)
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
def make_sample(radius: float, center_shift: tuple, max_intensity=255) -> np.ndarray:
    if radius < 1.0:
        radius = 1.0
    max_size = 4*int(round(radius, 0)) + 1
    i_shift, j_shift = center_shift
    i_img_center = max_size // 2; j_img_center = max_size // 2
    if i_shift <= 1.0 and j_shift <= 1.0:
        i_center = i_img_center + i_shift; j_center = j_img_center + j_shift
    else:
        i_center = i_img_center; j_center = j_img_center
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
    for i in range(max_size):
        for j in range(max_size):
            distance = np.sqrt(np.power(i - i_center, 2) + np.power(j - j_center, 2))
            if distance < 0.5*radius:
                pixel_value = max_intensity
            elif distance < radius:
                pixel_value = float(max_intensity)*np.exp(pow(0.5, 1.25) - np.power(distance/radius, 1.25))
            else:
                pixel_value = float(max_intensity)*np.exp(pow(0.5, 2.5) - np.power(distance/radius, 2.5))
            if 'uint' in img_type:
                pixel_value = int(round(pixel_value, 0))
            img[i, j] = pixel_value
    return img


# %% Tests
if __name__ == '__main__':
    orders1 = (0, 2); orders2 = (0, 0); orders3 = (-1, 1); orders4 = (-3, 3); plot_pure_psfs = False; plot_photo_convolution = False
    plot_photo_convolution_row = False; figsizes = (6.5, 6.5); test_write_read_psf = False; test_disk_show = True

    # Plotting
    plt.ion(); plt.close('all'); conv_pic_size = 14; detailed_plots_sizes = 24; calibration_coeff = pixel2um_coeff/1.75; alpha = 2.0
    if plot_pure_psfs:
        t1 = time.time()
        p_img = show_ideal_psf(orders2, detailed_plots_sizes, calibration_coeff, alpha, "Piston")
        print(f"Calculation (steps on phi/p {n_phi_points}/{n_p_points}) of Piston takes s:", round((time.time() - t1), 3)); t1 = time.time()
        ytilt_img = show_ideal_psf(orders3, detailed_plots_sizes, calibration_coeff, alpha, "Y Tilt")
        print(f"Calculation (steps on phi/p {n_phi_points}/{n_p_points}) of Y Tilt takes s:", round((time.time() - t1), 3)); t1 = time.time()
        # defocus_img = show_ideal_psf(orders1, detailed_plots_sizes, pixel2um_coeff/3.0, alpha, "Defocus")
        t1 = time.time()
        aberr4 = show_ideal_psf(orders4, detailed_plots_sizes, calibration_coeff, alpha, "Vertical Trefoil")
        print(f"Calculation (steps on phi/p {n_phi_points}/{n_p_points}) of Vertical Trefoil takes s:", round((time.time() - t1), 3))

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

    # Testing the kernel calculation
    # v_coma_m = get_psf_kernel((-1, 3), calibration_coeff, -1.0)
    # plt.figure(figsize=figsizes); plt.imshow(v_coma_m, cmap=plt.cm.viridis); plt.tight_layout()
    # v_coma_p = get_psf_kernel((-1, 3), calibration_coeff, 1.0)
    # plt.figure(figsize=figsizes); plt.imshow(v_coma_p, cmap=plt.cm.viridis); plt.tight_layout()

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
        i_shift = 0.2; j_shift = -0.0; disk_r = 2.0
        disk1 = make_sample(radius=disk_r, center_shift=(i_shift, j_shift))
        plt.figure(figsize=figsizes); axes_img = plt.imshow(disk1, cmap=plt.cm.viridis); plt.tight_layout()
        m_center, n_center = disk1.shape; m_center = m_center // 2 + i_shift; n_center = n_center // 2 + j_shift
        axes_img.axes.add_patch(Circle((n_center, m_center), disk_r, edgecolor='red', facecolor='none'))
