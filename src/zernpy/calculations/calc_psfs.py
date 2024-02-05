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

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calc_zernike_pol import define_orders
    from calc_psfs_check import save_psf, read_psf
else:
    from .calc_zernike_pol import define_orders
    from .calc_psfs_check import save_psf, read_psf


# %% Module parameters
__docformat__ = "numpydoc"

# %% Reference
def airy_ref_pattern(r: float):
    r = round(r, 12)
    if r == 0.0:
        ratio = jv(1, 1E-11)/1E-11
    else:
        ratio = jv(1, r)/r
    return 4.0*pow(ratio, 2)


# %% PSF pixel value calc.
def diffraction_integral_r(zernike_pol: ZernPol, alpha: float, phi: float, p,
                           theta: float, r: float, k: float, NA: float) -> np.array:
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
    k : float
        Spatial frequency (k = 2pi/lambda).
    NA : float
        Property of the microobjective.

    References
    ----------
    [1] Principles of Optics, by M. Born and E. Wolf, 4 ed., 1968
    [2] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf

    Returns
    -------
    numpy.ndarray
        Values of the diffraction integral.

    """
    phase_arg = (alpha*zernike_pol.polynomial_value(p, phi) - k*NA*r*p*np.cos(phi - theta))*1j  # ??? multiple zernike_pol on k or not?
    return np.exp(phase_arg)*p


def radial_integral(zernike_pol: ZernPol, r: float, theta: float, phi: float, alpha: float,
                    k: float, NA: float, n_int_r_points: int) -> complex:
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
    k : float
        Spatial frequency (k = 2pi/lambda).
    NA : float
        Property of the microobjective.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).

    Returns
    -------
    complex
        Complex amplitude of the field as result of integration on the pupil radius coordinate.

    """
    # Integration on the pupil angle. Vectorized form of the trapezoidal rule
    h_p = 1.0/n_int_r_points; p = np.arange(start=h_p, stop=1.0, step=h_p)
    fa = diffraction_integral_r(zernike_pol, alpha, phi, 0.0, theta, r, k, NA)
    fb = diffraction_integral_r(zernike_pol, alpha, phi, 1.0, theta, r, k, NA)
    ang_int = np.sum(diffraction_integral_r(zernike_pol, alpha, phi, p, theta, r, k, NA)) + 0.5*(fa + fb)
    return h_p*ang_int


def get_psf_point_r(zernike_pol: ZernPol, r: float, theta: float, k: float, NA: float, alpha: float = 1.0,
                    n_int_r_points: int = 320, n_int_phi_points: int = 300) -> float:
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
    k : float
        Spatial frequency (k = 2pi/lambda).
    NA : float
        Property of the microobjective.
    alpha : float, optional
        Amplitude of the polynomial. The default is 1.0.
    n_int_r_points : int
        Number of integration points used for integration on the radius of the entrance pupil (normalized to the range [0.0, 1.0]).
        The default is 320.
    n_int_phi_points : int
        Number of integration points used for integration on the polar angle of the entrance pupil (from the range [0.0, 2pi]).
        The default is 300.

    Returns
    -------
    float
        |U|**2 - the module and square of the amplitude, intensity as the PSF value.

    """
    # Integration on the pupil radius using Simpson equation
    h_phi = 2.0*pi/n_int_phi_points; even_sum = 0.0j; odd_sum = 0.0j
    for i in range(2, n_int_phi_points-2, 2):
        phi = i*h_phi; even_sum += radial_integral(zernike_pol, r, theta, phi, alpha, k, NA, n_int_r_points)
    for i in range(1, n_int_phi_points-1, 2):
        phi = i*h_phi; odd_sum += radial_integral(zernike_pol, r, theta, phi, alpha, k, NA, n_int_r_points)
    yA = radial_integral(zernike_pol, r, theta, 0.0, alpha, k, NA, n_int_r_points)
    yB = radial_integral(zernike_pol, r, theta, 2.0*pi, alpha, k, NA, n_int_r_points)
    integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
    return np.power(np.abs(integral_sum), 2)*integral_normalization


# %% PSF kernel calc.
def get_psf_kernel(zernike_pol, calibration_coefficient: float, alpha: float, k: float, NA: float, unified_kernel_size: bool = False,
                   n_int_r_points: int = 320, n_int_phi_points: int = 300, show_kernel: bool = True, fig_title: str = None,
                   normalize_values: bool = False, airy_pattern: bool = False) -> np.ndarray:
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
    # Check and initialize Zernike polynomial if provided only orders
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    if not isinstance(zernike_pol, ZernPol):
        zernike_pol = ZernPol(m=m, n=n)
    # Estimation of the kernel size
    if m == 0:
        if n == 0:
            multiplier = 1.0
        else:
            multiplier = 2.0*sqrt(n)
    else:
        multiplier = 2.0*sqrt(n)
        if alpha > 1.0:
            multiplier *= alpha
    size = int(round(multiplier/calibration_coefficient, 0)) + 1  # Note that with the amplitude growth, it requires to grow as well
    print('defined size of kernel:', size)
    # Auto definition of the required PSF size is complicated for the different PSFs forms (e.g., vertical coma with different amplitudes)
    # Make kernel with odd sizes for precisely centering the kernel (in the center of an image)
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
            if not airy_pattern:
                kernel[i, j] = get_psf_point_r(zernike_pol, pixel_dist*calibration_coefficient, theta, k, NA, alpha, n_int_r_points, n_int_phi_points)
            else:
                kernel[i, j] = airy_ref_pattern(pixel_dist*calibration_coefficient*k*NA)
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
            plt.figure(f"{(m, n)}", figsize=(6, 6))
        plt.imshow(kernel, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()
    return kernel


# %% Define standard exports from this module
__all__ = ['get_psf_kernel', 'save_psf', 'read_psf']

# %% Tests
if __name__ == '__main__':
    plt.ion(); plt.close('all')  # close all plots before plotting new ones
    # Physical parameters
    wavelength = 0.55  # in micrometers
    k = 2.0*np.pi/wavelength  # angular frequency
    NA = 0.95  # microobjective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length for an image)
    # Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
    resolution = 0.61*wavelength/NA
    pixel_size_nyquist = 0.5*resolution
    pixel_size = 0.5*pixel_size_nyquist  # the relation between um / pixels for calculating the coordinate in physical units for each pixel
    # Checks for some polynomials
    pol1 = (0, 0); pol2 = (-1, 1); pol3 = (0, 2); pol4 = (-2, 2)
    kern1 = get_psf_kernel(pol1, calibration_coefficient=pixel_size, alpha=0.5, k=k, NA=NA, normalize_values=False)
    # kern_ref = get_psf_kernel(pol1, calibration_coefficient=pixel_size, alpha=0.5, k=k, NA=NA, normalize_values=False, airy_pattern=True)
    kern2 = get_psf_kernel(pol2, calibration_coefficient=pixel_size, alpha=0.5, k=k, NA=NA, normalize_values=False)
    kern3 = get_psf_kernel(pol3, calibration_coefficient=pixel_size, alpha=0.5, k=k, NA=NA, normalize_values=False)
    kern4 = get_psf_kernel(pol4, calibration_coefficient=pixel_size, alpha=0.5, k=k, NA=NA, normalize_values=False)
