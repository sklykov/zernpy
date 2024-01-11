# -*- coding: utf-8 -*-
"""
Calculation and plotting of associated with polynomials PSFs.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import jv
import warnings
from scipy.ndimage import convolve
from math import cos

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calc_zernike_pol import define_orders
else:
    from .calc_zernike_pol import define_orders

# %% Module parameters
__docformat__ = "numpydoc"


# %% Functions
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


def get_aberrated_psf(zernike_pol, r: float, theta: float, alpha: float = 1.0) -> float:
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
        return 0.0  # !!! Should be exchanged to the integral or precalculated equations


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


def get_psf_kernel(zernike_pol, calibration_coefficient: float, alpha: float) -> np.ndarray:
    """
    Calculate centralized matrix with PSF mask.

    Parameters
    ----------
    zernike_pol : tuple (m, n) or isntance of the ZernPol class.
        Required for calculation Zernike polynomial.

    calibration_coefficient : float
        Relation between pixels and distance (physical).

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.
    """
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    # Define the kernel size, including even small intensity pixels
    max_size = int(round(10.0*(1.0/calibration_coefficient), 0)) + 1
    for i in range(max_size):
        if abs(get_aberrated_psf(zernike_pol, i*calibration_coefficient, np.pi/2.0, alpha)) < 0.0001:
            break
    # Make kernel with odd sizes for precisely centering the kernel
    size = 2*i - 1
    if i % 2 == 0:
        size = i + 1
    kernel = np.zeros(shape=(size, size))
    i_center = size//2; j_center = size//2
    # Calculate the PSF kernel for usage in convolution operation
    for i in range(size):
        for j in range(size):
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))
            # The PSF also has the angular dependency, not only the radial one
            theta = np.arctan2((j - j_center), (i - i_center))
            # theta += np.pi  # shift angles to the range [0, 2pi]
            kernel[i, j] = get_aberrated_psf(zernike_pol, pixel_dist*calibration_coefficient, theta, alpha)
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
            theta = np.arctan2((j - j_center), (i - i_center))
            # theta += np.pi  # shift angles to the range [0, 2pi]
            img[i, j] = get_aberrated_psf(zernike_pol, pixel_dist*calibration_coefficient, theta, alpha)
    if img[0, 0] > np.max(img)/100:
        __warn_message = f"The provided size for plotting PSF ({size}) isn't sufficient for proper representation"
        warnings.warn(__warn_message)
    if title is not None and len(title) > 0:
        plt.figure(title, figsize=(6, 6))
    else:
        plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=plt.cm.viridis); plt.tight_layout()
    return img


def plot_correlation(zernike_pol, size: int, calibration_coefficient: float, alpha: float, title: str = None,
                     show_original: bool = True, show_psf: bool = True):
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


# %% Tests
if __name__ == '__main__':
    r = 0.0
    orders1 = (0, 2); orders2 = (0, 0); orders3 = (-1, 1)
    # Physical parameters
    wavelength = 0.55  # in micrometers
    k = 2.0*np.pi/wavelength  # angular frequency
    NA = 0.95  # microobjective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length)
    # Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
    pixel_size = 0.125  # in micrometers, physical length in pixels (um / pixels)
    pixel2um_coeff = k*NA*pixel_size  # coefficient used for relate pixels to physical units
    pixel2um_coeff_plot = k*NA*(pixel_size/10.0)  # coefficient used for better plotting with the reduced pixel size for preventing pixelated
    # Plotting
    plt.close('all'); conv_pic_size = 10; detailed_plots_sizes = 30
    # p_img = show_ideal_psf(orders2, 20, pixel2um_coeff/2, "Piston"); ytilt_img = show_ideal_psf(orders3, 20, pixel2um_coeff/2, "Y Tilt")
    p_img = show_ideal_psf(orders2, detailed_plots_sizes, pixel2um_coeff/4.5, 0.95, "Piston")
    ytilt_img = show_ideal_psf(orders3, detailed_plots_sizes, pixel2um_coeff/4.5, 0.95, "Y Tilt")
    defocus_img = show_ideal_psf(orders1, detailed_plots_sizes+10, pixel2um_coeff/4.5, 0.95, "Defocus")
    plot_correlation(orders2, conv_pic_size, pixel2um_coeff/2, 1.5, "Piston", show_psf=False)
    plot_correlation(orders3, conv_pic_size, pixel2um_coeff/2, 1.5, "Y Tilt", False, show_psf=False)
    plot_correlation(orders1, conv_pic_size, pixel2um_coeff/2, 0.95, "Defocus", False, show_psf=True)
    plt.show()
