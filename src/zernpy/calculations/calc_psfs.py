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

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calc_zernike_pol import define_orders
else:
    from .calc_zernike_pol import define_orders

# %% Module parameters
__docformat__ = "numpydoc"


# %% Functions
def radial_func(n: int, r: float) -> float:
    # Defining pure radial function (angular indenpendent) - Jv(r)/r
    if isinstance(r, int):  # Convert int to float explicitly
        r = float(r)
    radial = 0.0  # default value
    # Calculate value only for input r as the float number
    if isinstance(r, float):
        if abs(round(r, 12)) == 0.0:  # check that the argument provided with 0 value
            radial = round(pow(jv(n, 1E-11)/1E-11, 2), 11)  # approximation of the limit for the special condition jv(x)/x, where x -> 0
        else:
            radial = round(pow(jv(n, r)/r, 2), 12)
    return radial


def get_aberrated_psf(zernike_pol, r: float, theta: float) -> float:
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    # The analytical equation could be found in the Nijboer's thesis
    if m == 0:
        return 4.0*radial_func(n+1, r)
    # Approximation of the (-1, 1) polynomial
    elif m == -1 and n == 1:
        c1 = 169/96; c2 = 11/6; c3 = 15/64; c4 = 15/32; c5 = 1/12
        t1 = c1*radial_func(1, r) + c2*radial_func(2, r)*np.cos(theta) + radial_func(3, r)*(c3 + c4*np.cos(2.0*theta))
        t2 = c5*radial_func(4, r)*(np.cos(theta) + np.cos(3.0*theta))
        return pow((t1 + t2), 2)
    else:
        return 0.0  # !!! Should be exchanged to the integral or precalculated equations


def convolute_img_psf(img: np.ndarray, psf_kernel: np.ndarray) -> np.ndarray:
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
    convolved_img = convolved_img.astype(dtype=img_type)  # converting convolved image to the initial image
    return convolved_img


def get_psf_kernel(zernike_pol, calibration_coefficient: float) -> np.ndarray:
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
        if radial_func(n, i*calibration_coefficient) < 0.001:
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
            kernel[i, j] = get_aberrated_psf(zernike_pol, pixel_dist*calibration_coefficient, theta)
    return kernel


def show_ideal_psf(zernike_pol, size: int, calibration_coefficient: float, title: str = None):
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
            img[i, j] = get_aberrated_psf(zernike_pol, pixel_dist*calibration_coefficient, theta)
    if img[0, 0] > np.max(img)/100:
        __warn_message = f"The provided size for plotting PSF ({size}) isn't sufficient for proper representation"
        warnings.warn(__warn_message)
    if title is not None and len(title) > 0:
        plt.figure(title, figsize=(6, 6))
    else:
        plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=plt.cm.viridis); plt.tight_layout()
    return img


def plot_correlation(zernike_pol, size: int, calibration_coefficient: float, title: str = None, show_original: bool = True,
                     show_psf: bool = True):
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
    psf_kernel = get_psf_kernel(zernike_pol, calibration_coefficient)
    if show_psf:
        plt.figure(f"PSF for {title}", figsize=(6, 6)); plt.imshow(psf_kernel, cmap=plt.cm.viridis)
    conv_img = convolute_img_psf(img, psf_kernel)
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
    plt.close('all'); conv_pic_size = 12; detailed_plots_sizes = 80
    # p_img = show_ideal_psf(orders2, 20, pixel2um_coeff/2, "Piston"); ytilt_img = show_ideal_psf(orders3, 20, pixel2um_coeff/2, "Y Tilt")
    p_img = show_ideal_psf(orders2, detailed_plots_sizes, pixel2um_coeff_plot, "Detailed Piston")
    ytilt_img = show_ideal_psf(orders3, detailed_plots_sizes, pixel2um_coeff_plot, "Detailed Y Tilt")
    plot_correlation(orders2, conv_pic_size, pixel2um_coeff/2, "Piston", show_psf=False)
    plot_correlation(orders3, conv_pic_size, pixel2um_coeff/2, "Y Tilt", False, show_psf=False)
    plt.show()
