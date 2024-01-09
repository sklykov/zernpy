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

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calc_zernike_pol import define_orders
else:
    from .calc_zernike_pol import define_orders

# %% Module parameters
__docformat__ = "numpydoc"


# %% Functions
def get_aberrated_psf(zernike_pol, r: float, theta: float) -> np.ndarray:
    (m, n) = define_orders(zernike_pol)  # get polynomial orders
    if isinstance(r, int):  # Convert int to float explicitly
        r = float(r)
    # Defining pure radial component (angular indenpendent) - Jv(r)/r
    if isinstance(r, float):
        if abs(round(r, 12)) == 0.0:  # check that the argument provided with 0 value
            radial = 4.0*round(pow(jv(n+1, 1E-11)/1E-11, 2), 11)  # approximation of the limit for the special condition jv(x)/x, where x -> 0
        else:
            radial = 4.0*round(pow(jv(n+1, r)/r, 2), 12)
    # !!! There isn't analytical predefined equation for describing the angular/radial (they are connected) PSF in case m!= 0
    if m == 0:
        return 1.0*radial
    else:
        return 0.0  # !!! Should be exchanged to the integral equations


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
            theta = np.arctan2((i - i_center), (j - j_center))
            theta += np.pi  # shift angles to the range [0, 2pi]
            img[i, j] = get_aberrated_psf(zernike_pol, pixel_dist*calibration_coefficient, theta)
    if img[0, 0] > np.max(img)/100:
        __warn_message = f"The provided size for plotting PSF ({size}) isn't sufficient for proper representation"
        warnings.warn(__warn_message)
    if title is not None and len(title) > 0:
        plt.figure(title, figsize=(6, 6))
    else:
        plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=plt.cm.viridis, aspect='auto', origin='lower', extent=(0, size, 0, size))
    plt.tight_layout()


# %% Tests
if __name__ == '__main__':
    r = 0.0
    orders1 = (0, 2); orders2 = (0, 0)
    # Physical parameters
    wavelength = 0.55  # in micrometers
    k = 2.0*np.pi/wavelength  # angular frequency
    NA = 0.95  # microobjective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length)
    # Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
    pixel_size = 0.125  # in micrometers, physical length in pixels (um / pixels)
    pixel2um_coeff = k*NA*pixel_size  # coefficient used for relate pixels to physical units
    pixel2um_coeff_plot = k*NA*(pixel_size/10.0)  # coefficient used for better plotting with the reduced pixel size for preventing pixelated
    plt.close('all')
    show_ideal_psf(orders1, 40, pixel2um_coeff/2, "Defocus"); show_ideal_psf(orders1, 140, pixel2um_coeff_plot, "Detailed Defocus")
    show_ideal_psf(orders2, 40, pixel2um_coeff/2, "Piston"); show_ideal_psf(orders2, 150, pixel2um_coeff_plot, "Detailed Piston")
    plt.show()
