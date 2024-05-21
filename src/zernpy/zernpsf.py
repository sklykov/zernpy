# -*- coding: utf-8 -*-
"""
PSF class definition based on Zernike polynomial for computation of its kernel for convolution / deconvolution.

@author: Sergei Klykov, @year: 2024, @licence: MIT \n

"""
# %% Global imports
import numpy as np
from pathlib import Path
import warnings

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calculations.calc_psfs import (get_psf_kernel, lambda_char, um_char, pi_char)
    from zernikepol import ZernPol
else:
    from .calculations.calc_psfs import (get_psf_kernel, lambda_char, um_char, pi_char)
    from .zernikepol import ZernPol

# %% Module parameters
__docformat__ = "numpydoc"


# %% PSF class
class ZernPSF:
    """PSF (2D) associated with Zernike polynomial."""

    # Class predefined values along with their types for providing reference how the default values computed
    kernel: np.ndarray = np.ones(shape=(1, 1))  # by default, single point as the 2D matrix [[1.0]]
    NA: float = 1.0; wavelength: float = 0.532; expansion_coeff: float = 1.0; pixel_size: float = wavelength / 5.0
    zernpol: ZernPol = ZernPol(m=0, n=0)  # by default, Piston as the zero case (Airy pattern)
    __physical_props_set: bool = False  # flag for saving that physical properties have been provided
    __warn_message: str = ""; pixel_size_nyquist: float = 0.5*0.5*wavelength
    alpha: float = expansion_coeff / wavelength  # the role of amplitude for PSF calculation

    def __init__(self, zernpol: ZernPol):
        """
        Initiate the PSF wrapping class.

        Parameters
        ----------
        zernpol : ZernPol
            See description of ZernPol() class.

        Raises
        ------
        ValueError
            If not instance of ZernPol class provided as the input parameter.

        Returns
        -------
        None.

        """
        if isinstance(zernpol, ZernPol):
            self.zernpol = zernpol
        else:
            raise ValueError("ZernPSF class required ZernPol class as the input")

    def set_physical_props(self, NA: float, wavelength: float, expansion_coeff: float, pixel_physical_size: float):
        # Sanity check for NA
        if NA < 0.0 or NA > 1.7:
            raise ValueError("NA should lay in the range of (0.0, 1.7] at most - for common microscopic objectives")
        # Sanity check for wavelength
        if wavelength <= 0.0:
            raise ValueError("Wavelength should be positive real number")
        # Sanity check of provided wavelength, pixel physical size (Nyquist createria)
        self.pixel_size_nyquist = 0.5*0.5*wavelength/NA  # based on half of the Abbe resolution limit
        # Ref1: https://www.microscopyu.com/techniques/super-resolution/the-diffraction-barrier-in-optical-microscopy
        # Ref2: https://www.edinst.com/de/blog/the-rayleigh-criterion-for-microscope-resolution/
        self.pixel_size_nyquist_eStr = "{:.3e}".format(self.pixel_size_nyquist)  # formatting calculated pixel size in scientific notation
        if pixel_physical_size <= 0.0:
            raise ValueError("Pixel physical size should be positive real number")
        if pixel_physical_size > self.pixel_size_nyquist:
            raise ValueError(f"Pixel physical size should be less than Nyquist pixel size {self.pixel_size_nyquist_eStr}"
                             + f"computed from the Abbe's resolution limit (0.5*{lambda_char}/NA)")
        self.pixel_size = pixel_physical_size
        # Sanity check for the expansion coefficient of the polynomial
        if abs(expansion_coeff) / wavelength > 10.0:
            self.__warn_message = (f"Expansion coefficient spupposed to be less than 10*{lambda_char} - an amplitude of Zernike polynomial"
                                   + "Otherwise, the kernel size should be too big for sufficient PSF representation")
            warnings.warn(self.__warn_message)
        self.alpha = expansion_coeff / wavelength  # required by the used PSF calculation equation
        self.__physical_props_set = True  # set internal flag True if no ValueError raised


    def calculate_psf_kernel(self, suppress_warnings: bool = False):
        if len(self.__warn_message) > 0 and not suppress_warnings:
            warnings.warn(self.__warn_message)
        if not self.__physical_props_set:
            self.__warn_message = "Physical properties for calculation hasn't been set, the default values will be used"
            warnings.warn(self.__warn_message)


# %% Test as the main script
if __name__ == "__main__":
    zpsf = ZernPSF(ZernPol(m=0, n=0))
