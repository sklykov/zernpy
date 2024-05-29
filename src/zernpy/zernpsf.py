# -*- coding: utf-8 -*-
"""
PSF class definition based on Zernike polynomial for computation of its kernel for convolution / deconvolution.

@author: Sergei Klykov, @year: 2024, @licence: MIT \n

"""
# %% Global imports
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from functools import partial
from math import pi
import time

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calculations.calc_psfs import (get_psf_kernel, lambda_char, um_char, pi_char, radial_integral_s, radial_integral,
                                        get_kernel_size)
    from zernikepol import ZernPol
    from utils.intmproc import DispenserManager
else:
    from .calculations.calc_psfs import (get_psf_kernel, lambda_char, um_char, pi_char, radial_integral_s, radial_integral,
                                         get_kernel_size)
    from .zernikepol import ZernPol
    from .utils.intmproc import DispenserManager

# %% Module parameters
__docformat__ = "numpydoc"


# %% PSF class
class ZernPSF:
    """PSF (2D) associated with Zernike polynomial."""

    # Class predefined values along with their types for providing reference how the default values computed
    kernel: np.ndarray = np.ones(shape=(1, 1)); kernel_size: int = 1  # by default, single point as the 2D matrix [[1.0]]
    NA: float = 1.0; wavelength: float = 0.532; expansion_coeff: float = 0.5  # in um
    zernpol: ZernPol = ZernPol(m=0, n=0)  # by default, Piston as the zero case (Airy pattern)
    __physical_props_set: bool = False  # flag for saving that physical properties have been provided
    __warn_message: str = ""; pixel_size_nyquist: float = 0.5*0.5*wavelength  # based on the Abbe limit
    pixel_size = 0.98*pixel_size_nyquist  # default value based on the limit above
    alpha: float = expansion_coeff / wavelength  # the role of amplitude for PSF calculation
    airy: bool = False  # flag if the Piston is provided as the Zernike polynomial
    __ParallelCalc: DispenserManager = None; __integration_params: list = []  # for speeding up the calculations using several Processes
    n_int_r_points: int = 320; n_int_phi_points: int = 300  # integration parameters on the unit radius and angle - polar coordinates
    k: float = 2.0*pi/wavelength  # angular frequency

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
            m, n = self.zernpol.get_mn_orders()
            if m == 0 and n == 0:
                self.airy = True
            else:
                self.airy = False
        else:
            raise ValueError("ZernPSF class required ZernPol class as the input")

    def set_physical_props(self, NA: float, wavelength: float, expansion_coeff: float, pixel_physical_size: float) -> None:
        f"""
        Set parameters in physical units.

        Parameters
        ----------
        NA : float
            Numerical aperture of an objective, assumed usage of microscopic ones.
        wavelength : float
            Wavelength of monochromatic light ({lambda_char}) used for imaging in physical units (e.g., as {um_char}).
        expansion_coeff : float
            Amplitude or expansion coefficient of the Zernike polynomial in physical units.
            Note that according to the used equation for PSF calculation it will be adjusted to the units of wavelength:
            alpha = expansion_coeff/wavelength. See the equation in the method "calculate_psf_kernel".
        pixel_physical_size : float
            Pixel size of the formed image in physical units (do not mix up with the physical sensor (camera) pixel size!).
            The sanity check performed as the comparison with the Abbe resolution limit (see ref. [1] and [2]), provided pixel size
            should be less than this limit.

        Raises
        ------
        ValueError
            If one of the sanity check has been failed, check the Error message for details.

        References
        ----------
        [1] https://www.microscopyu.com/techniques/super-resolution/the-diffraction-barrier-in-optical-microscopy
        [2] https://www.edinst.com/de/blog/the-rayleigh-criterion-for-microscope-resolution/

        Returns
        -------
        None.

        """
        # Sanity check for NA
        if NA < 0.0 or NA > 1.7:
            raise ValueError("NA should lay in the range of (0.0, 1.7] at most - for common microscopic objectives")
        # Sanity check for wavelength
        if wavelength <= 0.0:
            raise ValueError("Wavelength should be positive real number")
        self.NA = NA; self.wavelength = wavelength  # save as the class properties
        # Sanity check of provided wavelength, pixel physical size (Nyquist createria)
        self.pixel_size_nyquist = 0.5*0.5*wavelength/NA  # based on half of the Abbe resolution limit, see references in the docstring
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
        self.expansion_coeff = expansion_coeff; self.alpha = self.expansion_coeff / self.wavelength
        self.k = 2.0*pi/self.wavelength  # Calculate angular frequency (k)
        self.kernel_size = get_kernel_size(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.alpha)
        self.__physical_props_set = True  # set internal flag True if no ValueError raised


    def calculate_psf_kernel(self, suppress_warnings: bool = False, normalized: bool = True, verbose_info: bool = False) -> np.ndarray:
        """
        Calcualte PSF kernel using specified or default parameters and physical values.

        Parameters
        ----------
        suppress_warnings : bool, optional
            Flag to suppress all possible warnings. The default is False.
        normalized : bool, optional
            Flag to return normalized PSF values to max=1.0. The default is True.
        verbose_info : bool, optional
            Flag to print out each step complition report and measured elapsed time in ms. The default is False.

        Returns
        -------
        numpy.ndarray
            PSF kernel.

        """
        if len(self.__warn_message) > 0 and not suppress_warnings:
            warnings.warn(self.__warn_message)
        if not self.__physical_props_set and not suppress_warnings:
            self.__warn_message = "Physical properties for calculation haven't been set, the default values will be used"
            warnings.warn(self.__warn_message)
        # Calculation using the vectorised form
        self.kernel = get_psf_kernel(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.expansion_coeff, wavelength=self.wavelength,
                                     NA=self.NA, normalize_values=normalized, airy_pattern=self.airy, test_vectorized=True, verbose=verbose_info,
                                     kernel_size=self.kernel_size, n_int_r_points=self.n_int_r_points, n_int_phi_points=self.n_int_phi_points)
        return self.kernel

    def plot_kernel(self, id_str: str = ""):
        """
        Plot on the external figure (matplotlib.pyplot.figire()) calculated PSF kernel.

        Parameters
        ----------
        id_str : str, optional
            String with ID for making unique Figure. The default is "".

        Returns
        -------
        None.

        """
        if self.airy:
            fig_title = f"Airy pattern with {round(self.expansion_coeff, 2)} expansion coeff."
        else:
            fig_title = (f"{self.zernpol.get_mn_orders()} {self.zernpol.get_polynomial_name(True)}: "
                        + f"{round(self.expansion_coeff, 2)} expansion coeff. {id_str}")
        plt.figure(fig_title, figsize=(6, 6)); plt.imshow(self.kernel, cmap=plt.cm.viridis, origin='upper'); plt.tight_layout()

    # %% Parallelized computing methods
    def initialize_parallel_workers(self):
        """
        Initialize 4 Processes() for performing integration.

        See intmproc.py script for implementation details.

        Returns
        -------
        None.

        """
        if self.__ParallelCalc is None and not self.airy:
            self.__integration_params = [(i, i, i, i, i, i) for i in range(10)]  # placeholder only, will be replaced with the actual list later
            self.__ParallelCalc = DispenserManager(compute_func=radial_integral_s, params_list=self.__integration_params,
                                                   n_workers=4, verbose_info=False)

    def get_psf_point_r_parallel(self, r: float, theta: float) -> float:
        """
        Parallel implementation of numerical integration.

        Parameters
        ----------
        r : float
            Input radial polar coordinate.
        theta : float
            Input angular polar coordinate.

        Returns
        -------
        float
            Each point for PSF kernel.

        """
        h_phi = 2.0*pi/self.n_int_phi_points; even_sum = 0.0j; odd_sum = 0.0j
        if self.__ParallelCalc is not None:
            # t1 = time.perf_counter()
            constants = (self.zernpol, r, theta, self.alpha, self.n_int_r_points)
            self.__integration_params = [(i*h_phi, constants) for i in range(2, self.n_int_phi_points-2, 2)]
            self.__ParallelCalc.update_params(self.__integration_params)
            # print("Forming params takes ms:", int(round(1000*(time.perf_counter() - t1), 0)))
            # t1 = time.perf_counter()
            even_sum = np.sum(np.asarray(self.__ParallelCalc.compute()))
            # print("Calculation takes ms:", int(round(1000*(time.perf_counter() - t1), 0)))
            self.__integration_params = [(i*h_phi, constants) for i in range(1, self.n_int_phi_points-1, 2)]
            self.__ParallelCalc.update_params(self.__integration_params)
            odd_sum = np.sum(np.asarray(self.__ParallelCalc.compute()))
        # Simpson integration rule implementation
        yA = radial_integral(self.zernpol, r, theta, 0.0, self.alpha, self.n_int_r_points)
        yB = radial_integral(self.zernpol, r, theta, 2.0*pi, self.alpha, self.n_int_r_points)
        integral_sum = (h_phi/3.0)*(yA + yB + 2.0*even_sum + 4.0*odd_sum); integral_normalization = 1.0/(pi*pi)
        return np.power(np.abs(integral_sum), 2)*integral_normalization

    def get_kernel_parallel(self, normalize_values: bool = True) -> np.ndarray:
        """
        Parallelized implementation of PSF kernel calculation.

        Note it's much less effective than common calculate_psf_kernel() method.

        Parameters
        ----------
        normalize_values : bool, optional
            Flag to normalize values. The default is True.

        Returns
        -------
        numpy.ndarray
            PSF kernel.

        """
        if self.kernel_size < 3:
            self.kernel_size = get_kernel_size(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.alpha, wavelength=self.wavelength)
        self.kernel = np.zeros(shape=(self.kernel_size, self.kernel_size)); i_center = self.kernel_size//2; j_center = self.kernel_size//2
        calculated_points = 0
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                t1 = time.perf_counter()
                pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))  # in pixels
                # Convert pixel distance in the required k*NA*pixel_dist*calibration coefficient
                distance = self.k*self.NA*self.pixel_size*pixel_dist  # conversion from pixel distance into phase multiplier in the diffraction integral
                theta = np.arctan2((i - i_center), (j - j_center))  # The PSF also has the angular dependency, not only the radial one
                theta += np.pi  # shift angles to the range [0, 2pi]
                self.kernel[i, j] = self.get_psf_point_r_parallel(r=distance, theta=theta)
                # print(f"Calculated point {[i, j]} from {[self.kernel_size-1, self.kernel_size-1]}")
                calculated_points += 1
                print(f"Calculated point #{calculated_points} from {self.kernel_size*self.kernel_size}, takes ms: ",
                      int(round(1000*(time.perf_counter() - t1), 0)))
        if normalize_values:
            self.kernel /= np.max(self.kernel)
        return self.kernel

    def deinitialize_workers(self):
        """
        Release initialized before Processes for performung parallel computation.

        Returns
        -------
        None.

        """
        if self.__ParallelCalc is not None:
            self.__ParallelCalc.close()


# %% Test as the main script
if __name__ == "__main__":
    # zpsf1 = ZernPSF(ZernPol(m=0, n=0)); kernel1 = zpsf1.calculate_psf_kernel(suppress_warns=True); zpsf1.plot_kernel()  # Airy
    zpsf2 = ZernPSF(ZernPol(m=-3, n=3))
    kernel3 = zpsf2.calculate_psf_kernel(suppress_warnings=False, verbose_info=False); zpsf2.plot_kernel("for loop")
    # zpsf2.initialize_parallel_workers(); kernel2 = zpsf2.get_kernel_parallel(); zpsf2.plot_kernel("parallel"); zpsf2.deinitialize_workers()

