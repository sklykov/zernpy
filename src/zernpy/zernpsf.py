# -*- coding: utf-8 -*-
"""
PSF class definition based on Zernike polynomial for computation of its kernel for convolution / deconvolution.

@author: Sergei Klykov, @year: 2025, @licence: MIT \n

"""
# %% Global imports
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from math import pi
import time
from typing import Union, Sequence
from importlib.metadata import version

# Check if numba library installed for importing compilable methods
numba_installed = False  # default value for checking if 'numba' library is installed
try:
    import numba
    if numba is not None:
        numba_installed = True; numba_version = version('numba'); numba_ver_n = numba_version.split('.')
except ModuleNotFoundError:
    pass

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calculations.calc_psfs import (get_psf_kernel, lambda_char, radial_integral_s, radial_integral, get_kernel_size,
                                        convolute_img_psf, get_bumped_circle, save_psf, read_psf, get_psf_kernel_zerns)
    from zernikepol import ZernPol
    from utils.intmproc import DispenserManager
    if numba_installed:
        from calculations.calc_psfs_numba import get_psf_kernel_comp, methods_compiled, set_methods_compiled
    else:
        methods_compiled = False
else:
    from .calculations.calc_psfs import (get_psf_kernel, lambda_char, radial_integral_s, radial_integral, get_kernel_size,
                                         convolute_img_psf, get_bumped_circle, save_psf, read_psf, get_psf_kernel_zerns)
    from .zernikepol import ZernPol
    from .utils.intmproc import DispenserManager
    if numba_installed:
        from .calculations.calc_psfs_numba import get_psf_kernel_comp, methods_compiled, set_methods_compiled
    else:
        methods_compiled = False

# %% Module parameters
__docformat__ = "numpydoc"


# %% PSF class
class ZernPSF:
    """
    The PSF (2D) class for focal plane of a microscopic system assuming that the phase profile is described by the Zernike polynomial.

    In other words, the PSF describing the image of point source formed by the microscopic system. \n
    Check the set_physical_props() method for the list of expected physical parameters for PSF calculation and calculate_psf_kernel() for
    the references to the diffraction integral used for calculations. \n

    References
    ----------
    [1] Principles of Optics, by M. Born and E. Wolf, 4 ed., 1968  \n
    [2] https://en.wikipedia.org/wiki/Optical_aberration#Zernike_model_of_aberrations  \n
    [3] https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2016/03/ZP-Lecture-12.pdf   \n
    [4] https://www.researchgate.net/publication/236097143_Imaging_characteristics_of_Zernike_and_annular_polynomial_aberrations  \n
    [5] https://nijboerzernike.nl/_PDF/JOSA-A-19-849-2002.pdf#[0,{%22name%22:%22Fit%22}]  \n

    """

    # Class predefined values along with their types for providing reference how the default values computed
    kernel: np.ndarray = np.ones(shape=(1, 1)); kernel_size: int = 1  # by default, single point as the 2D matrix [[1.0]]
    NA: float = 1.0; wavelength: float = 0.532; expansion_coeff: float = 0.5  # in um (or other physical unit)
    zernpol: ZernPol = ZernPol(m=0, n=0)  # by default, Piston as the zero case (Airy pattern)
    polynomials = ()  # empty tuple - holder for provided Sequence with instances of ZernPol
    __physical_props_set: bool = False  # flag for saving that physical properties have been provided
    __warn_message: str = ""; pixel_size_nyquist: float = 0.5*0.5*wavelength  # based on the Abbe limit
    pixel_size = 0.98*pixel_size_nyquist  # default value based on the limit above
    alpha: float = expansion_coeff / wavelength  # the role of amplitude for PSF calculation
    airy: bool = False  # flag if the Piston is provided as the Zernike polynomial
    __ParallelCalc: DispenserManager = None; __integration_params: list = []  # for speeding up the calculations using several Processes
    n_int_r_points: int = 320; n_int_phi_points: int = 300  # integration parameters on the unit radius and angle - polar coordinates
    k: float = 2.0*pi/wavelength  # angular frequency
    json_file_path: str = ""  # shifted to the __init__ method to prevent putting path to API doc (by pydoc)
    coefficients: np.ndarray = None; amplitudes: np.ndarray = None  # for storing amplitudes of polynomials
    # Dev. Note: always carefully check the definition of variables above, since the error in their definition may cause hard traceable bug

    def __init__(self, zernpol: Union[ZernPol, Sequence[ZernPol]]):
        """
        Initiate the PSF wrapping class.

        Parameters
        ----------
        zernpol : ZernPol | Sequence[ZernPol]
            Single instance of ZernPol() class or Sequence with ZernPol instances.

        Raises
        ------
        ValueError
            1) If not instance(-s) of ZernPol class provided as the input parameter; \n
            2) If the provided sequence has zero length; \n
            3) If not all members of the provided sequence are instance of the "ZernPol" class; \n
            4) If the provided sequence contain repeated polynomials.

        Returns
        -------
        ZernPSF() class instance.

        """
        self.json_file_path = str(Path(__file__).parent.absolute())  # initialize default path as the root folder containing the script
        if not hasattr(zernpol, '__len__') and isinstance(zernpol, ZernPol):
            self.zernpol = zernpol; m, n = self.zernpol.get_mn_orders()
            if m == 0 and n == 0:
                self.airy = True
            else:
                self.airy = False
        elif hasattr(zernpol, '__len__'):
            seq_length = len(zernpol)
            # Empty sequence as the input not acceptable
            if seq_length == 0:
                raise ValueError("Provided empty Sequence, expected - Sequence with the 'ZernPol' class instances")
            # Single element should be unpacked and used
            elif seq_length == 1:
                self.zernpol = zernpol[0]; m, n = self.zernpol.get_mn_orders()
                if m == 0 and n == 0:
                    self.airy = True
                else:
                    self.airy = False
            # Several polynomials provided
            else:
                all_are_polls = True; osa_orders = []
                for pol in zernpol:
                    if not isinstance(pol, ZernPol):
                        all_are_polls = False; break
                    else:
                        m, n = pol.get_mn_orders()
                        osa_orders.append(pol.get_indices()[1])  # collect OSA index of each provided polynomial
                        if m == 0 and n == 0:
                            self.airy = True  # check if Airy pattern is among provided polynomials
                if all_are_polls:
                    self.polynomials = zernpol; self.zernpol = None
                    unique_osa_orders = set(osa_orders)  # check that all OSA orders are unique
                    if len(osa_orders) > len(unique_osa_orders):
                        raise ValueError("There might be some repeated polynomials provided, "
                                         + f"# of provided: {len(osa_orders)} and # of unique: {len(unique_osa_orders)}")
                else:
                    raise ValueError("Not all objects in Sequence are instances of the 'ZernPol' class")
        else:
            ValueError("ZernPSF class requires single 'ZernPol' instance or Sequence (class with __len__ attr.) with 'ZernPol' instances")

    # %% Set properties
    def set_physical_props(self, NA: float, wavelength: float, expansion_coeff: Union[float, Sequence[float]], pixel_physical_size: float):
        """
        Set parameters in physical units.

        Parameters
        ----------
        NA : float
            Numerical aperture of an objective, assumed usage of microscopic ones.
        wavelength : float
            Wavelength of monochromatic light (\u03BB) used for imaging in physical units (e.g., as \u00B5m).
        expansion_coeff : float | Sequence[float]
            Amplitude(-s) or expansion coefficient(-s) of the Zernike polynomial in physical units.
            Note that according to the used equation for PSF calculation it will be adjusted to the units of wavelength:
            alpha = expansion_coeff/wavelength. See the equation in the method "calculate_psf_kernel". \n
            Note that if Airy pattern (PSF for Piston polynomial) is provided, it's required to provide amplitude for it.
            However, this amplitude will be ignored in general for calculation because of its properties.
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
        [1] https://www.microscopyu.com/techniques/super-resolution/the-diffraction-barrier-in-optical-microscopy \n
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
        self.k = 2.0*pi/self.wavelength  # Calculate angular frequency (k)
        self.NA = NA; self.wavelength = wavelength  # save as the class properties
        # Sanity check of provided wavelength, pixel physical size (Nyquist criteria)
        self.pixel_size_nyquist = 0.5*0.5*wavelength/NA  # based on half of the Abbe resolution limit, see references in the docstring
        self.pixel_size_nyquist_eStr = "{:.3e}".format(self.pixel_size_nyquist)  # formatting calculated pixel size in scientific notation
        if pixel_physical_size <= 0.0:
            raise ValueError("Pixel physical size should be positive real number")
        if pixel_physical_size > self.pixel_size_nyquist:
            raise ValueError(f"Pixel physical size should be less than Nyquist pixel size {self.pixel_size_nyquist_eStr}"
                             + f" computed from the Abbe's resolution limit (0.5*{lambda_char}/NA)")
        self.pixel_size = pixel_physical_size
        # Check which type is provided as the expansion_coeff parameter
        if hasattr(expansion_coeff, '__len__'):
            coeffs_len = len(expansion_coeff)  # for checking how many amplitudes provided
            if coeffs_len != len(self.polynomials):
                if coeffs_len == 1 and len(self.polynomials) == 0:  # polynomials maybe provided also as a sequence with 1 element
                    expansion_coeff = float(expansion_coeff[0])
                    # Sanity check for the expansion coefficient of the polynomial
                    self.__warn_message = _sanity_check_expansion_coefficient(abs(expansion_coeff) / wavelength)
                    if len(self.__warn_message) > 0:
                        warnings.warn(self.__warn_message); self.__warn_message = ""
                    self.expansion_coeff = expansion_coeff; self.alpha = self.expansion_coeff / self.wavelength
                else:
                    raise ValueError(f"Length of provided coefficients ({coeffs_len}) is not equal to stored number "
                                     + f"of polynomials ({len(self.polynomials)})")
            else:
                self.coefficients = np.asarray(expansion_coeff)  # conversion to efficient array format
                # Sanity check for the maximum expansion coefficient of the polynomial
                max_module_coeff = max(np.max(self.coefficients), abs(np.min(self.coefficients)))
                self.__warn_message = _sanity_check_expansion_coefficient(abs(max_module_coeff) / wavelength, max_coeff_check=True)
                if len(self.__warn_message) > 0:
                    warnings.warn(self.__warn_message); self.__warn_message = ""
                self.amplitudes = self.coefficients / self.wavelength
        else:
            # Check consistency of provided type of polynomials and coefficients
            if self.zernpol is None and len(self.polynomials) > 1:  # only if 2 and more polynomials provided
                raise ValueError(f"Class initialized with #{len(self.polynomials)} polynomials and expected "
                                 + "to get their amplitudes in Sequence (as 'expansion_coeff' parameter)")
            # explicitly conversion to float for raising implicit conversion errors if not valid type provided
            if not isinstance(expansion_coeff, float):
                expansion_coeff = float(expansion_coeff)
            # Sanity check for the expansion coefficient of the polynomial
            self.__warn_message = _sanity_check_expansion_coefficient(abs(expansion_coeff) / wavelength)
            if len(self.__warn_message) > 0:
                warnings.warn(self.__warn_message); self.__warn_message = ""
            self.expansion_coeff = expansion_coeff; self.alpha = self.expansion_coeff / self.wavelength
        # Kernel size estimation (could be changed explicitly in the method 'set_calculation_props'). Redefine it for each call
        if self.zernpol is not None:  # for single polynomial
            self.kernel_size = get_kernel_size(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.alpha,
                                               wavelength=self.wavelength, NA=self.NA)
        else:
            max_kernel_size = 0; max_ampl = 0.0
            for i, pol in enumerate(self.polynomials):
                kernel_size = get_kernel_size(zernike_pol=pol, len2pixels=self.pixel_size, alpha=self.amplitudes[i],
                                              wavelength=self.wavelength, NA=self.NA)
                if max_kernel_size < kernel_size:
                    max_kernel_size = kernel_size
                if max_ampl < abs(self.amplitudes[i]):
                    max_ampl = abs(self.amplitudes[i])
            # increase depending on the # of provided pol-s and max provided amplitude the kernel size
            if 0.25 <= max_ampl < 0.5:
                self.kernel_size = max_kernel_size + int(round(0.6*len(self.polynomials), 0))
            elif 0.5 <= max_ampl < 0.75:
                self.kernel_size = max_kernel_size + int(round(1.0*len(self.polynomials), 0))
            elif max_ampl >= 0.75:
                self.kernel_size = max_kernel_size + int(round(1.4*len(self.polynomials), 0))
            else:
                self.kernel_size = max_kernel_size
            if self.kernel_size % 2 == 0:
                self.kernel_size += 1  # kernel size should odd
        self.__physical_props_set = True  # set internal flag True if no ValueError raised

    def set_calculation_props(self, kernel_size: int, n_integration_points_r: int, n_integration_points_phi: int) -> None:
        """
        Set calculation properties: kernel size, number of integration points on polar coordinates.

        Note that it's recommended to set the physical properties by set_physical_props() method for getting estimated
        required kernel size.

        Parameters
        ----------
        kernel_size : int
            Size of PSF kernel (2D matrix used for convolution). Should be odd integer not less than 3.
        n_integration_points_r : int
            Number of integration points used for calculation diffraction integral on the radius of the entrance pupil
            (normalized to the range [0.0, 1.0]). Should be integer not less than 20.
        n_integration_points_phi : int
            Number of integration points used for calculation diffraction integral on the polar angle phi of the entrance pupil
            (from the range [0.0, 2pi]). Should be integer not less than 36.

        Raises
        ------
        ValueError
            If sanity checks fail (if provided parameters less than sanity values).

        Returns
        -------
        None

        """
        # Sanity check for kernel size
        if kernel_size < 3:
            raise ValueError("Kernel size should be more than 3")
        # Check provided kernel size and the estimated one
        if kernel_size % 2 == 0:
            kernel_size += 1; print("Note that kernel_size should be odd integer for centering PSF kernel")
        if self.__physical_props_set:
            if kernel_size < self.kernel_size:
                self.__warn_message = (f"Empirically estimated kernel size = {self.kernel_size} based on the physical properties "
                                       + f"is larger than provided size = {kernel_size}. PSF may be truncated.")
                warnings.warn(self.__warn_message); self.__warn_message = ""
        else:
            self.__warn_message = "Please set the physical properties first for estimation of kernel size and after call this method"
            warnings.warn(self.__warn_message); self.__warn_message = ""
        self.kernel_size = kernel_size  # overwrite stored property, show before all warnings if kernel size is inconsistent
        # Sanity checks for n_integration_points_r and n_integration_points_phi
        if n_integration_points_r < 20:
            raise ValueError("# of integration points for radius should be not less than 20")
        if n_integration_points_phi < 36:
            raise ValueError("# of integration points for angle phi should be not less than 36")
        self.n_int_r_points = n_integration_points_r; self.n_int_phi_points = n_integration_points_phi
        # Associated with slow calculations warnings
        if n_integration_points_r > 500 and n_integration_points_phi > 400:
            self.__warn_message = "Selected integration precision may be unnecessary for PSF calculation and slow down it significantly"
            warnings.warn(self.__warn_message); self.__warn_message = ""
        if abs(n_integration_points_phi - 300) < 40 and abs(n_integration_points_r - 320) < 50 and self.kernel_size > 25:
            print(f"Note that the approximate calc. time: {int(round(self.kernel_size*self.kernel_size*38.5/1000, 0))} sec.")

    # %% Calculation
    def calculate_psf_kernel(self, suppress_warnings: bool = False, normalized: bool = True, verbose_info: bool = False,
                             accelerated: bool = None) -> np.ndarray:
        """
        Calculate PSF kernel using the specified or default calculation parameters and physical values.

        Calculation based on the diffraction integral for the circular aperture. The final equation is derived based from the 2 References. \n
        Kernel is defined as the image formed on the sensor (camera) by the diffraction-limited, ideal microscopic system.
        The diffraction integral is calculated numerically on polar coordinates, assuming circular aperture of
        an imaging system (micro-objective). \n
        The order of integration and used equations in short:
        1st - integration going on radius p, using trapezoidal rule: p\u2022(alpha\u2022zernike_pol.polynomial_value(p, phi) -
                                                                             r\u2022p\u2022cos(phi - theta))\u20221j \n
        2nd - integration going on angle phi, using Simpson rule, calling the returned integrals by 1st call for each phi and as the final
        output, it provides as the np.power(np.abs(integral_sum), 2)\u2022integral_normalization, there integral_normalization =
        1.0/(pi\u2022pi) - the square of the module of the diffraction integral (complex value), i.e. intensity as the PSF value. \n

        For details of implementation, explore methods in 'calculations' module, calc_psfs.py file. \n
        Note that the Nijboer's approximation for calculation of diffraction integral also has been tested, but the results is not in agreement
        with the reference [3] due to the large expansion coefficients for Zernike polynomials.  \n

        Note that for Piston (m=0, n=0) the exact equation is used, that provides Airy pattern as the result.

        Parameters
        ----------
        suppress_warnings : bool, optional
            Flag to suppress all possible warnings. The default is False.
        normalized : bool, optional
            Flag to return normalized PSF values to max=1.0. The default is True.
        verbose_info : bool, optional
            Flag to print out each step completion report and measured elapsed time in ms. The default is False.
        accelerated : bool, optional
            Flag to accelerate the calculations by using numba library for scripts compilation. Acceleration will be used automatically,
            if None is provided and 'numba' library is installed. The default is None.

        References
        ----------
        [1] Principles of Optics, by M. Born and E. Wolf, 4 ed., 1968  \n
        [2] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf  \n
        [3] PSF shapes: https://en.wikipedia.org/wiki/Zernike_polynomials#/media/File:ZernikeAiryImage.jpg  \n
        [4] https://opg.optica.org/ao/abstract.cfm?uri=ao-52-10-2062 (DOI: 10.1364/AO.52.002062)  \n

        Returns
        -------
        numpy.ndarray
            2D PSF kernel.

        """
        if len(self.__warn_message) > 0 and not suppress_warnings:
            warnings.warn(self.__warn_message)
        if not self.__physical_props_set and not suppress_warnings:
            self.kernel_size = 19; self.__warn_message = "\nPhysical properties haven't been set before, the default ones will be used"
            warnings.warn(self.__warn_message); self.__warn_message = ""
        global numba_installed  # access the flag
        # Check provided flag
        if numba_installed and accelerated is None:
            accelerated = True
        elif not numba_installed and accelerated is None:
            accelerated = False
        # Check if accelerated flag set to True but no numba installed
        if accelerated and not numba_installed and not suppress_warnings:
            self.__warn_message = "\nAcceleration isn't possible because 'numba' library not installed in the current environment"
            warnings.warn(self.__warn_message); self.__warn_message = ""
        # For providing performance verbose report
        if verbose_info:
            t1 = time.perf_counter()  # for explicit showing of performance
            if self.kernel_size*self.kernel_size >= 301:
                print("Kernel calculation started...")
        # Calculation using the vectorised form
        if self.zernpol is not None:
            if not accelerated or (accelerated and not numba_installed):
                self.kernel = get_psf_kernel(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.expansion_coeff,
                                             wavelength=self.wavelength, NA=self.NA, normalize_values=normalized, airy_pattern=self.airy,
                                             test_vectorized=True, verbose=verbose_info, kernel_size=self.kernel_size,
                                             n_int_r_points=self.n_int_r_points, n_int_phi_points=self.n_int_phi_points,
                                             suppress_warns=suppress_warnings)
            elif numba_installed:
                # if not suppress_warnings and not methods_compiled:
                #     self.__warn_message = ("\nCalculation methods have been not precompiled, the first step will take ~7 sec.,"
                #                            + " consider to run function 'force_get_psf_compilation()' before")
                #     warnings.warn(self.__warn_message); self.__warn_message = ""
                self.kernel = get_psf_kernel_comp(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.expansion_coeff,
                                                  wavelength=self.wavelength, NA=self.NA, normalize_values=normalized, verbose=verbose_info,
                                                  kernel_size=self.kernel_size, n_int_r_points=self.n_int_r_points,
                                                  suppress_warns=suppress_warnings, n_int_phi_points=self.n_int_phi_points)
        elif len(self.polynomials) > 0:
            if not accelerated or (accelerated and not numba_installed):
                self.kernel = get_psf_kernel_zerns(polynomials=self.polynomials, amplitudes=self.amplitudes, len2pixels=self.pixel_size,
                                                   wavelength=self.wavelength, NA=self.NA, normalize_values=normalized, verbose=verbose_info,
                                                   kernel_size=self.kernel_size, n_int_r_points=self.n_int_r_points,
                                                   n_int_phi_points=self.n_int_phi_points, suppress_warns=suppress_warnings)
            elif numba_installed:
                # if not suppress_warnings and not methods_compiled:
                #     self.__warn_message = ("\nCalculation methods have been not precompiled, the first step will take ~7 sec.,"
                #                            + " consider to run function 'force_get_psf_compilation()' before")
                #     warnings.warn(self.__warn_message); self.__warn_message = ""
                self.kernel = get_psf_kernel_comp(zernike_pol=self.polynomials, len2pixels=self.pixel_size, alpha=self.amplitudes,
                                                  wavelength=self.wavelength, NA=self.NA, normalize_values=normalized, verbose=verbose_info,
                                                  kernel_size=self.kernel_size, n_int_r_points=self.n_int_r_points,
                                                  suppress_warns=suppress_warnings, n_int_phi_points=self.n_int_phi_points)
        if verbose_info:
            passed_time_ms = int(round(1000.0*(time.perf_counter() - t1), 0))
            print("--------------------------------------------")
            if passed_time_ms > 1000:
                print(f"Overall kernel calculation took: {round(passed_time_ms/1000.0, 2)} sec.")
            else:
                print(f"Overall kernel calculation took: {passed_time_ms} ms.")
            print("")
        return self.kernel

    def plot_kernel(self, id_str: str = ""):
        """
        Plot on the external figure (matplotlib.pyplot.figure()) calculated PSF kernel.

        Parameters
        ----------
        id_str : str, optional
            String with ID for making unique Figure. The default is "".

        Returns
        -------
        None.

        """
        fig_title = ""  # default value
        if self.zernpol is not None:
            if self.airy:
                fig_title = f"Airy pattern with {round(self.expansion_coeff, 2)} expansion coeff. {id_str}"
            else:
                fig_title = (f"{self.zernpol.get_mn_orders()} {self.zernpol.get_polynomial_name(True)}: "
                             + f"{round(self.expansion_coeff, 2)} expansion coeff. {id_str}")
        else:
            orders = []
            for pol in self.polynomials:
                orders.append(pol.get_mn_orders())
            fig_title = f"Composed kernel for #{len(self.polynomials)} polynomials: {orders} {id_str}"
        if not plt.isinteractive():
            plt.ion()
        plt.figure(fig_title, figsize=(6, 6)); plt.imshow(self.kernel, cmap=plt.cm.viridis, origin='upper'); plt.colorbar(); plt.tight_layout()

    # %% Utilities
    def convolute_img(self, image: np.ndarray, scale2original: bool = True) -> np.ndarray:
        """
        Convolute the provided image with PSF kernel as 2D arrays and return the convolved image with the same type as the original one.

        Parameters
        ----------
        image : numpy.ndarray
            Sample image, not colour.
        scale2original : bool
            Convolution resulting image will be rescaled to the max intensity of the provided image if True. The default is True.

        Returns
        -------
        convolved_image : numpy.ndarray
            Result of convolution (by using function scipy.ndimage.convolve).

        """
        return convolute_img_psf(img=image, psf_kernel=self.kernel, scale2original=scale2original)

    def visualize_convolution(self, radius: float = 4.0, max_intensity: int = 255):
        """
        Plot convolution of the PSF kernel and sample image (even centered disk with blurred edges).

        Parameters
        ----------
        radius : float, optional
            Radius of the centered disk. The default is 4.0.
        max_intensity : int, optional
            Maximum intensity of the even disk. The default is 255.

        Returns
        -------
        None.

        """
        target_disk = get_bumped_circle(radius, max_intensity)  # get the sample image of the even centered circle with blurred edges
        if not plt.isinteractive():
            plt.ion()
        plt.figure("Sample Image: Disk", figsize=(6, 6)); plt.imshow(target_disk, cmap=plt.cm.viridis, origin='upper')
        plt.axis('off'); plt.tight_layout()
        convolved_img = self.convolute_img(image=target_disk); plt.figure("Convolved PSF and Disk", figsize=(6, 6))
        plt.imshow(convolved_img, cmap=plt.cm.viridis, origin='upper'); plt.axis('off'); plt.tight_layout()

    def crop_kernel(self, min_part_of_max: float = 0.01):
        """
        Crop redundant points from kernel.

        By default, all rows and columns containing any single value that more than 1% of max kernel value, are not cropped. \n
        Cropped kernel saved in class variable 'kernel'.

        Parameters
        ----------
        min_part_of_max : float, optional
            Part of kernel max that is used for checking rows / columns for cropping. The default is 0.01.

        Raises
        ------
        ValueError
            If parameter 'min_part_of_max' is not > 0.0 and not < 0.5.

        Returns
        -------
        None.

        """
        if self.kernel_size > 3:  # crop redundant points in kernel
            max_kernel_value = np.max(self.kernel)  # if kernel normalized, will be 1.0
            if min_part_of_max <= 0.0:
                raise ValueError("\nParameter 'min_part_of_max' should be > 0.0, but it is:", min_part_of_max)
            if min_part_of_max >= 0.5:
                raise ValueError("\nParameter 'min_part_of_max' should be < 0.5, but it is:", min_part_of_max)
            min_kernel_value = min_part_of_max*max_kernel_value  # by default, 1% of max kernel value
            # print("min value:", min_kernel_value)
            rows, cols = self.kernel.shape; stop_cropping_loop = False; step_row = 0; step_column = 0
            while not stop_cropping_loop:
                check_upper_row = np.max(self.kernel[0 + step_row, :]) < min_kernel_value
                check_bottom_row = np.max(self.kernel[rows - 1 - step_row, :]) < min_kernel_value
                check_left_column = np.max(self.kernel[:, 0 + step_column]) < min_kernel_value
                check_right_column = np.max(self.kernel[:, cols - 1 - step_column]) < min_kernel_value
                # print("Max values in rows and cols:", np.max(self.kernel[0 + step_row, :]), np.max(self.kernel[rows - 1 - step_row, :]),
                #       np.max(self.kernel[:, 0 + step_column]), np.max(self.kernel[:, cols - 1 - step_column]))
                if check_upper_row and check_bottom_row and check_left_column and check_right_column:
                    step_row += 1; step_column += 1
                else:
                    stop_cropping_loop = True
            if step_row > 0 and step_column > 0:
                self.kernel = self.kernel[step_row:rows-step_row, step_column:cols-step_column]
                self.kernel_size = self.kernel.shape[0]

    # %% I/O methods
    def save_json(self, abs_path: Union[str, Path] = None, overwrite: bool = False):
        """
        Save class attributes (PSF kernel, physical properties, etc.) in the JSON file for further reloading and avoiding long computation.

        Parameters
        ----------
        abs_path : str | Path, optional
            Absolute path for the file. If None, the file will be saved in the folder "saved_psfs" in the root of the package.
            The default is None.
        overwrite : bool, optional
            Flag for overwriting the existing file. The default is False.

        Returns
        -------
        None.

        """
        if abs_path is None:
            abs_path = Path(self.json_file_path).joinpath("saved_psfs")  # default folder for storing saved JSON files (root for the package)
        if not Path.exists(abs_path):
            abs_path = ""  # the folder "saved_psfs" will be created in the root of the package
        if isinstance(abs_path, Path):
            abs_path = str(abs_path)  # convert to the expected string format
        if self.kernel_size == 1:
            self.__warn_message = "Kernel most likely hasn't been calculated, the kernel size == 1 - default value"
            warnings.warn(self.__warn_message); self.__warn_message = ""
        if self.zernpol is not None:  # single polynomial saving
            self.json_file_path = save_psf(psf_kernel=self.kernel, NA=self.NA, wavelength=self.wavelength,
                                           expansion_coefficient=self.expansion_coeff, pixel_size=self.pixel_size,
                                           kernel_size=self.kernel_size, n_int_points_r=self.n_int_r_points,
                                           n_int_points_phi=self.n_int_phi_points, zernike_pol=self.zernpol,
                                           overwrite=overwrite, folder_path=abs_path)
        elif len(self.polynomials) > 0:  # saving several polynomials
            self.json_file_path = save_psf(psf_kernel=self.kernel, NA=self.NA, wavelength=self.wavelength,
                                           expansion_coefficient=self.coefficients, pixel_size=self.pixel_size,
                                           kernel_size=self.kernel_size, n_int_points_r=self.n_int_r_points,
                                           n_int_points_phi=self.n_int_phi_points, zernike_pol=self.polynomials,
                                           overwrite=overwrite, folder_path=abs_path)

    def read_json(self, abs_path: Union[str, Path] = None):
        """
        Read the JSON file with the saved attributes and setting it for the class.

        Parameters
        ----------
        abs_path : str | Path, optional
            Absolute path to the file. If None is provided, then the stored in the attribute path will be checked. The default is None.

        Returns
        -------
        None.

        """
        if abs_path is None:
            abs_path = self.json_file_path
        elif isinstance(abs_path, Path):
            abs_path = str(abs_path)
        json_data = read_psf(abs_path)  # raw parsed data from a file
        if json_data is not None:
            wavelen: float; na: float; a: float; pols: list; ps: float; read_props = 0
            for key, item in json_data.items():
                # Calculation properties + calculated kernel
                if key == "PSF Kernel":
                    self.kernel = np.asarray(item); read_props += 1
                elif key == "Kernel Size":
                    self.kernel_size = item; read_props += 1
                elif key == "# of integration points R":
                    self.n_int_r_points = item; read_props += 1
                elif key == "# of integration points angle":
                    self.n_int_phi_points = item; read_props += 1
                # Read the physical properties and store them in a list for further setting
                elif key == "NA":
                    na = item; read_props += 1
                elif key == "Wavelength":
                    wavelen = item; read_props += 1
                elif key == "Expansion Coefficient":
                    a = item; read_props += 1
                elif key == "Amplitudes":
                    a = np.asarray(item); read_props += 1
                elif key == "Pixel Size":
                    ps = item; read_props += 1
                # Getting and reassigning used polynomial(-s)
                elif key == "Polynomial":
                    osa_index = item; self.zernpol = ZernPol(osa=osa_index)
                elif key == "Polynomials":
                    osa_indices = item; pols = []; pols_len = len(osa_indices)
                    if pols_len == 1:
                        self.zernpol = ZernPol(osa=osa_indices[0])
                        if osa_indices[0] == 0:
                            self.airy = True
                        else:
                            self.airy = False
                    else:
                        for index in osa_indices:
                            pols.append(ZernPol(osa=index))
                        self.polynomials = pols; self.zernpol = None; self.airy = False
            # Assign read physical properties
            if read_props == 8:
                self.set_physical_props(NA=na, wavelength=wavelen, expansion_coeff=a, pixel_physical_size=ps)
            else:
                self.__warn_message = "Provided json file doesn't contain all necessary keys for physical / calculation properties"
                warnings.warn(self.__warn_message); self.__warn_message = ""
        else:
            self.__warn_message = "Provided path doesn't contain valid JSON data"
            warnings.warn(self.__warn_message); self.__warn_message = ""

    # %% Parallelized computing methods
    def initialize_parallel_workers(self):
        """
        Initialize 4 Processes() for performing integration.

        See intmproc.py script (utils module) for implementation details.
        Tests showed that this way doesn't provide performance gain.

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
        if self.kernel_size < 3 and self.zernpol is not None:
            self.kernel_size = get_kernel_size(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.alpha,
                                               wavelength=self.wavelength)
        if self.zernpol is None:
            self.kernel_size = 3
        self.kernel = np.zeros(shape=(self.kernel_size, self.kernel_size)); i_center = self.kernel_size//2; j_center = self.kernel_size//2
        calculated_points = 0
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                t1 = time.perf_counter(); pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))  # in pixels
                # Convert pixel distance in the required k*NA*pixel_dist*calibration coefficient
                distance = self.k*self.NA*self.pixel_size*pixel_dist  # conversion from pixel distance into phase multiplier
                theta = np.arctan2((i - i_center), (j - j_center))  # The PSF also has the angular dependency, not only the radial one
                theta += np.pi  # shift angles to the range [0, 2pi]
                if self.zernpol is not None:
                    self.kernel[i, j] = self.get_psf_point_r_parallel(r=distance, theta=theta); calculated_points += 1
                # print(f"Calculated point {[i, j]} from {[self.kernel_size-1, self.kernel_size-1]}")
                print(f"Calculated point #{calculated_points} from {self.kernel_size*self.kernel_size}, takes ms: ",
                      int(round(1000*(time.perf_counter() - t1), 0)))
        if normalize_values:
            self.kernel /= np.max(self.kernel)
        return self.kernel

    def deinitialize_workers(self):
        """
        Release initialized before Processes for performing parallel computation.

        Returns
        -------
        None.

        """
        if self.__ParallelCalc is not None:
            self.__ParallelCalc.close()


# %% Utility functions
def _sanity_check_expansion_coefficient(normalized_coefficient: float, max_coeff_check: bool = False) -> str:
    """
    Sanity check of the divided by wavelength expansion coefficient.

    Parameters
    ----------
    normalized_coefficient : float
        abs(expansion_coeff) / wavelength.
    max_coeff_check : bool, optional
        If this check happening for maximum abs. coefficient. The default is False.

    Returns
    -------
    str
        Warning message.

    """
    if normalized_coefficient > 10.0:
        if not max_coeff_check:
            return (f"\nExpansion coefficient supposed to be less than 10*{lambda_char} - an amplitude of a polynomial. "
                    + "Otherwise, the kernel size should be too big for sufficient PSF representation")
        else:
            return (f"\nMax abs. expansion coefficient supposed to be less than 10*{lambda_char} - an amplitude of a polynomial. "
                    + "Otherwise, the kernel size should be too big for sufficient PSF representation")
    else:
        return ""


def force_get_psf_compilation(verbose_report: bool = False) -> Union[tuple, None]:
    """
    Force compilation of computing functions for round and ellipse 'precise' shaped objects.

    verbose_report : bool, optional
        Flag for printing out the elapsed for compilation time. The default is False.

    Returns
    -------
    None or tuple with 2 used ZernPSF instances depending on 'numba_installed' flag.

    """
    if numba_installed:
        if verbose_report:
            print("Precompilation started..."); t1 = time.perf_counter()  # for explicit showing of performance
        # Check the version of numba for guarantee working of compilation calls
        try:
            if int(numba_ver_n[0]) == 0:
                if int(numba_ver_n[1]) <= 57:
                    if int(numba_ver_n[2]) < 1:
                        __warn_message = "\nRecommended numba version for compilation >= 0.57.1"; warnings.warn(__warn_message)
        except ValueError:
            __warn_message = "\nCannot parse 'numba' version, expected in format 'x.yy.zz' - all integers"; warnings.warn(__warn_message)
        # Compilation of methods for single polynomial PSF calculation
        NA = 0.2; wavelength = 0.55; pixel_size = wavelength / 0.82; ampl = 0.042
        zp_prc = ZernPol(m=-1, n=1); zpsf_prc = ZernPSF(zp_prc)  # Airy pattern - for compilation methods for single polynomial
        zpsf_prc.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf_prc.kernel_size = 1  # force to calculate only smallest kernel size for saving sometime
        # print("Precompilation kernel size for single pol.:", zpsf_prc.kernel_size)
        zpsf_prc.calculate_psf_kernel(normalized=True, accelerated=True, suppress_warnings=True, verbose_info=False)
        # Compilation of methods for multiple polynomials PSF calculation
        zp1 = ZernPol(m=-2, n=4); zp2 = ZernPol(m=3, n=3); zp3 = ZernPol(m=0, n=4); pols = (zp1, zp2, zp3); coeffs = (-0.01, 0.043, 0.027)
        zpsf_mpc = ZernPSF(pols); zpsf_mpc.set_physical_props(NA=0.2, wavelength=0.5, expansion_coeff=coeffs, pixel_physical_size=0.5/0.85)
        zpsf_mpc.kernel_size = 3  # force to calculate only small kernel size for saving sometime
        # print("Precompilation kernel size for several pol.:", zpsf_mpc.kernel_size)
        zpsf_mpc.calculate_psf_kernel(normalized=True, accelerated=True, suppress_warnings=True, verbose_info=False)
        set_methods_compiled()  # setting the flag implicitly in the module
        global methods_compiled; methods_compiled = True  # setting copied variable in this script
        if verbose_report:
            passed_time_ms = int(round(1000.0*(time.perf_counter() - t1), 0))
            if passed_time_ms > 1000:
                print(f"Precompilation took: {round(passed_time_ms/1000.0, 2)} sec.")
            else:
                print(f"Precompilation took: {passed_time_ms} ms.")
            print("--------------------------------------------")
        return (zpsf_prc, zpsf_mpc)
    else:
        __warn_message = "\nAcceleration isn't possible because 'numba' library not installed in the current environment"
        warnings.warn(__warn_message)
        return None


# %% Define default export classes and methods used with import * statement (import * from zernpsf)
__all__ = ['ZernPSF', 'force_get_psf_compilation']


# %% Test as the main script
if __name__ == "__main__":
    plt.close("all")  # close all opened before figures
    wavelength_um = 0.55  # used below in a several calls
    check_other_pols = False; check_small_na_wl = False  # flag for checking some other polynomials PSFs
    check_airy = False; check_common_psf = False; check_io_kernel = False; check_parallel_calculation = False; check_test = False
    check_faster_airy = False; check_test_conditions = False; check_test_conditions2 = False; check_several_pols = False
    check_edge_conditions = False; test_acceleration_single_pol = False; test_acceleration_few_pol = False
    prepare_pic_readme = False  # for plotting the sum of polynomials produced profile
    test_io_few_pols = False; standard_path = Path.home().joinpath("Desktop")  # for saving json on the Desktop
    check_acceleration_flag = False  # for testing fallback calculation with the wrong flag for calculate PSF kernel
    check_init_several_pols = False  # check calculation several pol-s: Airy (Piston) + some of polynomial
    check_airy_patterns = False  # check the difference between calculated by equation for Airy pattern and diffraction integral
    check_precompilation = False  # checks precompilation
    check_cropping = False  # checks how kernel is cropped

    # Common PSF for testing
    if check_common_psf:
        force_get_psf_compilation(True)
        zpsf1 = ZernPSF(ZernPol(m=1, n=1)); zpsf2 = ZernPSF(ZernPol(m=-3, n=3)); ampl = -0.43
        zpsf2.set_physical_props(NA=0.95, wavelength=wavelength_um, expansion_coeff=ampl, pixel_physical_size=wavelength_um/5.05)
        zpsf1.set_physical_props(NA=0.5, wavelength=0.6, expansion_coeff=0.25, pixel_physical_size=wavelength_um/5.5)
        zpsf2.set_calculation_props(kernel_size=zpsf2.kernel_size, n_integration_points_r=250, n_integration_points_phi=320)
        zpsf1.set_calculation_props(kernel_size=zpsf2.kernel_size, n_integration_points_r=220, n_integration_points_phi=360)
        kernel3 = zpsf2.calculate_psf_kernel(suppress_warnings=False, verbose_info=True); zpsf2.plot_kernel("for loop")
        zpsf2.visualize_convolution()  # Visualize convolution on the disk image
        if check_io_kernel:
            # default_path = Path(__file__).parent.joinpath("saved_psfs").absolute()  # default path for storing JSON files
            zpsf2.save_json(overwrite=True, abs_path=standard_path)
            zpsf1.read_json(zpsf2.json_file_path)  # for testing reading and assigning values to ZernPSF class (substitution)
        if check_parallel_calculation:
            zpsf2.initialize_parallel_workers(); kernel2 = zpsf2.get_kernel_parallel()
            zpsf2.plot_kernel("parallel"); zpsf2.deinitialize_workers()

    # Another Zernike polynomial, big NA
    if check_other_pols:
        zpsf3 = ZernPSF(ZernPol(m=0, n=4))
        zpsf3.set_physical_props(NA=1.25, wavelength=wavelength_um, expansion_coeff=0.4, pixel_physical_size=wavelength_um/5.25)
        zpsf3.calculate_psf_kernel(suppress_warnings=False, verbose_info=True); zpsf3.plot_kernel()

    # Another Zernike polynomial, average to small NA and wavelength
    if check_small_na_wl:
        NA = 0.45; wavelength = 0.4; pixel_size = wavelength / 5.25; ampl = 0.18
        zp2 = ZernPol(m=1, n=3); zpsf2 = ZernPSF(zp2)  # horizontal coma
        zpsf2.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf2.calculate_psf_kernel(normalized=True); zpsf2.plot_kernel()

    if check_airy:
        NA = 0.12; wavelength = 0.8; pixel_size = wavelength / 4.0; ampl = 1.25
        zp4 = ZernPol(m=0, n=0); zpsf4 = ZernPSF(zp4)  # piston for the Airy pattern
        zpsf4.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf4.calculate_psf_kernel(normalized=True); zpsf4.plot_kernel("Plus")
        zpsf4.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=-ampl, pixel_physical_size=pixel_size)
        zpsf4.calculate_psf_kernel(normalized=True); zpsf4.plot_kernel("Minus")

    if check_faster_airy:  # For set the test for pytest library
        NA = 0.35; wavelength = 0.55; pixel_size = wavelength / 3.05; ampl = -0.4
        zp4 = ZernPol(m=0, n=0); zpsf4 = ZernPSF(zp4)  # piston for the Airy pattern
        zpsf4.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf4.calculate_psf_kernel(normalized=True); zpsf4.plot_kernel()

    if check_test_conditions:
        NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 5.0; ampl = 0.55
        zp6 = ZernPol(m=0, n=2); zpsf6 = ZernPSF(zp6)  # defocus
        zpsf6.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)  # normal assignment
        zpsf6.set_calculation_props(kernel_size=zpsf6.kernel_size, n_integration_points_r=250, n_integration_points_phi=300)
        zpsf6.calculate_psf_kernel(normalized=True); zpsf6.plot_kernel()
    if check_test_conditions2:
        zp7 = ZernPol(m=1, n=3); zpsf7 = ZernPSF(zp7)  # horizontal coma
        NA = 0.4; wavelength = 0.4; pixel_size = wavelength / 3.2; ampl = 0.185  # Common physical properties
        zpsf7.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf7.calculate_psf_kernel(normalized=True); zpsf7.plot_kernel()

    # Test calculation of a PSF for several polynomials and I/O operations (see flags)
    if check_several_pols:
        zp1 = ZernPol(m=-2, n=2); zp2 = ZernPol(m=0, n=2); zp3 = ZernPol(m=2, n=2); pols = (zp1, zp2, zp3); coeffs = (-0.36, 0.25, 0.4)
        zpsf8 = ZernPSF(pols); zpsf8.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=coeffs, pixel_physical_size=0.5/4.5)
        composed_kernel = zpsf8.calculate_psf_kernel(normalized=True, verbose_info=True); zpsf8.plot_kernel()
        if test_io_few_pols:
            zpsf14 = ZernPSF(ZernPol(osa=19)); zpsf14.set_physical_props(NA=0.1, wavelength=0.4, expansion_coeff=0.82,
                                                                         pixel_physical_size=0.05)
            zpsf8.save_json(overwrite=True, abs_path=standard_path); zpsf14.read_json(zpsf8.json_file_path)  # save / read calculated kernel

    # Test some edge conditions - e.g., specifying 1 polynomial in a list with huge coefficient
    if check_edge_conditions:
        zp4 = ZernPol(m=3, n=3); pols2 = [zp4]; coeff = (5.1)
        zpsf9 = ZernPSF(pols2); zpsf9.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=coeff, pixel_physical_size=0.5/4.75)
        zpsf9.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=(-0.67), pixel_physical_size=0.5/4.75)
        zpsf9.calculate_psf_kernel(normalized=True, verbose_info=True); zpsf9.plot_kernel()
    # Test acceleration by using numba library utilities
    if test_acceleration_single_pol:
        force_get_psf_compilation(); NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 4.6; ampl = -0.16
        zp6 = ZernPol(m=0, n=2); zpsf6 = ZernPSF(zp6)  # defocus
        zpsf6.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        kernel_acc = zpsf6.calculate_psf_kernel(normalized=True, accelerated=True, verbose_info=True); zpsf6.plot_kernel("Accelerated")
        kernel_norm = zpsf6.calculate_psf_kernel(normalized=True); zpsf6.plot_kernel("Normal")
    if test_acceleration_few_pol:
        force_get_psf_compilation()
        zp1 = ZernPol(m=-2, n=2); zp2 = ZernPol(m=0, n=2); zp3 = ZernPol(m=2, n=2); pols = (zp1, zp2, zp3); coeffs = (-0.12, 0.15, 0.1)
        zpsf8 = ZernPSF(pols); zpsf8.set_physical_props(NA=0.35, wavelength=0.5, expansion_coeff=coeffs, pixel_physical_size=0.5/1.5)
        composed_kernel = zpsf8.calculate_psf_kernel(normalized=True, verbose_info=True); zpsf8.plot_kernel("Normal")
        composed_kernel_acc = zpsf8.calculate_psf_kernel(normalized=True, verbose_info=True, accelerated=True)
        zpsf8.plot_kernel("Accelerated")
    if prepare_pic_readme:
        force_get_psf_compilation(verbose_report=True)
        zp1 = ZernPol(m=-1, n=3); zp2 = ZernPol(m=2, n=4); zp3 = ZernPol(m=0, n=4); pols = (zp1, zp2, zp3); coeffs = (0.5, 0.21, 0.15)
        zpsf_pic = ZernPSF(pols); zpsf_pic.set_physical_props(NA=0.65, wavelength=0.6, expansion_coeff=coeffs, pixel_physical_size=0.6/5.0)
        zpsf_pic.calculate_psf_kernel(normalized=True, verbose_info=True, accelerated=True)
        zpsf_pic.plot_kernel("Vert. Coma Vert. 2nd Astigmatism Spherical")
    if check_acceleration_flag:
        zp16 = ZernPol(m=-1, n=3); zp18 = ZernPol(m=2, n=4); pols10 = (zp16, zp18); coeffs10 = (-0.1, 0.13); zpsf_acc = ZernPSF(pols10)
        zpsf_norm = ZernPSF(pols10); zpsf_acc.set_physical_props(NA=0.43, wavelength=0.6, expansion_coeff=coeffs10,
                                                                 pixel_physical_size=0.6/3.0)
        zpsf_norm.set_physical_props(NA=0.43, wavelength=0.6, expansion_coeff=coeffs10, pixel_physical_size=0.6/3.0)
        kern_acc = zpsf_acc.calculate_psf_kernel(normalized=True, accelerated=True)
        kern_norm = zpsf_norm.calculate_psf_kernel(normalized=True)
        kern_diff = np.round(kern_acc - kern_norm, 9)  # for checking the difference in calculations

    if check_test:
        pols = (ZernPol(osa=10), ZernPol(osa=15)); coeffs = (0.28, -0.33); NA = 0.35; wavelength = 0.55
        zpsf = ZernPSF(pols); zpsf.set_physical_props(NA, wavelength, expansion_coeff=coeffs, pixel_physical_size=wavelength / 3.5)
        zpsf.set_calculation_props(kernel_size=25, n_integration_points_r=200, n_integration_points_phi=180)
        psf_kernel = zpsf.calculate_psf_kernel(normalized=False); zpsf.plot_kernel()

    if check_init_several_pols:
        force_get_psf_compilation(True)
        zpsf30 = ZernPSF(zernpol=(ZernPol(osa=1), ZernPol(osa=5)))
        try:
            zpsf30.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=0.5, pixel_physical_size=0.5/5.0)
        except ValueError:
            print("Check for 1 ampl. and 2 pol-s passed")  # as expected, transfer to test function
        zpsf31 = ZernPSF(zernpol=(ZernPol(osa=0), ZernPol(m=-1, n=3)))
        zpsf31.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=[1.5, 0.24], pixel_physical_size=0.5/3.8)
        # zpsf31.calculate_psf_kernel(verbose_info=True, accelerated=False); zpsf31.plot_kernel("Not accelerated 2 pol-s")
        # kernel_not_acc = np.copy(zpsf31.kernel)
        zpsf31.calculate_psf_kernel(verbose_info=True, accelerated=True); zpsf31.plot_kernel("Airy 1.5")
        kernel_acc = np.copy(zpsf31.kernel)
        # zpsf31.kernel = np.abs(kernel_acc - kernel_not_acc); zpsf31.plot_kernel("Diff. 2 pol-s")
        zpsf31.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=[-1.5, 0.24], pixel_physical_size=0.5/3.8)
        zpsf31.calculate_psf_kernel(verbose_info=True, accelerated=True); zpsf31.plot_kernel("Airy -1.5")
        kernel_neg_acc = np.copy(zpsf31.kernel)
        zpsf31.kernel = np.abs(kernel_acc - kernel_neg_acc); zpsf31.plot_kernel("Diff. neg. pos. Airy + Coma")

    if check_airy_patterns:
        force_get_psf_compilation(verbose_report=True)
        zpsf40 = ZernPSF(zernpol=ZernPol(osa=0))
        zpsf40.set_physical_props(NA=0.95, wavelength=0.5, expansion_coeff=3.0, pixel_physical_size=0.5/4.0)
        zpsf40.calculate_psf_kernel(verbose_info=True, accelerated=False, normalized=False); zpsf40.plot_kernel("Not accelerated Airy")
        kernel_not_acc = np.copy(zpsf40.kernel)
        zpsf40.calculate_psf_kernel(verbose_info=True, accelerated=True, normalized=False); zpsf40.plot_kernel("Accelerated Airy")
        kernel_acc = np.copy(zpsf40.kernel)
        zpsf40.kernel = np.abs(kernel_acc - kernel_not_acc); zpsf40.plot_kernel("Diff. Airy")

    if check_precompilation:
        force_get_psf_compilation(True)
        zpsf50 = ZernPSF(zernpol=(ZernPol(m=1, n=3)))
        zpsf50.set_physical_props(NA=1.25, wavelength=0.52, expansion_coeff=0.5, pixel_physical_size=0.5/4.85)
        zpsf50.calculate_psf_kernel(verbose_info=True, accelerated=True, normalized=True); zpsf50.plot_kernel()

    if check_cropping:
        zpsf60 = ZernPSF(zernpol=(ZernPol(m=0, n=4)))
        zpsf60.set_physical_props(NA=1.25, wavelength=0.5, expansion_coeff=0.47, pixel_physical_size=0.5/5.0)
        zpsf60.calculate_psf_kernel(accelerated=True, verbose_info=True); zpsf60.plot_kernel("Not Cropped")
        original_kernel = np.copy(zpsf60.kernel)
        zpsf60.crop_kernel(min_part_of_max=0.025); zpsf60.plot_kernel("Cropped"); cropped_kernel = np.copy(zpsf60.kernel)
        print("Original kernel shape:", original_kernel.shape, "\nCropped kernel shape:", cropped_kernel.shape)
