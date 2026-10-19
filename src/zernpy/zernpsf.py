# -*- coding: utf-8 -*-
"""
PSF class definition based on Zernike polynomial for computation of its kernel for convolution / deconvolution.

@author: Sergei Klykov, @year: 2026, @license: MIT \n

"""
# %% Global imports
import time
import warnings
from importlib.metadata import version
from math import pi
from numbers import Real
from pathlib import Path
from typing import Optional, Sequence, SupportsFloat, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Check if numba library installed for importing compilable methods
numba_installed = False  # default value for checking if 'numba' library is installed
try:
    import numba
    if numba is not None:
        numba_installed = True; numba_version = version('numba'); numba_ver_n = numba_version.split('.')
except ModuleNotFoundError:
    pass

# %% Local (package-scoped) imports
from .calculations.calc_psfs import (
    convolute_img_psf,
    get_bumped_circle,
    get_kernel_size,
    get_psf_kernel,
    get_psf_kernel_zerns,
    lambda_char,
    read_psf,
    save_psf,
)
from .zernikepol import ZernPol

if numba_installed:
    from .calculations.calc_psfs_numba import get_psf_kernel_comp

# %% Module parameters
__docformat__ = "numpydoc"


# %% PSF class
class ZernPSF:
    """
    Compute two-dimensional focal-plane point spread functions (PSFs) from Zernike wavefronts.

    A PSF describes the image of a point source. The pupil phase can contain one Zernike polynomial
    or a sum of polynomials. \n
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
    NA: float = 1.0; wavelength: float = 0.532; expansion_coeff: float = 0.5  # wavelength and expansion_coeff share a length unit; NA is dimensionless
    zernpol: Optional[ZernPol] = None; polynomials: tuple = ()  # several provided polynomials will be stored in the last var.
    __physical_props_set: bool = False  # flag for saving that physical properties have been provided
    __warn_message: str = ""; pixel_size_nyquist: float = 0.5*0.5*wavelength  # half the Abbe resolution limit for the default NA = 1
    pixel_size_nyquist_eStr = "{:.3e}".format(pixel_size_nyquist)
    pixel_size = 0.98*pixel_size_nyquist  # default value based on the limit above
    alpha: float = (2.0*pi*expansion_coeff) / wavelength  # the role of amplitude for PSF calculation (converted to radians!)
    airy: bool = False  # flag if the Piston is provided as the Zernike polynomial
    n_int_r_points: int = 320; n_int_phi_points: int = 300  # integration parameters on the unit radius and angle - polar coordinates
    k: float = 2.0*pi/wavelength  # wave number in radians per unit length
    json_file_path: str = ""  # shifted to the __init__ method to prevent putting path to API doc (by pydoc)
    coefficients: Optional[np.ndarray] = None; amplitudes: Optional[np.ndarray] = None  # for storing amplitudes of polynomials
    # Dev. Note: always carefully check the definition of variables above, since the error in their definition may cause hard traceable bug

    def __init__(self, zernpol: Union[ZernPol, Sequence[ZernPol]]):
        """
        Initialize a PSF calculation for one or more Zernike polynomials.

        Parameters
        ----------
        zernpol : ZernPol | Sequence[ZernPol]
            Single instance of ZernPol() class or Sequence with ZernPol instances.

        Raises
        ------
        ValueError
            If the sequence is empty, contains invalid objects in the multiple-polynomial case, or contains repeated polynomials.
        TypeError
            If the input is neither a ZernPol instance nor a sequence.
        AttributeError
            If the only item in a sequence does not provide the ZernPol order interface.

        """
        self.json_file_path = str(Path(__file__).parent.absolute())  # start with the package directory before storing a saved filename
        if isinstance(zernpol, ZernPol):
            self.zernpol = zernpol; m, n = self.zernpol.get_mn_orders()
            if m == 0 and n == 0:
                self.airy = True
            else:
                self.airy = False
        else:
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
                    self.polynomials = tuple(zernpol); self.zernpol = None
                    unique_osa_orders = set(osa_orders)  # check that all OSA orders are unique
                    if len(osa_orders) > len(unique_osa_orders):
                        raise ValueError("There might be some repeated polynomials provided, "
                                         + f"# of provided: {len(osa_orders)} and # of unique: {len(unique_osa_orders)}")
                else:
                    raise ValueError("Not all objects in Sequence are instances of the 'ZernPol' class")

    # %% Set properties
    def set_physical_props(self, NA: float, wavelength: float, expansion_coeff: Union[SupportsFloat, Sequence[SupportsFloat], np.ndarray],
                           pixel_physical_size: float):
        """
        Set parameters required for a PSF computation in physical units.

        Parameters
        ----------
        NA : float
            Dimensionless numerical aperture of the objective, within (0, 1.7].
        wavelength : float
            Wavelength of monochromatic light (\u03BB) used for imaging in physical units (e.g., as \u00B5m).
        expansion_coeff : SupportsFloat or Sequence[SupportsFloat] or numpy.ndarray
            Amplitude(-s) or expansion coefficient(-s) of the Zernike polynomial in physical units.
            Use the same length unit as wavelength. Coefficients are converted to phase amplitudes in radians:
            alpha = 2.0*pi*expansion_coeff/wavelength. See the equation in calculate_psf_kernel(). \n
            The 2*pi factor was introduced in version 0.1.2. \n
            Note that if Airy pattern (PSF for Piston polynomial) is provided, it's required to provide amplitude for it.
            However, this amplitude will be ignored in general for calculation because of its properties.
        pixel_physical_size : float
            Sampling interval in the object plane, in the same length unit as wavelength.
            It must be positive and no greater than wavelength/(4*NA), half the Abbe resolution limit.\n
            Way of calculation = pixel physical size = pixel camera size (\u00B5m) / Total Magnification, where:\n
            Total Magnification (M) = Objective Magn. * Tube Lens Magn. (Mismatch) * Camera Adapter Magn. \n

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
        if NA <= 0.0 or NA > 1.7:
            raise ValueError("NA should lay in the range of (0.0, 1.7] at most - for common microscopic objectives")
        # Sanity check for wavelength
        if wavelength <= 0.0:
            raise ValueError("Wavelength should be positive real number")
        self.NA = NA; self.wavelength = wavelength  # save as the class properties
        self.k = 2.0*pi/self.wavelength  # wave number in radians per unit length
        self.amplitudes: np.ndarray  # set explicitly expected data type
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
        if isinstance(expansion_coeff, (Sequence, np.ndarray)):  # effectively, it should be some sequence / numpy array with numbers
            coeffs_len = len(expansion_coeff)  # for checking how many amplitudes provided
            if coeffs_len != len(self.polynomials):
                if coeffs_len == 1 and len(self.polynomials) == 0:  # polynomials maybe provided also as a sequence with 1 element
                    expansion_coeff = float(expansion_coeff[0])
                    # Sanity check for the expansion coefficient of the polynomial
                    self.__warn_message = _sanity_check_expansion_coefficient(abs(2.0*pi*expansion_coeff) / wavelength)
                    if len(self.__warn_message) > 0:
                        warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
                    self.expansion_coeff = expansion_coeff; self.alpha = (2.0*pi*self.expansion_coeff) / self.wavelength
                else:
                    raise ValueError(f"Length of provided coefficients ({coeffs_len}) is not equal to stored number "
                                     + f"of polynomials ({len(self.polynomials)})")
            else:
                self.coefficients = np.asarray(expansion_coeff)  # conversion to efficient array format
                # Sanity check for the maximum expansion coefficient of the polynomial
                max_module_coeff = max(np.max(self.coefficients), abs(np.min(self.coefficients)))
                self.__warn_message = _sanity_check_expansion_coefficient(abs(2.0*pi*max_module_coeff) / wavelength, max_coeff_check=True)
                if len(self.__warn_message) > 0:
                    warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
                self.amplitudes = (2.0*pi*self.coefficients) / self.wavelength
        else:
            # Check consistency of provided type of polynomials and coefficients
            if self.zernpol is None and len(self.polynomials) > 1:  # only if 2 and more polynomials provided
                raise ValueError(f"Class initialized with #{len(self.polynomials)} polynomials and expected "
                                 + "to get their amplitudes in Sequence (as 'expansion_coeff' parameter)")
            # explicitly conversion to float for raising implicit conversion errors if not valid type provided
            if not isinstance(expansion_coeff, float):
                expansion_coeff = float(expansion_coeff)
            # Sanity check for the expansion coefficient of the polynomial
            if not self.airy:
                self.__warn_message = _sanity_check_expansion_coefficient(abs(2.0*pi*expansion_coeff) / wavelength)
            if len(self.__warn_message) > 0:
                warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
            self.expansion_coeff = expansion_coeff; self.alpha = (2.0*pi*self.expansion_coeff) / self.wavelength
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

    def set_calculation_props(self, kernel_size: int, n_integration_points_r: int = 320, n_integration_points_phi: int = 300):
        """
        Set calculation-only properties: PSF kernel size, number of integration points on polar coordinates.

        Note that it's recommended to set the physical properties by set_physical_props() method for getting estimated
        required kernel size.

        Parameters
        ----------
        kernel_size : int
            Kernel width and height in pixels, at least 3. Even sizes are increased by one to center the PSF.
        n_integration_points_r : int, optional
            Number of integration points used for calculation diffraction integral on the radius of the entrance pupil
            (normalized to the range [0.0, 1.0]). Should be integer not less than 20. The default is 320.
        n_integration_points_phi : int, optional
            Number of integration points used for calculation diffraction integral on the polar angle phi of the entrance pupil
            (from the range [0.0, 2pi]). Should be integer not less than 36. The default is 300.

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
                self.__warn_message = (f"\nEmpirically estimated kernel size = {self.kernel_size} based on the physical properties "
                                       + f"is larger than provided size = {kernel_size}. PSF may be truncated.")
                warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
        else:
            self.__warn_message = "\nPlease set the physical properties first for estimation of kernel size and after call this method"
            warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
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
            warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
        if abs(n_integration_points_phi - 300) < 40 and abs(n_integration_points_r - 320) < 50 and self.kernel_size > 25:
            print(f"Note that the approximate calc. time: {int(round(self.kernel_size*self.kernel_size*38.5/1000, 0))} sec.")

    # %% Calculation
    def calculate_psf_kernel(self, suppress_warnings: bool = False, normalized: bool = True, verbose_info: bool = False,
                             accelerated: Optional[bool] = None) -> np.ndarray:
        """
        Calculate PSF kernel using the specified or default calculation parameters and physical values.

        Calculation based on the diffraction integral for the circular aperture. The final equation is derived based from the 2 References. \n
        The kernel samples the intensity response of the optical system with the supplied pupil aberrations.
        The diffraction integral is calculated numerically on polar coordinates, assuming circular aperture of
        an imaging system (micro-objective). \n
        The order of integration and used equations in short: \n
        1st - integrate over the pupil radius p using the trapezoidal rule, with integrand
        p * exp(1j * (alpha * Z(p, phi) - r * p * cos(phi - theta))). Here Z is the normalized Zernike
        polynomial, alpha is its phase coefficient in radians, and r is the scaled image-plane radius
        k * NA * pixel_size * pixel_distance, where k = 2*pi/wavelength. \n
        2nd - integration going on angle phi, using Simpson rule, calling the returned integrals by 1st call for each phi and as the final
        output, it provides as the np.power(np.abs(integral_sum), 2)*integral_normalization, there integral_normalization =
        1.0/(pi\u2022pi) - the square of the module of the diffraction integral (complex value), i.e. intensity as the PSF value. \n

        For details of implementation, explore methods in 'calculations' module, calc_psfs.py file. \n
        Note that the Nijboer's approximation for calculation of diffraction integral also has been tested, but the results is not in agreement
        with the reference [3] due to the large expansion coefficients for Zernike polynomials.  \n

        For Piston (m=0, n=0), the non-accelerated calculation uses the analytic Airy pattern;
        the accelerated calculation evaluates the diffraction integral numerically.

        Parameters
        ----------
        suppress_warnings : bool, optional
            Suppress warnings issued by this calculation and the selected kernel routine. The default is False.
        normalized : bool, optional
            Flag to return normalized PSF values to max=1.0. The default is True.
        verbose_info : bool, optional
            Flag to print out each step completion report and measured elapsed time in ms. The default is False.
        accelerated : bool or None, optional
            Flag to accelerate the calculations by using numba library for scripts compilation. Acceleration will be used automatically,
            if None is provided and 'numba' is installed. The default is None.
            Set False for radial orders above 10, which are unsupported by the accelerated kernel routine.

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
            warnings.warn(self.__warn_message, stacklevel=2)
        if not self.__physical_props_set and not suppress_warnings:
            self.kernel_size = 19; self.__warn_message = "\nPhysical properties haven't been set before, the default ones will be used"
            warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
        global numba_installed  # access the flag
        # Check provided flag
        if numba_installed and accelerated is None:
            accelerated = True
        elif not numba_installed and accelerated is None:
            accelerated = False
        # Check if accelerated flag set to True but no numba installed
        if accelerated and not numba_installed and not suppress_warnings:
            self.__warn_message = "\nAcceleration isn't possible because 'numba' library not installed in the current environment"
            warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
        # For providing performance verbose report
        if verbose_info:
            t1 = time.perf_counter()  # for explicit showing of performance
            if self.kernel_size*self.kernel_size >= 301:
                print("Kernel calculation started...")
        # Calculation using the vectorized form
        if self.zernpol is not None:
            if not accelerated or (accelerated and not numba_installed):
                self.kernel = get_psf_kernel(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.expansion_coeff,
                                             wavelength=self.wavelength, NA=self.NA, normalize_values=normalized, airy_pattern=self.airy,
                                             use_vectorized=True, verbose=verbose_info, kernel_size=self.kernel_size,
                                             n_int_r_points=self.n_int_r_points, n_int_phi_points=self.n_int_phi_points,
                                             suppress_warns=suppress_warnings)
            elif numba_installed:
                self.kernel = get_psf_kernel_comp(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.expansion_coeff,
                                                  wavelength=self.wavelength, NA=self.NA, normalize_values=normalized, verbose=verbose_info,
                                                  kernel_size=self.kernel_size, n_int_r_points=self.n_int_r_points,
                                                  suppress_warns=suppress_warnings, n_int_phi_points=self.n_int_phi_points)
        elif len(self.polynomials) > 0:
            self.amplitudes: np.ndarray  # set explicitly expected data type
            if not accelerated or (accelerated and not numba_installed):
                self.kernel = get_psf_kernel_zerns(polynomials=self.polynomials, amplitudes=self.amplitudes, len2pixels=self.pixel_size,
                                                   wavelength=self.wavelength, NA=self.NA, normalize_values=normalized, verbose=verbose_info,
                                                   kernel_size=self.kernel_size, n_int_r_points=self.n_int_r_points,
                                                   n_int_phi_points=self.n_int_phi_points, suppress_warns=suppress_warnings)
            elif numba_installed:
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
        plt.figure(fig_title, figsize=(6, 6)); plt.imshow(self.kernel, cmap=plt.colormaps["viridis"], origin='upper')
        plt.colorbar(); plt.tight_layout()

    # %% Utilities
    def convolute_img(self, image: np.ndarray, scale2original: bool = True) -> np.ndarray:
        """
        Convolute the provided image with PSF kernel as 2D arrays and return the convolved image with the same type as the original one.

        Parameters
        ----------
        image : numpy.ndarray
            Sample gray-scaled (2D array) image.
        scale2original : bool, optional
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
        plt.figure("Sample Image: Disk", figsize=(6, 6)); plt.imshow(target_disk, cmap=plt.colormaps["viridis"], origin='upper')
        plt.axis('off'); plt.tight_layout()
        convolved_img = self.convolute_img(image=target_disk); plt.figure("Convolved PSF and Disk", figsize=(6, 6))
        plt.imshow(convolved_img, cmap=plt.colormaps["viridis"], origin='upper'); plt.axis('off'); plt.tight_layout()

    def crop_kernel(self, min_part_of_max: float = 0.01):
        """
        Crop redundant points from kernel.

        Remove matching outer rows and columns while all four sides lie below the threshold
        min_part_of_max * max(kernel). Stop as soon as any side reaches that threshold.
        Update kernel and kernel_size in place. Kernels of size 3 or smaller are left unchanged.

        Parameters
        ----------
        min_part_of_max : float, optional
            Part of kernel max that is used for checking rows / columns for cropping. The default is 0.01.

        Raises
        ------
        ValueError
            If the kernel size exceeds 3 and min_part_of_max is outside the open interval (0, 0.5).

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
            rows, cols = self.kernel.shape; stop_cropping_loop = False; step_row = 0; step_column = 0
            while not stop_cropping_loop:
                check_upper_row = np.max(self.kernel[0 + step_row, :]) < min_kernel_value
                check_bottom_row = np.max(self.kernel[rows - 1 - step_row, :]) < min_kernel_value
                check_left_column = np.max(self.kernel[:, 0 + step_column]) < min_kernel_value
                check_right_column = np.max(self.kernel[:, cols - 1 - step_column]) < min_kernel_value
                if check_upper_row and check_bottom_row and check_left_column and check_right_column:
                    step_row += 1; step_column += 1
                else:
                    stop_cropping_loop = True
            if step_row > 0 and step_column > 0:
                self.kernel = self.kernel[step_row:rows-step_row, step_column:cols-step_column]
                self.kernel_size = self.kernel.shape[0]

    # %% I/O methods
    def save_json(self, abs_path: Union[str, Path, None] = None, overwrite: bool = False):
        """
        Save class attributes (PSF kernel, physical properties, etc.) in the JSON file for further reloading and avoiding long computation.

        Parameters
        ----------
        abs_path : str | Path, optional
            Path to an existing destination folder. If the folder does not exist, warn and use the default location.
            If None or an empty string, reuse the last saved file's folder; before the first save, use "saved_psfs" in the
            package folder. If saving to the default location fails, warn and fall back to "saved_psfs" in the current
            working directory. Only default folders are created, without their parents. The default is None.
        overwrite : bool, optional
            Flag for overwriting the existing file. The default is False.

        Returns
        -------
        None.

        """
        if abs_path is not None and abs_path != "" and not Path(abs_path).is_dir():
            warnings.warn(f"Destination folder '{abs_path}' does not exist or is not a directory; using the default location.",
                          stacklevel=2)
            abs_path = None
        folders: Tuple[Path, ...]
        if abs_path is None or abs_path == "":
            # json_file_path initially holds the package directory, then the last saved filename
            saved_path = Path(self.json_file_path)
            default_folder = saved_path.joinpath("saved_psfs") if saved_path.is_dir() else saved_path.parent
            fallback_folder = Path.cwd().joinpath("saved_psfs"); use_default = True
            folders = (default_folder, fallback_folder) if default_folder != fallback_folder else (default_folder,)
        else:
            folders = (Path(abs_path),); use_default = False  # handle string and Path destinations identically
        if self.kernel_size == 1:
            self.__warn_message = "\nKernel most likely hasn't been calculated, the kernel size == 1 - default value"
            warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
        for folder_index, folder in enumerate(folders):
            try:
                if use_default:
                    folder.mkdir(exist_ok=True)  # create only the default destination, without creating parent folders
                if self.zernpol is not None:  # single polynomial saving
                    self.json_file_path = save_psf(psf_kernel=self.kernel, NA=self.NA, wavelength=self.wavelength,
                                                   expansion_coefficient=self.expansion_coeff, pixel_size=self.pixel_size,
                                                   kernel_size=self.kernel_size, n_int_points_r=self.n_int_r_points,
                                                   n_int_points_phi=self.n_int_phi_points, zernike_pol=self.zernpol,
                                                   overwrite=overwrite, folder_path=str(folder))
                elif len(self.polynomials) > 0:  # saving several polynomials
                    self.coefficients: np.ndarray   # set explicitly data type for mypy
                    self.json_file_path = save_psf(psf_kernel=self.kernel, NA=self.NA, wavelength=self.wavelength,
                                                   expansion_coefficient=self.coefficients, pixel_size=self.pixel_size,
                                                   kernel_size=self.kernel_size, n_int_points_r=self.n_int_r_points,
                                                   n_int_points_phi=self.n_int_phi_points, zernike_pol=self.polynomials,
                                                   overwrite=overwrite, folder_path=str(folder))
                if use_default:
                    print("Auto assigned folder for saving calculated PSF kernel:", folder, flush=True)
                break
            except OSError as error:
                if not use_default or folder_index == len(folders) - 1:
                    raise  # re-raise OSError for notify the caller if: the folder provided or if not, saving in CWD failed
                # Retry file-writing failures as well as failures to create the package's default folder
                warnings.warn(f"Cannot save PSF in '{folder}': {error}. Falling back to '{folders[folder_index + 1]}'.", stacklevel=2)

    def read_json(self, abs_path: Union[str, Path, None] = None):
        """
        Restore the saved PSF kernel and physical and calculation properties from a JSON file.

        Parameters
        ----------
        abs_path : str | Path, optional
            Absolute or relative path to the file. If None, use json_file_path, which initially holds the package directory
            and is replaced by the saved filename after save_json(). The default is None.

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
            wavelen: float; na: float; a: Union[Real, Sequence[Real]]; pols: list; ps: float; read_props = 0
            saved_kernel_size: int
            for key, item in json_data.items():
                # Calculation properties + calculated kernel
                if key == "PSF Kernel":
                    self.kernel = np.asarray(item); read_props += 1
                elif key == "Kernel Size":
                    saved_kernel_size = item; read_props += 1
                elif key == "# of integration points R":
                    self.n_int_r_points = item; read_props += 1
                elif key == "# of integration points angle":
                    self.n_int_phi_points = item; read_props += 1
                # Read the physical properties and store them in a list for further setting
                elif key == "NA":
                    na = item; read_props += 1
                elif key == "Wavelength":
                    wavelen = item; read_props += 1
                elif key == "Expansion Coefficient" or key == "Amplitudes":
                    a = item; read_props += 1
                elif key == "Pixel Size":
                    ps = item; read_props += 1
                # Getting and reassigning used polynomial(-s)
                elif key == "Polynomial":
                    osa_index = item; self.zernpol = ZernPol(osa=osa_index)
                elif key == "Polynomials":
                    osa_indices = item; pols = []; pols_len = len(osa_indices)
                    if pols_len == 1:
                        self.zernpol = ZernPol(osa=osa_indices[0])
                    else:
                        for index in osa_indices:
                            pols.append(ZernPol(osa=index))
                        self.polynomials = tuple(pols); self.zernpol = None; self.airy = False
            # Assign read physical properties
            if read_props == 8:
                if self.zernpol is not None:
                    # A loaded single polynomial replaces the previous sequence and its coefficient arrays
                    self.polynomials = (); self.coefficients = None; self.amplitudes = None
                    self.airy = self.zernpol.get_mn_orders() == (0, 0)  # only piston uses the single-polynomial Airy calculation
                self.set_physical_props(NA=na, wavelength=wavelen, expansion_coeff=a, pixel_physical_size=ps)
                # Restore the recorded size after physical properties re-estimate it, including for cropped kernels
                self.kernel_size = saved_kernel_size
            else:
                self.__warn_message = "\nProvided json file doesn't contain all necessary keys for physical / calculation properties"
                warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""
        else:
            self.__warn_message = "\nProvided path doesn't contain valid JSON data"
            warnings.warn(self.__warn_message, stacklevel=2); self.__warn_message = ""


# %% Utility functions
def _sanity_check_expansion_coefficient(normalized_coefficient: float, max_coeff_check: bool = False) -> str:
    """
    Check the magnitude of a Zernike phase coefficient in radians.

    Parameters
    ----------
    normalized_coefficient : float
        Phase magnitude, computed as 2*pi*abs(expansion_coeff)/wavelength.
    max_coeff_check : bool, optional
        Whether the value represents the largest absolute coefficient in a sequence. The default is False.

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


def force_get_psf_compilation(verbose_report: bool = False) -> Optional[Tuple[ZernPSF, ZernPSF]]:
    """
    Compile the Numba calculation functions for single-polynomial and multiple-polynomial PSFs.

    Parameters
    ----------
    verbose_report : bool, optional
        Print the elapsed compilation time. The default is False.

    Returns
    -------
    tuple of ZernPSF or None
        The two PSF instances used for compilation, or None if Numba is unavailable.

    """
    if numba_installed:
        if verbose_report:
            print("Precompilation of 'zernpy' started...", flush=True); t1 = time.perf_counter()  # for explicit showing of performance
        # Check the version of numba for guarantee working of compilation calls
        try:
            if int(numba_ver_n[0]) == 0 and int(numba_ver_n[1]) <= 57 and int(numba_ver_n[2]) < 1:
                __warn_message = "\nRecommended numba version for compilation >= 0.57.1"; warnings.warn(__warn_message, stacklevel=2)
        except ValueError:
            __warn_message = "\nCannot parse 'numba' version, expected in format 'x.yy.zz' - all integers"
            warnings.warn(__warn_message, stacklevel=2)
        # Compilation of methods for single polynomial PSF calculation
        NA = 0.2; wavelength = 0.55; pixel_size = wavelength / 0.82; ampl = 0.042
        zp_prc = ZernPol(m=-1, n=1); zpsf_prc = ZernPSF(zp_prc)  # tilt polynomial for compiling the single-polynomial functions
        zpsf_prc.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf_prc.kernel_size = 1  # request the smallest kernel to reduce compilation warm-up time
        zpsf_prc.calculate_psf_kernel(normalized=True, accelerated=True, suppress_warnings=True, verbose_info=False)
        # Compilation of methods for multiple polynomials PSF calculation
        zp1 = ZernPol(m=-2, n=4); zp2 = ZernPol(m=3, n=3); zp3 = ZernPol(m=0, n=4); pols = (zp1, zp2, zp3); coeffs = (-0.01, 0.043, 0.027)
        zpsf_mpc = ZernPSF(pols); zpsf_mpc.set_physical_props(NA=0.2, wavelength=0.5, expansion_coeff=coeffs, pixel_physical_size=0.5/0.85)
        zpsf_mpc.kernel_size = 3  # request a small kernel to reduce compilation warm-up time
        zpsf_mpc.calculate_psf_kernel(normalized=True, accelerated=True, suppress_warnings=True, verbose_info=False)
        if verbose_report:
            passed_time_ms = int(round(1000.0*(time.perf_counter() - t1), 0))
            if passed_time_ms > 1000:
                print(f"Precompilation took: {round(passed_time_ms/1000.0, 2)} sec.", flush=True)
            else:
                print(f"Precompilation took: {passed_time_ms} ms.", flush=True)
            print("--------------------------------------------", flush=True)
        return zpsf_prc, zpsf_mpc
    else:
        __warn_message = "\nAcceleration isn't possible because 'numba' library not installed in the current environment"
        warnings.warn(__warn_message, stacklevel=2)
        return None


def clean_zernpy_cache() -> bool:
    """
    Clean locally cached files created by numba after compilation of computation methods from the 'calc_psfs_numba' module.

    Returns
    -------
    bool
        True if Numba is available and at least six local .nbc or .nbi files are removed without an error.
        This does not clear the in-memory coefficient caches managed by lru_cache.

    """
    _local_cache_cleaned = False  # flag to show that local numba cache found and has been cleaned
    if numba_installed:
        root = Path(__file__).resolve().parent; cache_folder = root.joinpath("calculations").joinpath("__pycache__")
        if cache_folder.exists() and cache_folder.is_dir():
            _n_cleaned = 0
            for obj in cache_folder.iterdir():
                if obj.is_file() and (obj.suffix == ".nbc" or obj.suffix == ".nbi"):
                    try:
                        obj.unlink(missing_ok=True); _n_cleaned += 1
                    except (PermissionError, OSError):
                        _n_cleaned = 0; break  # automatically breaks loop assuming that invalid permission is universal
            if _n_cleaned >= 6:  # report success after at least six cache files are removed without an error
                _local_cache_cleaned = True
    return _local_cache_cleaned


# %% Define default export classes and methods used with import * statement (import * from zernpsf)
__all__ = ['ZernPSF', 'force_get_psf_compilation', 'clean_zernpy_cache']
