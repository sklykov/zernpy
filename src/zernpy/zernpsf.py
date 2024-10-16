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
from math import pi
import time

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calculations.calc_psfs import (get_psf_kernel, lambda_char, um_char, radial_integral_s, radial_integral, get_kernel_size,
                                        convolute_img_psf, get_bumped_circle, save_psf, read_psf)
    from zernikepol import ZernPol
    from utils.intmproc import DispenserManager
else:
    from .calculations.calc_psfs import (get_psf_kernel, lambda_char, um_char, radial_integral_s, radial_integral, get_kernel_size,
                                         convolute_img_psf, get_bumped_circle, save_psf, read_psf)
    from .zernikepol import ZernPol
    from .utils.intmproc import DispenserManager

# %% Module parameters
__docformat__ = "numpydoc"


# %% PSF class
class ZernPSF:
    """
    The PSF (2D) calculated for image (focal) plane of a microscopic system assuming that the phase profile is described
    by the Zernike polynomial. In other words, the PSF describing the image of point source formed by the microscopic system. \n
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
    __physical_props_set: bool = False  # flag for saving that physical properties have been provided
    __warn_message: str = ""; pixel_size_nyquist: float = 0.5*0.5*wavelength  # based on the Abbe limit
    pixel_size = 0.98*pixel_size_nyquist  # default value based on the limit above
    alpha: float = expansion_coeff / wavelength  # the role of amplitude for PSF calculation
    airy: bool = False  # flag if the Piston is provided as the Zernike polynomial
    __ParallelCalc: DispenserManager = None; __integration_params: list = []  # for speeding up the calculations using several Processes
    n_int_r_points: int = 320; n_int_phi_points: int = 300  # integration parameters on the unit radius and angle - polar coordinates
    k: float = 2.0*pi/wavelength  # angular frequency
    json_file_path: str = str(Path(__file__).parent.absolute())  # default path to the saved file with calculated kernel and metadata

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
        ZernPSF() class instance.

        """
        if isinstance(zernpol, ZernPol):
            self.zernpol = zernpol; m, n = self.zernpol.get_mn_orders()
            if m == 0 and n == 0:
                self.airy = True
            else:
                self.airy = False
        else:
            raise ValueError("ZernPSF class required ZernPol class as the input")

    # %% Set properties
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
        # Sanity check of provided wavelength, pixel physical size (Nyquist criteria)
        self.pixel_size_nyquist = 0.5*0.5*wavelength/NA  # based on half of the Abbe resolution limit, see references in the docstring
        self.pixel_size_nyquist_eStr = "{:.3e}".format(self.pixel_size_nyquist)  # formatting calculated pixel size in scientific notation
        if pixel_physical_size <= 0.0:
            raise ValueError("Pixel physical size should be positive real number")
        if pixel_physical_size > self.pixel_size_nyquist:
            raise ValueError(f"Pixel physical size should be less than Nyquist pixel size {self.pixel_size_nyquist_eStr}"
                             + f" computed from the Abbe's resolution limit (0.5*{lambda_char}/NA)")
        self.pixel_size = pixel_physical_size
        # Sanity check for the expansion coefficient of the polynomial
        if abs(expansion_coeff) / wavelength > 10.0:
            self.__warn_message = (f"Expansion coefficient supposed to be less than 10*{lambda_char} - an amplitude of Zernike polynomial"
                                   + "Otherwise, the kernel size should be too big for sufficient PSF representation")
            warnings.warn(self.__warn_message); self.__warn_message = ""
        self.expansion_coeff = expansion_coeff; self.alpha = self.expansion_coeff / self.wavelength
        self.k = 2.0*pi/self.wavelength  # Calculate angular frequency (k)
        if self.kernel_size == 1:  # default value
            self.kernel_size = get_kernel_size(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.alpha,
                                               wavelength=self.wavelength, NA=self.NA)
        self.__physical_props_set = True  # set internal flag True if no ValueError raised

    def set_calculation_props(self, kernel_size: int, n_integration_points_r: int, n_integration_points_phi: int) -> None:
        """
        Set calculation properties: kernel size, number of integration points on polar coordinates. \n

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
            self.__warn_message = "Recommended to set the physical properties for getting estimated kernel size and after set it by this method"
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
    def calculate_psf_kernel(self, suppress_warnings: bool = False, normalized: bool = True, verbose_info: bool = False) -> np.ndarray:
        """
        Calculate PSF kernel using the specified or default calculation parameters and physical values. \n

        Calculation based on the diffraction integral for the circular aperture. The final equation is derived based from the 2 References. \n
        Kernel is defined as the image formed on the sensor (camera) by the diffraction-limited, ideal microscopic system.
        The diffraction integral is calculated numerically on polar coordinates, assuming circular aperture of imaging system (micro-objective). \n
        The order of integration and used equations in short:
        1st - integration going on radius p, using trapezoidal rule: p\*(alpha\*zernike_pol.polynomial_value(p, phi) - r\*p\*np.cos(phi - theta))\*1j \n
        2nd - integration going on angle phi, using Simpson rule, calling the returned integrals by 1st call for each phi and as final output, it
        provides as the np.power(np.abs(integral_sum), 2)\*integral_normalization, there integral_normalization = 1.0/(pi \* pi) - the square of
        the module of the diffraction integral (complex value), i.e. intensity as the PSF value. \n

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

        References
        ----------
        [1] Principles of Optics, by M. Born and E. Wolf, 4 ed., 1968  \n
        [2] https://nijboerzernike.nl/_downloads/Thesis_Nijboer.pdf  \n
        [3] PSF shapes: https://en.wikipedia.org/wiki/Zernike_polynomials#/media/File:ZernikeAiryImage.jpg  \n
        [4] https://opg.optica.org/ao/abstract.cfm?uri=ao-52-10-2062 (DOI: 10.1364/AO.52.002062)  \n

        Returns
        -------
        numpy.ndarray
            PSF kernel.

        """
        if len(self.__warn_message) > 0 and not suppress_warnings:
            warnings.warn(self.__warn_message)
        if not self.__physical_props_set and not suppress_warnings:
            self.__warn_message = "Physical properties for calculation haven't been set, the default values will be used"
            warnings.warn(self.__warn_message); self.__warn_message = ""
        # Calculation using the vectorised form
        self.kernel = get_psf_kernel(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.expansion_coeff, wavelength=self.wavelength,
                                     NA=self.NA, normalize_values=normalized, airy_pattern=self.airy, test_vectorized=True, verbose=verbose_info,
                                     kernel_size=self.kernel_size, n_int_r_points=self.n_int_r_points, n_int_phi_points=self.n_int_phi_points)
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
        if self.airy:
            fig_title = f"Airy pattern with {round(self.expansion_coeff, 2)} expansion coeff. {id_str}"
        else:
            fig_title = (f"{self.zernpol.get_mn_orders()} {self.zernpol.get_polynomial_name(True)}: "
                         + f"{round(self.expansion_coeff, 2)} expansion coeff. {id_str}")
        plt.figure(fig_title, figsize=(6, 6)); plt.imshow(self.kernel, cmap=plt.cm.viridis, origin='upper')
        plt.colorbar(); plt.tight_layout()

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
        plt.ion(); plt.figure("Sample Image: Disk", figsize=(6, 6)); plt.imshow(target_disk, cmap=plt.cm.viridis, origin='upper')
        plt.axis('off'); plt.tight_layout()
        convolved_img = self.convolute_img(image=target_disk); plt.figure("Convolved PSF and Disk", figsize=(6, 6))
        plt.imshow(convolved_img, cmap=plt.cm.viridis, origin='upper'); plt.axis('off'); plt.tight_layout()

    def save_json(self, abs_path: str | Path = None, overwrite: bool = False):
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
            abs_path = Path(self.json_file_path).joinpath("saved_psfs")  # default folder for storing the saved JSON files (root for the package)
        if not Path.exists(abs_path):
            abs_path = ""  # the folder "saved_psfs" will be created in the root of the package
        if isinstance(abs_path, Path):
            abs_path = str(abs_path)  # convert to the expected string format
        if self.kernel_size == 1:
            self.__warn_message = "Kernel most likely hasn't been calculated, the kernel size == 1 - default value"
            warnings.warn(self.__warn_message); self.__warn_message = ""
        self.json_file_path = save_psf(psf_kernel=self.kernel, NA=self.NA, wavelength=self.wavelength, expansion_coefficient=self.expansion_coeff,
                                       pixel_size=self.pixel_size, kernel_size=self.kernel_size, n_int_points_r=self.n_int_r_points,
                                       n_int_points_phi=self.n_int_phi_points, zernike_pol=self.zernpol, overwrite=overwrite, folder_path=abs_path)

    def read_json(self, abs_path: str | Path = None):
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
        json_data = read_psf(abs_path)
        if json_data is not None:
            l: float; na: float; a: float; ps: float; read_props = 0
            for key, item in json_data.items():
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
                    l = item; read_props += 1
                elif key == "Expansion Coefficient":
                    a = item; read_props += 1
                elif key == "Pixel Size":
                    ps = item; read_props += 1
            if read_props == 8:
                self.set_physical_props(NA=na, wavelength=l, expansion_coeff=a, pixel_physical_size=ps)
            else:
                self.__warn_message = "Provided json file doesn't contain all necessary keys"
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
        if self.kernel_size < 3:
            self.kernel_size = get_kernel_size(zernike_pol=self.zernpol, len2pixels=self.pixel_size, alpha=self.alpha, wavelength=self.wavelength)
        self.kernel = np.zeros(shape=(self.kernel_size, self.kernel_size)); i_center = self.kernel_size//2; j_center = self.kernel_size//2
        calculated_points = 0
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                t1 = time.perf_counter(); pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))  # in pixels
                # Convert pixel distance in the required k*NA*pixel_dist*calibration coefficient
                distance = self.k*self.NA*self.pixel_size*pixel_dist  # conversion from pixel distance into phase multiplier
                theta = np.arctan2((i - i_center), (j - j_center))  # The PSF also has the angular dependency, not only the radial one
                theta += np.pi  # shift angles to the range [0, 2pi]
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


# %% Test as the main script
if __name__ == "__main__":
    plt.close("all")  # close all opened before figures
    # zpsf1 = ZernPSF(ZernPol(m=0, n=0)); kernel1 = zpsf1.calculate_psf_kernel(suppress_warns=True); zpsf1.plot_kernel()  # Airy
    check_other_pols = False; check_small_na_wl = False  # flag for checking some other polynomials PSFs
    check_airy = False; check_common_psf = False; check_faster_airy = False
    check_test_conditions = False; check_test_conditions2 = True

    # Common PSF for testing
    if check_common_psf:
        zpsf2 = ZernPSF(ZernPol(m=-3, n=3)); wavelength_um = 0.55; ampl=-0.25
        # ampl = 0.85  # for saving the plot of the kernel
        zpsf2.set_physical_props(NA=0.95, wavelength=wavelength_um, expansion_coeff=ampl, pixel_physical_size=wavelength_um/5.05)
        zpsf2.set_calculation_props(kernel_size=zpsf2.kernel_size, n_integration_points_r=250, n_integration_points_phi=320)
        kernel3 = zpsf2.calculate_psf_kernel(suppress_warnings=False, verbose_info=False); zpsf2.plot_kernel("for loop")
    # zpsf2.initialize_parallel_workers(); kernel2 = zpsf2.get_kernel_parallel(); zpsf2.plot_kernel("parallel"); zpsf2.deinitialize_workers()

    # Another Zernike polynomial, big NA
    if check_other_pols:
        zpsf3 = ZernPSF(ZernPol(m=0, n=4))
        zpsf3.set_physical_props(NA=1.25, wavelength=wavelength_um, expansion_coeff=0.4, pixel_physical_size=wavelength_um/5.25)
        zpsf3.calculate_psf_kernel(suppress_warnings=False, verbose_info=False); zpsf3.plot_kernel()

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
        zpsf6.set_calculation_props(kernel_size=zpsf6.kernel_size, n_integration_points_r=250, n_integration_points_phi=300)  # normal assignment
        zpsf6.calculate_psf_kernel(normalized=True); zpsf6.plot_kernel()

    if check_test_conditions2:
        zp7 = ZernPol(m=1, n=3); zpsf7 = ZernPSF(zp7)  # horizontal coma
        NA = 0.4; wavelength = 0.4; pixel_size = wavelength / 3.2; ampl = 0.185  # Common physical properties
        zpsf7.set_physical_props(NA=NA, wavelength=wavelength, expansion_coeff=ampl, pixel_physical_size=pixel_size)
        zpsf7.calculate_psf_kernel(normalized=True); zpsf7.plot_kernel()

    # Utilities tests (visualize, save, read)
    zpsf7.visualize_convolution()  # Visualize convolution on the disk image
    # default_path = Path(__file__).parent.joinpath("saved_psfs").absolute()  # default path for storing JSON files
    # zpsf2.save_json(abs_path=default_path, overwrite=True)  # save calculated PSF kernel along with metadata
    # zpsf2.save_json(overwrite=True)  # save calculated PSF kernel along with metadata in the standard location
    # standard_path = Path.home().joinpath("Desktop")  # saving json on the Desktop
    # zpsf2.save_json(overwrite=True, abs_path=standard_path)
    # saved_test_file_path = default_path.joinpath("psf_(-3, 3)_-0.25.json")
    # zpsf2.read_json()  # for testing reading
