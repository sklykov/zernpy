# -*- coding: utf-8 -*-
"""
Main script with the class definition for accessing Zernike polynomial initialization, calculation and plotting.

Also, provides a few functions useful for fitting set of Zernike polynomials to an image with phases.

@author: Sergei Klykov, @year: 2023 \n
@licence: MIT \n

"""
# %% Global imports
import numpy as np
from pathlib import Path
import warnings
import math
from collections import namedtuple
import matplotlib.pyplot as plt
import random

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calculations.calc_zernike_pol import (normalization_factor, radial_polynomial, triangular_function,
                                               triangular_derivative, radial_derivative,
                                               radial_polynomial_eq, radial_derivative_eq,
                                               radial_polynomial_coeffs, radial_polynomial_coeffs_dr)
    from plotting.plot_zerns import plot_sum_fig, subplot_sum_on_fig
    from calculations.fit_zernike_pols import crop_phases_img, fit_zernikes
    from props.properties import polynomial_names, short_polynomial_names
else:
    from .calculations.calc_zernike_pol import (normalization_factor, radial_polynomial, triangular_function,
                                                triangular_derivative, radial_derivative,
                                                radial_polynomial_eq, radial_derivative_eq,
                                                radial_polynomial_coeffs, radial_polynomial_coeffs_dr)
    from .plotting.plot_zerns import plot_sum_fig, subplot_sum_on_fig
    from .calculations.fit_zernike_pols import crop_phases_img, fit_zernikes
    from .props.properties import polynomial_names, short_polynomial_names

# %% Module parameters
__docformat__ = "numpydoc"
polar_vectors = namedtuple("PolarVectors", "R Theta")  # re-used below as the return type
zernikes_surface = namedtuple("ZernikesSurface", "ZernSurf R Theta")  # used as input type


# %% Class def.
class ZernPol:
    """Define the Zernike polynomial class and associated calculation methods."""

    # Pre-initialized class variables
    __initialized: bool = False  # will be set to true after successful construction
    __n: int = 0; __m: int = 0; __osa_index: int = 0; __noll_index: int = 0; __fringe_index: int = 0

    def __init__(self, **kwargs):
        """
        Initialize of the class for Zernike polynomial definition as the object.

        Parameters
        ----------
        **kwargs : orders or index for Zernike polynomial initialization. \n
            Acceptable variants for key word arguments: \n
            1) n=int, m=int with alternatives: "radial_order" for n; "l", "azimuthal_order", "angular_frequency" for m; \n
            2) osa_index=int with alternatives: "osa", "ansi_index", "ansi"; \n
            3) noll_index=int with alternative "noll"; \n
            4) fringe_index=int with alternative "fringe" \n

        Raises
        ------
        ValueError
            Raised if the specified orders (m, n) or index (OSA, Noll...) have some inconsistencies.

        References
        ----------
        [1] Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials \n
        [2] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013) \n
        [3] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011) \n

        Returns
        -------
        ZernPol class' instance.

        """
        key = ""
        # Zernike polynomial specified with key arguments m, n - check firstly for these parameters
        if len(kwargs.keys()) == 2:
            if "n" in kwargs.keys() or "radial_order" in kwargs.keys():
                if self.__initialized:
                    raise ValueError("The polynomial has been already initialized, but radial_order/n/m parsed")
                else:
                    # get the actual name of the key for radial order
                    if "n" in kwargs.keys():
                        key = "n"
                    else:
                        key = "radial_order"
                    if isinstance(kwargs.get(key), int):
                        self.__n = kwargs.get(key)  # radial order acknowledged
                        # Below each time key arguments are checked in the list of keys
                        if ("m" in kwargs.keys() or "l" in kwargs.keys() or "azimuthal_order" in kwargs.keys()
                           or "angular_frequency" in kwargs.keys()):
                            if "m" in kwargs.keys():
                                key = "m"
                            elif "l" in kwargs.keys():
                                key = "l"
                            elif "azimuthal_order" in kwargs.keys():
                                key = "azimuthal_order"
                            else:
                                key = "angular_frequency"
                            if isinstance(kwargs.get(key), int):
                                self.__m = kwargs.get(key)  # azimuthal order acknowledged
                                # Checking that the provided orders are reasonable
                                if not (self.__n - abs(self.__m)) % 2 == 0:  # see [1]
                                    raise ValueError("Failed sanity check: n - |m| == even number")
                                elif self.__n < 0:
                                    raise ValueError("Failed sanity check: order n less than 0")
                                elif self.__n == 0 and self.__m != 0:
                                    raise ValueError("Failed sanity check: when n == 0, m should be also == 0")
                                # m and n specified correctly, calculate other properties - various indices
                                else:
                                    self.__initialized = True  # set the flag to True, polynomial class initialized
                                    # Calculation of various indices according to [1]
                                    self.__osa_index = ZernPol.get_osa_index(self.__m, self.__n)
                                    self.__noll_index = ZernPol.get_noll_index(self.__m, self.__n)
                                    self.__fringe_index = ZernPol.get_fringe_index(self.__m, self.__n)
                            else:
                                raise ValueError("Azimuthal order m provided not as an integer")
                        else:
                            # the n order defined, but m hasn't been found
                            self.__n = 0
                    else:
                        raise ValueError("Radial order n provided not as an integer")
        elif len(kwargs.keys()) == 1:
            # OSA / ANSI index used for Zernike polynomial initialization
            if ("osa_index" in kwargs.keys() or "osa" in kwargs.keys() or "ansi_index" in kwargs.keys()
               or "ansi" in kwargs.keys()):
                if self.__initialized:
                    raise ValueError("The polynomial has been already initialized, but osa_index/osa... parsed")
                else:
                    if "osa_index" in kwargs.keys():
                        key = "osa_index"
                    elif "osa" in kwargs.keys():
                        key = "osa"
                    elif "ansi_index" in kwargs.keys():
                        key = "ansi_index"
                    elif "ansi" in kwargs.keys():
                        key = "ansi"
                    if isinstance(kwargs.get(key), int):
                        osa_i = kwargs.get(key)
                        if osa_i < 0:
                            raise ValueError("OSA / ANSI index should be non-negative integer")
                        else:
                            self.__osa_index = osa_i; self.__initialized = True
                            self.__m, self.__n = ZernPol.index2orders(osa_index=self.__osa_index)
                            self.__noll_index = ZernPol.get_noll_index(self.__m, self.__n)
                            self.__fringe_index = ZernPol.get_fringe_index(self.__m, self.__n)
                    else:
                        raise ValueError("OSA / ANSI index provided not as an integer")
            # Noll index used for Zernike polynomial initialization
            elif "noll_index" in kwargs.keys() or "noll" in kwargs.keys():
                if self.__initialized:
                    raise ValueError("The polynomial has been already initialized, but noll_index/noll parsed")
                else:
                    if "noll_index" in kwargs.keys():
                        key = "noll_index"
                    elif "noll" in kwargs.keys():
                        key = "noll"
                    if isinstance(kwargs.get(key), int):
                        noll_i = kwargs.get(key)
                        if noll_i < 1:
                            raise ValueError("Noll index should be not less than 1 integer")
                        else:
                            self.__noll_index = noll_i; self.__initialized = True
                            self.__m, self.__n = ZernPol.index2orders(noll_index=self.__noll_index)
                            self.__osa_index = ZernPol.get_osa_index(self.__m, self.__n)
                            self.__fringe_index = ZernPol.get_fringe_index(self.__m, self.__n)
                    else:
                        raise ValueError("Noll index provided not as an integer")
            # Fringe / Univ. of Arizona index used for Zernike polynomial initialization
            elif "fringe_index" in kwargs.keys() or "fringe" in kwargs.keys():
                if self.__initialized:
                    raise ValueError("The polynomial has been already initialized, but fringe_index/fringe parsed")
                else:
                    if "fringe_index" in kwargs.keys():
                        key = "fringe_index"
                    elif "fringe" in kwargs.keys():
                        key = "fringe"
                    if isinstance(kwargs.get(key), int):
                        fringe_i = kwargs.get(key)
                        if fringe_i < 1:
                            raise ValueError("Fringe index should be not less than 1 integer")
                        else:
                            self.__fringe_index = fringe_i; self.__initialized = True
                            self.__m, self.__n = ZernPol.index2orders(fringe_index=self.__fringe_index)
                            self.__osa_index = ZernPol.get_osa_index(self.__m, self.__n)
                            self.__noll_index = ZernPol.get_noll_index(self.__m, self.__n)
                    else:
                        raise ValueError("Fringe index provided not as an integer")
        else:
            raise ValueError("Length of provided key arguments are not equal 1 (OSA/Fringe/Noll index) or 2 (m, n orders)")
        # Also raise the ValueError if the ZernPol hasn't been initialized by orders / indices
        if not self.__initialized:
            raise ValueError("The initialization parameters for Zernike polynomial hasn't been parsed / recognized")

    def get_indices(self):
        """
        Return the tuple with following orders: ((m, n), OSA index, Noll index, Fringe index).

        Returns
        -------
        tuple
            with elements: (tuple (azimuthal (m), radial (n)) orders, OSA index, Noll index, Fringe index) \n
            All indices are integers.
        """
        return (self.__m, self.__n), self.__osa_index, self.__noll_index, self.__fringe_index

    def get_mn_orders(self) -> tuple:
        """
        Return tuple with the (azimuthal, radial) orders, i.e. return (m, n).

        Returns
        -------
        tuple
            with the (azimuthal, radial) orders for the initialized Zernike polynomial.

        """
        return self.__m, self.__n

    def get_polynomial_name(self, short: bool = False) -> str:
        """
        Return string with the conventional name (see references for details) of polynomial up to 7th order.

        Parameters
        ----------
        short : bool, optional
            If True, this method returns shortened name. The default is False.

        References
        ----------
        [1] Up to 4th order: Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials \n
        [2] 5th order names: from the website https://www.telescope-optics.net/monochromatic_eye_aberrations.htm \n
        6th order - 7th order: my guess about the naming \n

        Returns
        -------
        str
            Name of the initialized polynomial.

        """
        name = ""
        if short:
            if (self.__m, self.__n) in short_polynomial_names.keys():
                name = short_polynomial_names[(self.__m, self.__n)]
        else:
            if (self.__m, self.__n) in polynomial_names.keys():
                name = polynomial_names[(self.__m, self.__n)]
        return name

    # %% Calculations
    def polynomial_value(self, r, theta, use_exact_eq: bool = False):
        """
        Calculate Zernike polynomial value(-s) within the unit circle.

        Calculation up to 10th order of Zernike function performed by exact equations from Ref.[2],
        after - using the recurrence equations taken from the Ref.[1]. \n
        Surprisingly, the exact equation for polynomials are just pretty fast to use them directly, without
        any need to use the recurrence equations. It can be used by providing the flag as the input parameter.

        References
        ----------
        [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013) \n
        [2] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011) \n
        [3] Andersen T. B. "Efficient and robust recurrence relations for the Zernike
        circle polynomials and their derivatives in Cartesian coordinates" (2018) \n

        Parameters
        ----------
        r : float or numpy.ndarray
            Radius (radii) from unit circle or the range [0.0, 1.0], float / array, for which the polynomial is calculated.
        theta : float or numpy.ndarray
            Theta - angle in radians from the range [0, 2*pi], float or array, for which the polynomial is calculated.
            Note that the theta counting is counterclockwise, as it is default for the matplotlib library.
        use_exact_eq : bool, optional
            Flag for using the exact equation with factorials. The default is False.
            Surprisingly, but even factorials for high numbers are calculated actually pretty fast, so the radial
            polynomials could be calculated without using any recursion.

        Raises
        ------
        ValueError
            Check the Error stack trace for reasons of raising, most probably input parameters aren't acceptable.
        Warning
            If the theta angles lie outside the range [0, 2*pi] (entire period).

        Returns
        -------
        float or numpy.ndarray
            Calculated polynomial values on provided float values / arrays.

        """
        # Checking input parameters for avoiding errors and unexpectable values
        # Check radii type and that they are not lying outside range [0.0, 1.0] - unit circle
        r = ZernPol._check_radii(r)
        # Checking that angles lie in the range [0, 2*pi] and their type
        theta = ZernPol._check_angles(theta)
        # Checking coincidence of shapes if theta and r are arrays
        if isinstance(r, np.ndarray) and isinstance(theta, np.ndarray):
            if r.shape != theta.shape:
                raise ValueError("Shape of input arrays r and theta is not equal")
        # Calculation using imported function from submodule depending on radial order, use different eq.
        if self.__n <= 15:  # after, the direct recursive eq. becomes ineffective
            return normalization_factor(self)*radial_polynomial(self, r)*triangular_function(self, theta)
        else:
            if not use_exact_eq:
                return normalization_factor(self)*radial_polynomial_coeffs(self, r)*triangular_function(self, theta)
            else:
                return normalization_factor(self)*radial_polynomial_eq(self, r)*triangular_function(self, theta)

    def radial(self, r, use_exact_eq: bool = False):
        """
        Calculate R(m, n) - radial Zernike function value(-s) within the unit circle.

        Calculation up to 10th order of Zernike function performed by exact equations from Ref.[2],
        after - using the recurrence equations taken from the Ref.[1]. \n
        Surprisingly, the exact equation for polynomials are just pretty fast to use them directly, without
        any need to use the recurrence equations. It can be used by providing the flag as the input parameter.

        References
        ----------
        [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013) \n
        [2] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011) \n
        [3] Andersen T. B. "Efficient and robust recurrence relations for the Zernike
        circle polynomials and their derivatives in Cartesian coordinates" (2018) \n

        Parameters
        ----------
        r : float or numpy.ndarray
            Radius (radii) from unit circle or the range [0.0, 1.0], float / array, for which the function is calculated.
        use_exact_eq : bool, optional
            Flag for using the exact equation with factorials. The default is False.
            Surprisingly, but even factorials for high numbers are calculated actually pretty fast, so the radial
            polynomials could be calculated without using any recursion.

        Raises
        ------
        ValueError
            The probable reasons: radius (radii) doesn't belong to a unit circle, input type isn't acceptable.

        Returns
        -------
        float or numpy.ndarray
            Calculated Zernike radial function value(-s) on provided float values / arrays of radiuses.

        """
        # Check radii type and that they are not lying outside range [0.0, 1.0] - unit circle
        r = ZernPol._check_radii(r)
        # Calculation using imported function from submodule depending on radial order, use different eq.
        if self.__n <= 15:  # after, the direct recursive eq. becomes ineffective
            return radial_polynomial(self, r)
        else:
            if not use_exact_eq:
                return radial_polynomial_coeffs(self, r)
            else:
                return radial_polynomial_eq(self, r)

    def triangular(self, theta):
        """
        Calculate triangular Zernike function value(-s) within the unit circle.

        Parameters
        ----------
        theta : float or numpy.ndarray
            Theta - angle in radians from the range [0, 2*pi], float or array, for which the polynomial is calculated.
            Note that the theta counting is counterclockwise, as it is default for the matplotlib library.

        Raises
        ------
        ValueError
            Most probably, raised if the conversion to float number is failed. \n
            It happens when input parameter is not float, numpy.ndarray, list or tuple.
        Warning
            If the theta angles lie outside the range [0, 2*pi] (entire period).

        Returns
        -------
        float or numpy.ndarray
            Calculated value(-s) of Zernike triangular function on provided angle.

        """
        # Check theta type and that angles are lying in the single period range [0, 2pi]
        theta = ZernPol._check_angles(theta)
        # Calculation using imported function
        return triangular_function(self, theta)

    def radial_dr(self, r, use_exact_eq: bool = False):
        """
        Calculate derivative of radial Zernike polynomial value(-s) within the unit circle.

        Calculation up to 10th order of Zernike polynomials performed by exact equations,
        after - using the recurrence equations.

        References
        ----------
        Same as for the method "radial" or "polynomial value"

        Parameters
        ----------
        r : float or numpy.ndarray
            Radius (radii) from unit circle or the range [0.0, 1.0], float / array, for which the polynomial is calculated.
        use_exact_eq : bool, optional
            Flag for using the exact equation with factorials. The default is False.
            Surprisingly, but even factorials for high numbers are calculated actually pretty fast, so the radial
            polynomials could be calculated without using any recursion.

        Raises
        ------
        ValueError
            The probable reasons: radius (radii) doesn't belong to a unit circle, input type isn't acceptable.

        Returns
        -------
        float or numpy.ndarray
            Calculated derivative of Zernike radial function value(-s) on provided float values / arrays of radiuses.

        """
        # Checking input parameters for avoiding errors and unexpectable values
        r = ZernPol._check_radii(r)
        # Calculation using imported function from submodule depending on radial order, use different eq.
        if self.__n <= 15:  # after, the direct recursive eq. becomes ineffective
            return radial_derivative(self, r)
        else:
            if not use_exact_eq:
                return radial_polynomial_coeffs_dr(self, r)
            else:
                return radial_derivative_eq(self, r)

    def triangular_dtheta(self, theta):
        """
        Calculate derivative from triangular function on angle theta.

        Parameters
        ----------
        theta : float or numpy.ndarray
            Theta - angle in radians from the range [0, 2*pi], float or array, for which the function is calculated.
            Note that the theta counting is counterclockwise, as it is default for the matplotlib library.

        Raises
        ------
        ValueError
            Most probably, raised if the conversion to float number is failed. \n
            It happens when input parameter is not float, numpy.ndarray, list or tuple.
        Warning
            If the theta angles lie outside the range [0, 2*pi] (entire period).

        Returns
        -------
        float or numpy.ndarray
            Calculated derivative value(-s) of Zernike triangular function on provided angle.

        """
        # Check input parameter type and attempt to convert to acceptable types
        theta = ZernPol._check_angles(theta)
        return triangular_derivative(self, theta)

    def normf(self):
        """
        Calculate normalization factor for the Zernike polynomial calculated according to the Reference below.

        References
        ----------
        [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)

        Returns
        -------
        float
            Normalization factor calculated according to the Reference.

        """
        return normalization_factor(self)

    # %% Static methods
    @staticmethod
    def get_osa_index(m: int, n: int) -> int:
        """
        Calculate OSA / ANSI index from the 2 orders of Zernike polynomials.

        Parameters
        ----------
        m : int
            Azimuthal order (angular frequency).
        n : int
            Radial order.

        References
        ----------
        [1] Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials

        Returns
        -------
        int
            OSA index according to [1].

        """
        return (n*(n + 2) + m)//2

    @staticmethod
    def get_noll_index(m: int, n: int) -> int:
        """
        Calculate Noll index from the 2 orders of Zernike polynomials.

        Parameters
        ----------
        m : int
            Azimuthal order (angular frequency).
        n : int
            Radial order.

        References
        ----------
        [1] Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials

        Returns
        -------
        int
            Noll index.

        """
        add_n = 1
        if m > 0:
            if (n % 4) == 0:
                add_n = 0
            elif ((n - 1) % 4) == 0:
                add_n = 0
        elif m < 0:
            if ((n - 2) % 4) == 0:
                add_n = 0
            elif ((n - 3) % 4) == 0:
                add_n = 0
        return (n*(n + 1))//2 + abs(m) + add_n

    @staticmethod
    def get_fringe_index(m: int, n: int) -> int:
        """
        Calculate Fringe / University of Arizona index from the 2 orders of Zernike polynomials.

        Parameters
        ----------
        m : int
            Azimuthal order (angular frequency).
        n : int
            Radial order.

        References
        ----------
        [1] Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials

        Returns
        -------
        int
            Fringe index.

        """
        return (1 + (n + abs(m))//2)**2 - 2*abs(m) + (1 - np.sign(m))//2

    @staticmethod
    def index2orders(**kwargs) -> tuple:
        """
        Return tuple as (azimuthal, radial) orders for the specified by osa_, noll_ or fringe_index input parameter.

        Parameters
        ----------
        **kwargs : dict
            Recognizable values: osa_index, noll_index, fringe_index.

        Returns
        -------
        tuple
            (m, n) - contains azimuthal and radial orders as integers.

        """
        osa_index = -1; noll_index = -1; fringe_index = -1; m = -1; n = -1
        if "osa_index" in kwargs.keys():
            osa_index = kwargs.get("osa_index")
        elif "noll_index" in kwargs.keys():
            noll_index = kwargs.get("noll_index")
        elif "fringe_index" in kwargs.keys():
            fringe_index = kwargs.get("fringe_index")
        # Define m, n orders up to 50th order (assuming it as maximum)
        stop_search = False
        for order in range(0, 51):
            m = -order  # azimuthal order
            n = order  # radial order
            for polynomial in range(0, order+1):
                if osa_index >= 0:
                    if osa_index == ZernPol.get_osa_index(m, n):
                        stop_search = True; break
                elif noll_index >= 0:
                    if noll_index == ZernPol.get_noll_index(m, n):
                        stop_search = True; break
                elif fringe_index >= 0:
                    if fringe_index == ZernPol.get_fringe_index(m, n):
                        stop_search = True; break
                m += 2
            if stop_search:
                break
        return m, n

    @staticmethod
    def osa2noll(osa_index: int) -> int:
        """
        Convert the OSA / ANSI index of a Zernike polynomial to the Noll index.

        Parameters
        ----------
        osa_index : int
            OSA / ANSI index with int type, it must be not less than 0.

        Raises
        ------
        ValueError
            If the index provided with not int type or index < 0.

        Returns
        -------
        int
            Converted Noll index.

        """
        if isinstance(osa_index, int) and osa_index >= 0:
            m, n = ZernPol.index2orders(osa_index=osa_index)
            return ZernPol.get_noll_index(m, n)
        else:
            raise ValueError(f"Provided {osa_index} isn't integer or less than 0")

    @staticmethod
    def noll2osa(noll_index: int) -> int:
        """
        Convert the Noll index of a Zernike polynomial to the OSA / ANSI index.

        Parameters
        ----------
        noll_index : int
            The Noll index with int type, it must be not less than 1.

        Raises
        ------
        ValueError
            If the index provided with not int type or index < 1.

        Returns
        -------
        int
            Converted OSA / ANSI index.

        """
        if isinstance(noll_index, int) and noll_index >= 1:
            m, n = ZernPol.index2orders(noll_index=noll_index)
            return ZernPol.get_osa_index(m, n)
        else:
            raise ValueError(f"Provided {noll_index} isn't integer or less than 1")

    @staticmethod
    def osa2fringe(osa_index: int) -> int:
        """
        Convert the OSA / ANSI index of a Zernike polynomial to the Fringe index.

        Parameters
        ----------
        osa_index : int
            OSA / ANSI index with int type, it must be not less than 0.

        Raises
        ------
        ValueError
            If the index provided with not int type or index < 0.

        Returns
        -------
        int
            Converted Fringe index.

        """
        if isinstance(osa_index, int) and osa_index >= 0:
            m, n = ZernPol.index2orders(osa_index=osa_index)
            return ZernPol.get_fringe_index(m, n)
        else:
            raise ValueError(f"Provided {osa_index} isn't integer or less than 0")

    @staticmethod
    def fringe2osa(fringe_index: int) -> int:
        """
        Convert the Fringe / Univ. of Arizona index of a Zernike polynomial to the OSA / ANSI index.

        Parameters
        ----------
        fringe_index : int
            The noll index with int type, it must be not less than 1.

        Raises
        ------
        ValueError
            If the index provided with not int type or index < 1.

        Returns
        -------
        int
            Converted OSA / ANSI index.

        """
        if isinstance(fringe_index, int) and fringe_index >= 1:
            m, n = ZernPol.index2orders(fringe_index=fringe_index)
            return ZernPol.get_osa_index(m, n)
        else:
            raise ValueError(f"Provided {fringe_index} isn't integer or less than 1")

    @staticmethod
    def sum_zernikes(coefficients: list, polynomials: list, r, theta, get_surface: bool = False):
        """
        Calculate sum of Zernike polynomials along with their coefficients (e.g., for plotting over a unit circle).

        Parameters
        ----------
        coefficients : list
            Coefficients of Zernike polynomials for summing.
        polynomials : list
            Initialized polynomials as class instances of ZernPol class specified in this module.
        r : float or numpy.ndarray
            Radius(-s) from a unit circle.
        theta : float or numpy.ndarray
            Polar angle(-s) from a unit circle.
        get_surface : bool, optional
            If True, it forces to calculate 2D sum of polynomials based on r and theta (as a mesh). The default is False.

        Raises
        ------
        TypeError
            If the input parameters aren't iterable (doesn't support len() function), this error will be raised.
        ValueError
            If the lengths of lists (tuples, numpy.ndarrays) aren't equal for coefficients and polynomials. \n
            Or if the list (tuple, numpy.ndarray vector, ...) with Zernike polynomials instances (ZernPol()).

        Returns
        -------
        Sum of Zernike polynomials
            Depending on the input values and parameter get_surface - can be: float, 1D or 2D numpy.ndarrays.

        """
        S = 0.0  # default value - sum
        if len(coefficients) != len(polynomials):
            raise ValueError("Lengths of coefficients and polynomials aren't equal")
        else:
            if not get_surface or not isinstance(r, np.ndarray) or not isinstance(theta, np.ndarray):
                for i, coefficient in enumerate(coefficients):
                    if not isinstance(polynomials[i], ZernPol):
                        raise ValueError(f"Variable {polynomials[i]} isn't an instance of ZernPol class")
                    if i == 0:
                        S = coefficient*polynomials[i].polynomial_value(r, theta)
                    else:
                        S += coefficient*polynomials[i].polynomial_value(r, theta)
            elif get_surface:
                if not isinstance(r, np.ndarray) or not isinstance(theta, np.ndarray):
                    warnings.warn("Requested calculation of surface (mesh) values with"
                                  + " provided r or theta as not numpy.ndarray.\n"
                                  + "The surface will be generated automatically.")
                else:
                    r_size = np.size(r, 0); theta_size = np.size(theta, 0)
                    S = np.zeros(shape=(r_size, theta_size))
                    for i, coefficient in enumerate(coefficients):
                        if not isinstance(polynomials[i], ZernPol):
                            raise ValueError(f"Variable {polynomials[i]} isn't an instance of ZernPol class")
                        if theta_size > r_size:
                            for j in range(r_size):
                                S[j, :] += coefficient*polynomials[i].polynomial_value(r[j], theta)[:]
                        else:
                            for j in range(theta_size):
                                S[:, j] += coefficient*polynomials[i].polynomial_value(r, theta[j])[:]
        return S

    @staticmethod
    def gen_polar_coordinates(r_step: float = 0.01, theta_rad_step: float = (np.pi/180)) -> polar_vectors:
        """
        Generate named tuple "PolarVectors" with R and Theta - vectors with polar coordinates for an entire unit circle.

        Note that R and Theta are generated as the numpy.ndarrays vectors (shape like (n elements, )). Their shapes are
        defined by the specification of r_step and theta_rad_step parameters.

        Parameters
        ----------
        r_step : float, optional
            Step for generation the vector with radiuses for an entire unit circle. The default is 0.01.
        theta_rad_step : float, optional
            Step for generation the vector with theta angles for an entire unit circle. The default is (np.pi/90).

        Raises
        ------
        ValueError
            If the r_step or theta_rad_step provided in the way, that vectors R and Theta cannot be generated.

        Returns
        -------
        polar_vectors
            namedtuple("PolarVectors", "R Theta"), where R - vector with radiuses values [0.0, r_step, ... 1.0],
            Theta - vector with theta angles values [0.0, theta_rad_step, ... 2*pi].

        """
        if 0.0 >= r_step > 0.5:
            raise ValueError("Provided step on radiuses less than 0.0 or more than 0.5")
        if 0.0 >= theta_rad_step > np.pi:
            raise ValueError("Provided step on theta angles less than 0.0 or more than pi")
        R = np.arange(0.0, 1.0+r_step, r_step); Theta = np.arange(0.0, 2*np.pi+theta_rad_step, theta_rad_step)
        return polar_vectors(R, Theta)

    @staticmethod
    def plot_zernike_polynomial(polynomial, color_map: str = "coolwarm", show_title: bool = True):
        """
        Plot the provided Zernike polynomial (instance of ZernPol class) on the matplotlib figure.

        Note that the plotting function plt.show() creates the plot in non-interactive mode.

        Parameters
        ----------
        polynomial : ZernPol
            Instance of ZernPol class.
        color_map : str, optional
            Color map of the polar plot, recommended values: coolwarm, jet, turbo, rainbow. The default is "coolwarm".
        show_title : bool, optional
            Toggle for showing the name of polynomial on the plot or not.

        Returns
        -------
        None.

        """
        if isinstance(polynomial, ZernPol):
            r, theta = ZernPol.gen_polar_coordinates()
            amplitudes = [1.0]; zernikes = [polynomial]  # for reusing the sum function of polynomials
            zern_surface = ZernPol.sum_zernikes(amplitudes, zernikes, r, theta, get_surface=True)
            if show_title:
                plot_sum_fig(zern_surface, r, theta, title=polynomial.get_polynomial_name(),
                             color_map=color_map)
            else:
                plot_sum_fig(zern_surface, r, theta, "", color_map)

    @staticmethod
    def gen_zernikes_surface(coefficients: list, polynomials: list, r_step: float = 0.01,
                             theta_rad_step: float = (np.pi/180)) -> zernikes_surface:
        """
        Generate surface of provided Zernike polynomials on the generated polar coordinates used steps.

        Parameters
        ----------
        coefficients : list
            Coefficients of Zernike polynomials for summing.
        polynomials : list
            Initialized polynomials as class instances of ZernPol class specified in this module.
        r_step : float, optional
            Step for generation the vector with radiuses for an entire unit circle. The default is 0.01. \n
            See also the documentation for the method gen_polar_coordinates().
        theta_rad_step : float, optional
            Step for generation the vector with theta angles for an entire unit circle. The default is (np.pi/90). \n
            See also the documentation for the method gen_polar_coordinates().

        Returns
        -------
        zernikes_surface
            namedtuple("ZernikesSurface", "ZernSurf R Theta") - tuple for storing mesh values for polar coordinates.
            ZernSurf variable is 2D matrix with the sum of the input polynomials on generated polar coordinates (R, Theta).

        """
        polar_vectors = ZernPol.gen_polar_coordinates(r_step, theta_rad_step)
        zernikes_sum = ZernPol.sum_zernikes(coefficients, polynomials, polar_vectors.R,
                                            polar_vectors.Theta, get_surface=True)
        return zernikes_surface(zernikes_sum, polar_vectors.R, polar_vectors.Theta)

    @staticmethod
    def plot_sum_zernikes_on_fig(coefficients: list, polynomials: list, figure: plt.Figure,
                                 use_defaults: bool = True, zernikes_sum_surface: zernikes_surface = (),
                                 show_range: bool = True, color_map: str = "coolwarm") -> plt.Figure:
        """
        Plot a sum of the specified Zernike polynomials by input lists (see function parameters) on the provided figure.

        Note that for showing the plotted figure, one needs to call appropriate functions (e.g., matplotlib.pyplot.show()
        or figure.show()) as the method of the input parameter figure.

        Parameters
        ----------
        coefficients : list
            Coefficients of Zernike polynomials for summing.
        polynomials : list
            Initialized polynomials as class instances of ZernPol class specified in this module.
        figure : plt.Figure
            Figure() class there the plotting will be done, previous plot will be cleared.
        use_defaults : bool, optional
            Use for plotting default values for generation of a mesh of polar coordinates and calculation
            of Zernike polynomials sum. The default is True.
        zernikes_sum_surface : namedtuple("ZernikesSurface", "ZernSurf R Theta") , optional
            This tuple should contain the ZernSurf calculated on a mesh of polar coordinates R, Theta.
            This tuple could be generated by the call of the static method gen_zernikes_surface().
            Check the method signature for details. The default is ().
        show_range : bool, optional
            Flag for showing range of provided values as the colorbar on the figure. The default is True.
        color_map : str, optional
            Color map of the polar plot, recommended values: coolwarm, jet, turbo, rainbow. The default is "coolwarm".

        Raises
        ------
        ValueError
            Check the signature for details. In general, it will be raised if some input parameters are inconsistent.

        Returns
        -------
        figure : plt.Figure
            Matplotlib.pyplot Figure class there the Zernike polynomials sum plotted.

        """
        if len(coefficients) == 0:
            raise ValueError("Input list with coefficients is empty")
        if not use_defaults and len(zernikes_sum_surface) != 3:
            raise ValueError("Zernike surface isn't specified as tuple with values Sum surface, R, Theta")
        if use_defaults:
            polar_vectors = ZernPol.gen_polar_coordinates()
            zernikes_sum = ZernPol.sum_zernikes(coefficients, polynomials, polar_vectors.R, polar_vectors.Theta,
                                                get_surface=True)
            figure = subplot_sum_on_fig(figure, zernikes_sum, polar_vectors.R, polar_vectors.Theta,
                                        show_range_colorbar=show_range, color_map=color_map)
        else:
            figure = subplot_sum_on_fig(figure, zernikes_sum_surface.ZernSurf, zernikes_sum_surface.R,
                                        zernikes_sum_surface.Theta, show_range_colorbar=show_range,
                                        color_map=color_map)
        return figure

    @staticmethod
    def _plot_zernikes_half_pyramid():
        """
        Generate halb-pyramid with Zernikes polynomials.

        Returns
        -------
        None.

        """
        n_cols = 7; n_rows = 6; fig = plt.figure(figsize=(10, 7.8))
        axes = fig.subplots(n_rows, n_cols, subplot_kw=dict(projection='polar'), squeeze=False)
        zps = []; ampls = [1.0]; k = 1  # k for OSA indexing
        ignored_column = n_cols - 2
        for i in range(len(axes)):
            for j in range(len(axes[0])):
                axes[i, j].grid(False)  # demanded by pcolormesh function, if not called - deprecation warning
                if j > ignored_column-1:
                    zps = [ZernPol(osa=k)]
                    zernike_surface, r, theta = ZernPol.gen_zernikes_surface(coefficients=ampls, polynomials=zps)
                    axes[i, j].pcolormesh(theta, r, zernike_surface, cmap=plt.cm.coolwarm, shading='nearest')
                    k += 1
                axes[i, j].axis('off')  # off polar coordinate axes
            ignored_column -= 1
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.tight_layout()

    @staticmethod
    def _check_radii(radii):
        """
        Perform check of type of input, attempt to convert to numpy.ndarray or float, check that r is inside [0.0, 1.0].

        Parameters
        ----------
        radii : float or numpy.ndarray
            Radii from the unit circle (range [0.0, 1.0]), float or array, for which the Zernike polynomials are defined.

        Raises
        ------
        ValueError
            The probable reasons: radius (radii) doesn't belong to a unit circle, input type isn't acceptable.

        Returns
        -------
        radii : float or numpy.ndarray
            Return radii, converted, if it's applicable, to float or numpy.ndarray.

        """
        # Trying to convert known (list, tuple) data types into numpy, if they provided as input
        if not isinstance(radii, np.ndarray) and not isinstance(radii, float):
            if isinstance(radii, list) or isinstance(radii, tuple):
                radii = np.asarray(radii)  # convert list or tuple to np.array
            else:
                radii = float(radii)  # attempt to convert r to float number, will raise ValueError if it's impossible
        # Checking that radii or radius lie in the range [0.0, 1.0]
        if isinstance(radii, np.ndarray):
            if np.min(radii) < 0.0 or np.max(radii) > 1.0:
                raise ValueError("Minimal or maximal value of radii laying outside unit circle [0.0, 1.0]")
        elif isinstance(radii, float):
            if 0.0 > radii > 1.0:
                raise ValueError("Radius laying outside unit circle [0.0, 1.0]")
        return radii

    @staticmethod
    def _check_angles(angles):
        """
        Perform check of type of input, attempt to convert to numpy.ndarray or float, check that angles is inside [0, 2pi].

        Parameters
        ----------
        angles : float or numpy.ndarray
            Theta polar coordinates from unit circle there Zernike polynomials are defined.

        Raises
        ------
        ValueError
            Most probably, raised if the conversion to float number is failed. \n
            It happens when input parameter is not float, numpy.ndarray, list or tuple.
        Warning
            If the theta angles lie outside the range [0, 2*pi] (entire period).

        Returns
        -------
        angles : float or numpy.ndarray
            Return angles, converted, if it's applicable, to float or numpy.ndarray.

        """
        # Check input parameter type and attempt to convert to acceptable types
        if not isinstance(angles, np.ndarray) and not isinstance(angles, float):
            if isinstance(angles, list) or isinstance(angles, tuple):
                angles = np.asarray(angles)  # convert list or tuple to np.array
            else:
                angles = float(angles)  # attempt to convert to float number, will raise ValueError if it's impossible
        # Checking that angles lie in the range [0, 2*pi]
        if isinstance(angles, np.ndarray):
            if np.max(angles) - np.min(angles) > 2.0*np.pi:
                _warn_message_ = "Theta angles defined in range outside of interval [0.0, 2.0*pi]"
                warnings.warn(_warn_message_)
            elif np.max(angles) > 2.0*np.pi or np.min(angles) < 0.0:
                _warn_message_ = "Max or min of theta angles lies outside of interval [0.0, 2.0*pi]"
                warnings.warn(_warn_message_)
        elif isinstance(angles, float):
            if angles < 0.0 or angles > 2.0*np.pi:
                _warn_message_ = "Max or min of theta angles lies outside of interval [0.0, 2.0*pi]"
                warnings.warn(_warn_message_)
        return angles


# %% Methods defs.
def generate_polynomials(max_order: int = 10) -> tuple:
    """
    Generate tuple with ZernPol instances (ultimately, representing Zernike polynomials) indexed using OSA scheme.

    Parameters
    ----------
    max_order : int, optional
        Maximum overall radial order (n) for generated Zernike list (pyramid). The default is 10.

    Raises
    ------
    ValueError
        Raised if max_order < 1, because it should be not less than 0.

    Returns
    -------
    tuple
        It composes generated ZernPol instances (Zernike polynomials) ordered using OSA indexing scheme.

    """
    # Sanity check of max_order parameter
    if not isinstance(max_order, int):
        __warning_mess = "The parameter max_order provided not as integer, there will be attempt to convert it to int"
        warnings.warn(__warning_mess)
        max_order = int(max_order)
    if max_order < 0:
        raise ValueError("The maximum order should be not less than 0")
    if max_order > 30:
        __warning_mess = "Calculation polynomial values with orders higher than 30 is really slow"
        warnings.warn(__warning_mess)
    polynomials_list = []
    for order in range(1, max_order + 1):  # going through all specified orders
        m = -order  # azimuthal order
        n = order  # radial order
        for _ in range(order + 1):  # number of polynomials = order + 1
            polynomials_list.append(ZernPol(azimuthal_order=m, radial_order=n))
            m += 2  # according to the specification of Zernike polynomial
    return tuple(polynomials_list)


def generate_random_phases(max_order: int = 4, img_width: int = 513, img_height: int = 513,
                           round_digits: int = 4) -> tuple:
    """
    Generate phases image (profile) for random set of polynomials with randomly selected amplitudes.

    Parameters
    ----------
    max_order : int, optional
        Maximum radial order of generated Zernike polynomials. The default is 4.
    img_width : int, optional
        Width of generated image. The default is 513.
    img_height : int, optional
        Height of generated image. The default is 513.
    round_digits : int, optional
        Round digits for polynomials amplitudes generation (numpy.round(...) function call). The default is 4.

    Returns
    -------
    tuple
        Consisting of:
        2D phase profile image with provided width and height;
        1D numpy.ndarray containing randomly selected polynomials amplitudes;
        tuple with generated Zernike polynomials.

    """
    # Generate list with radial and angular orders for identifying polynomials
    polynomials_list = list(generate_polynomials(max_order))
    # Generate list with non-zero polynomials
    if max_order < 3:
        selector_list = [False, False, True]
    elif 3 <= max_order <= 4:
        selector_list = [False, False, False, True]
    elif 4 < max_order <= 8:
        selector_list = [False, False, False, False, False, True]
    else:
        selector_list = [False, False, False, False, False, False, False, True]
    polynomials_amplitudes = np.zeros(shape=(len(polynomials_list, )))
    for i in range(polynomials_amplitudes.shape[0]):
        if random.choice(selector_list):
            polynomials_amplitudes[i] = random.uniform(-1.5, 1.5)
    polynomials_amplitudes = np.round(polynomials_amplitudes, round_digits)
    # Additional check that at least some amplitude is non-zero and sufficiently high
    if np.max(np.abs(polynomials_amplitudes)) < 0.08:
        list_indices = [i for i in range(polynomials_amplitudes.shape[0])]
        index = random.choice(list_indices)
        rand_ampl = random.uniform(-1.5, 1.5)
        if abs(rand_ampl) < 0.08:  # if selected amplitude again is low, shift choice borders
            polynomials_amplitudes[index] = random.uniform(0.25, 1.5)
        else:
            polynomials_amplitudes[index] = rand_ampl
    # Generate some phases image - sum of polynomials on some 2D array of pixels converted to polar coordinates
    phases_image = np.zeros(shape=(img_height, img_width))  # blank image
    row_center = img_height // 2; cols_center = img_width // 2
    # Below - define radius of the Zernike circular profile, it is +1 for including more pixels in profile
    min_img_size = min(img_width, img_height); img_radius = 1 + min_img_size // 2
    center = np.asarray([row_center, cols_center])  # center point of an image
    # Calculation of 2D image with phases, which actually are composed by the sum of randomly selected Zernike polynomials
    for i in range(img_height):
        r = np.zeros(shape=(img_width, )); theta = np.zeros(shape=(img_width, ))
        for j in range(img_width):
            euclidean_dist = np.linalg.norm(center - np.asarray([i, j]))
            if euclidean_dist <= img_radius:
                r[j] = euclidean_dist / img_radius
                theta[j] = np.arctan2(row_center - i, j - cols_center)
                if theta[j] < 0.0:
                    theta[j] += 2.0*np.pi
        # Speed up calculations by using vectors (r, theta) as the input parameters
        polynomials_ampl_list = polynomials_amplitudes.tolist()
        phases_image[i, :] = ZernPol.sum_zernikes(polynomials_ampl_list,
                                                  polynomials_list, r, theta)
    # Final conversion
    polynomials_list = tuple(polynomials_list)
    return phases_image, polynomials_amplitudes, polynomials_list


def fit_polynomials(phases_image: np.ndarray, polynomials: tuple, crop_radius: float = 1.0,
                    suppress_warnings: bool = False, strict_circle_border: bool = False,
                    round_digits: int = 4, return_cropped_image: bool = False) -> tuple:
    """
    Fit provided Zernike polynomials (instances of ZernPol class) as the input tuple to the 2D phase image.

    Note that Piston (Z(0, 0) polynomial) is ignored and not fitted, because it represents the constant phase offset
    over a unit aperture (pupil).

    Parameters
    ----------
    phases_image : numpy.ndarray
        2D image with recorded phases which should be approximated by the sum of Zernike polynomials.
    polynomials : tuple
        Initialized tuple with instances of the ZernPol class that effectively represents target set of Zernike polynomials.
    crop_radius : float, optional
        Allow cropping pixel from range [0.5, 1.0], where 1.0 corresponds to radius of circle = min image size.
        The default is 1.0.
    suppress_warnings : bool, optional
        Flag for suppress warnings about the provided 2D image sizes. The default is False.
    strict_circle_border : bool, optional
        Flag for controlling how the border pixels (on the circle radius) are treated: strictly or less strict for
        allowing more pixels to be treated as belonged to a cropped circle. The default is False.
    round_digits : int, optional
        Round digits for returned polynomials amplitudes (numpy.round(...) function call). The default is 4.
    return_cropped_image : bool, optional
        Flag for calculating and returning cropped image, used for fitting procedure. The default is False.

    Returns
    -------
    tuple
        Depending on the input parameter (flag) "return_cropped_image":
        if it is True, then tuple returned with following variables: zernike_coefficients - 1D numpy.ndarray
        with the fitted coefficients of Zernike polynomials, and cropped_image - the cropped part from the
        input image with phases that is used for fitting procedure (useful for debugging purposes);
        if it is False, the following tuple will be returned: zernike_coefficients, None - 1st with the same
        meaning and type as explained before.
    """
    zernike_coefficients = np.zeros(shape=(len(polynomials), ))
    logic_mask, polar_coordinates = crop_phases_img(phases_image, crop_radius, suppress_warnings,
                                                    strict_circle_border)
    if return_cropped_image:
        cropped_image = logic_mask*phases_image  # for debugging
    zernike_coefficients = fit_zernikes(phases_image, logic_mask, polar_coordinates, polynomials)
    zernike_coefficients = np.round(zernike_coefficients, round_digits)
    if return_cropped_image:
        return zernike_coefficients, cropped_image
    else:
        return zernike_coefficients, None


# %% Test functions for the external call
def check_conformity():
    """
    Test initialization parameters and transform between indices consistency.

    Returns
    -------
    None.

    """
    zp = ZernPol(m=-2, n=2)  # Initialization with orders
    (m1, n1), osa_i, noll_i, fringe_i = zp.get_indices()
    assert (osa_i == 3 and noll_i == 5 and fringe_i == 6), (f"Check consistency of Z{(m1, n1)} indices: "
                                                            + f"OSA: {osa_i}, Noll: {noll_i}, Fringe: {fringe_i}")
    zp = ZernPol(l=-3, n=5)
    (m2, n2), osa_i, noll_i, fringe_i = zp.get_indices()
    assert (osa_i == 16 and noll_i == 19 and fringe_i == 20), (f"Check consistency of Z{(m2, n2)} indices: "
                                                               + f"OSA: {osa_i}, Noll: {noll_i}, Fringe: {fringe_i}")
    assert len(zp.get_polynomial_name(short=True)) > 0, f"Short name for Z{(m2, n2)} is zero length"
    zp = ZernPol(azimuthal_order=-1, radial_order=5)
    (m3, n3), osa_i, noll_i, fringe_i = zp.get_indices()
    assert (osa_i == 17 and noll_i == 17 and fringe_i == 15), (f"Check consistency of Z{(m3, n3)} indices: "
                                                               + f"OSA: {osa_i}, Noll: {noll_i}, Fringe: {fringe_i}")
    assert len(zp.get_polynomial_name()) > 0, f"Name for Z{(m3, n3)} is zero length"
    m4, n4 = zp.get_mn_orders()
    assert m4 == m3 and n3 == n4, f"Check method get_mn_orders() for Z{(m3, n3)}"
    print(f"Initialization of polynomials Z{(m1, n1)}, Z{(m2, n2)}, Z{(m3, n3)} tested")
    osa_i = 12; zp = ZernPol(osa_index=osa_i)  # Initialization with OSA index
    m, n = zp.get_mn_orders()
    assert (m == 0 and n == 4), f"Check consistency of Z[OSA index = {osa_i}] orders {m, n}"
    assert zp.get_fringe_index(m, n) == 9, f"Check consistency of Z[OSA index = {osa_i}] Fringe index"
    assert zp.get_noll_index(m, n) == 11, f"Check consistency of Z[OSA index = {osa_i}] Noll index"
    print(f"Initialization of polynomial Z[OSA index = {osa_i}] tested")
    noll_i = 10  # Testing static methods
    assert ZernPol.noll2osa(noll_i) == 9, f"Check consistency of Noll index {noll_i} conversion to OSA index"
    assert ZernPol.osa2fringe(ZernPol.noll2osa(noll_i)) == 10, ("Check consistency of Noll "
                                                                + f"index {noll_i} conversion to OSA index")
    print(f"Conversion of Noll index {noll_i} to OSA and Fringe indices tested")
    # Test for not proper initialization
    try:
        m_f = 2; n_f = -2
        zp = ZernPol(m=m_f, n=n_f)
        asserting_value = False
    except ValueError:
        print(f"Polynomial Z{(m_f, n_f)} haven't been initialized, test passed")
        asserting_value = True
    assert asserting_value, f"Polynomial Z{(m_f, n_f)} initialized with wrong orders assignment"
    # Testing input parameters for calculation
    zp = ZernPol(m=0, n=2); r = 0.0; theta = math.pi
    assert abs(zp.polynomial_value(r, theta) + math.sqrt(3)) < 1E-6, f"Check value of Z[{m}, {n}]({r}, {theta})"
    zp = ZernPol(m=-1, n=1); r = 0.5; theta = math.pi/2
    assert abs(zp.polynomial_value(r, theta) - 1.0) < 1E-6, f"Check value of Z[{m}, {n}]({r}, {theta})"
    print("Simple values of Zernike polynomials tested successfully")
    try:
        r = 'd'; theta = [1, 2]
        zp.polynomial_value(r, theta)
        asserting_value = False
    except ValueError:
        print("Input as string is not allowed for calculation of polynomial value, tested successfully")
        asserting_value = True
    assert asserting_value, "Wrong parameter passed (string) for calculation of polynomial value"
    try:
        r = [0.1, 0.2, 1.0+1E-9]; theta = math.pi
        zp.polynomial_value(r, theta)
        asserting_value = False
    except ValueError:
        print("Radius more than 1.0 is not allowed, tested successfully")
        asserting_value = True
    assert asserting_value, "Wrong parameter passed (r > 1.0) for calculation of polynomial value"
    print("ALL TEST PASSED")


# %% Tests
if __name__ == "__main__":
    _test_plots = False  # regulates plots
    check_conformity()  # testing initialization
    if _test_plots:
        plt.close("all")  # Check plotting functions
        zp = ZernPol(m=-4, n=4); ZernPol.plot_zernike_polynomial(zp, color_map="jet", show_title=False)  # basic plot
        zp = ZernPol(m=-10, n=30); ZernPol.plot_zernike_polynomial(zp, color_map="jet", show_title=False)  # high order plot
        # ZernPol._plot_zernikes_half_pyramid()
        fig3 = plt.figure(figsize=(2, 2)); zp3 = ZernPol(osa=58); polynomials = [zp3]; coefficients = [1.0]
        fig3 = ZernPol.plot_sum_zernikes_on_fig(coefficients, polynomials, fig3, show_range=False, color_map="turbo")
        fig3.subplots_adjust(0, 0, 1, 1)
        # Tests with generation / restoring Zernike profiles (phases images)
        phases_image, polynomials_ampls, polynomials = generate_random_phases(img_height=301, img_width=301)
        plt.figure(); plt.axis("off"); plt.imshow(phases_image, cmap="jet"); plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)
        polynomials_amplitudes, cropped_img = fit_polynomials(phases_image, polynomials, return_cropped_image=True,
                                                              strict_circle_border=False, crop_radius=1.0)
        plt.figure(); plt.axis("off"); plt.imshow(cropped_img, cmap="jet"); plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)
        plt.show()  # show all images created by plt.figure() calls
    # Simple test of two concepts of calculations - exact and recursive equations
    z = ZernPol(n=30, m=-2); print("Diff. between recursive and exact equations:",
                                   round(z.radial(0.85) - z.radial(0.85, use_exact_eq=True), 9))
    z = ZernPol(n=32, l=0); print("Diff. between recursive and exact equations:",
                                  round(z.radial(0.35) - z.radial(0.35, use_exact_eq=True), 9))
    r = -0.955; theta = np.pi/8; z = ZernPol(osa=55)
    print("Diff. between recursive and exact equations:",
          round(z.polynomial_value(r, theta) - z.polynomial_value(r, theta, use_exact_eq=True), 9))
    z = ZernPol(n=35, l=-1); print("Diff. between recursive & exact eq-s for derivatives:",
                                   round(z.radial_dr(0.78) - z.radial_dr(0.78, use_exact_eq=True), 9))
    z = ZernPol(n=38, m=-2); print("Diff. between recursive & exact eq-s for derivatives:",
                                   round(z.radial_dr(0.9) - z.radial_dr(0.9, use_exact_eq=True), 9))
