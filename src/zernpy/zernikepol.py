# -*- coding: utf-8 -*-
"""
Main script with the class definition for accessing Zernike polynomial initialization, calculation and plotting.

Also, provides a few functions useful for fitting set of Zernike polynomials to an image with phases.

@author: Sergei Klykov, @year: 2024, @licence: MIT \n

"""
# %% Global imports
import numpy as np
from pathlib import Path
import warnings
import math
from collections import namedtuple
import matplotlib.pyplot as plt
import random
import time
from typing import Union, Sequence

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calculations.calc_zernike_pol import (normalization_factor, radial_polynomial, triangular_function, triangular_derivative,
                                               radial_derivative, radial_polynomial_eq, radial_derivative_eq, radial_polynomial_coeffs,
                                               radial_polynomial_coeffs_dr, MAX_RADIAL_ORDER_COEFFS, MAX_RADIAL_ORDER_COEFFS_dR)
    from plotting.plot_zerns import plot_sum_fig, subplot_sum_on_fig, plot_sum_fig_3d, subplot_sum_on_fig_3d
    from calculations.fit_zernike_pols import crop_phases_img, fit_zernikes
    from props.properties import polynomial_names, short_polynomial_names, warn_mess_r_long, warn_mess_dr_long, warn_mess_slow_calc
else:
    from .calculations.calc_zernike_pol import (normalization_factor, radial_polynomial, triangular_function, triangular_derivative,
                                                radial_derivative, radial_polynomial_eq, radial_derivative_eq, radial_polynomial_coeffs,
                                                radial_polynomial_coeffs_dr, MAX_RADIAL_ORDER_COEFFS, MAX_RADIAL_ORDER_COEFFS_dR)
    from .plotting.plot_zerns import plot_sum_fig, subplot_sum_on_fig, plot_sum_fig_3d, subplot_sum_on_fig_3d
    from .calculations.fit_zernike_pols import crop_phases_img, fit_zernikes
    from .props.properties import polynomial_names, short_polynomial_names, warn_mess_r_long, warn_mess_dr_long, warn_mess_slow_calc

# %% Module parameters
__docformat__ = "numpydoc"
polar_vectors = namedtuple("PolarVectors", "R Theta")  # re-used below as the return type from the method
zernikes_surface = namedtuple("ZernikesSurface", "ZernSurf R Theta")  # used as the input type


# %% Zernike Pol. class
class ZernPol:
    """Define the Zernike polynomial class and associated calculation methods."""

    # Pre-initialized class variables
    __initialized: bool = False  # will be set to true after successful construction
    __n: int = 0; __m: int = 0; __osa_index: int = 0; __noll_index: int = 0; __fringe_index: int = 0
    __show_slow_calc_warn: bool = False  # internal flag for showing the warning about slow performant calculations

    # %% Initialization and getters for the common class properties
    def __init__(self, **kwargs):
        """
        Initialize of the class for Zernike polynomial definition as the object.

        Parameters
        ----------
        **kwargs : orders or index for Zernike polynomial initialization expected as key=value pairs \n
            Acceptable variants for key=value pairs arguments: \n
            1) n=int, m=int with alternatives: "radial_order" for n; "l", "azimuthal_order", "angular_frequency" for m; \n
            2) osa_index=int with alternatives: "osa", "ansi_index", "ansi"; \n
            3) noll_index=int with alternative "noll"; \n
            4) fringe_index=int with alternative "fringe" \n

        Raises
        ------
        ValueError
            Raised if the specified orders (m, n) or index (OSA, Noll...) have inconsistencies like: \n
            for specified orders:
            1) m < 0 or n < 0; 2) m or n is not int; 3) (self.__n - abs(self.__m)) % 2 == 0; \n
            4) if n > 54 - because the polynomials with higher orders are meaningless due to very slow calculations; \n
            for indices - see the raised error message.

        References
        ----------
        [1] Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials \n
        [2] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013) \n
        [3] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011) \n

        Returns
        -------
        ZernPol class instance.

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
                                elif self.__n < 0:
                                    raise ValueError("Failed sanity check: order n less than 0")
                                elif self.__n > 54:
                                    raise ValueError("Initialization of Zernike with radial order higher than 54"
                                                     + " is meaningless because of very slow calculation performance")
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
                        elif osa_i > 1539:
                            ValueError("Initialization of Zernike with OSA index higher than 1539"
                                       + " is meaningless because of very slow calculation performance")
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
                        elif noll_i > 1540:
                            ValueError("Initialization of Zernike with Noll index higher than 1540"
                                       + " is meaningless because of very slow calculation performance")
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
                            if self.__n > 54:
                                raise ValueError("Initialization of Zernike with radial order higher than 54"
                                                 + " is meaningless because of very slow calculation performance")
                            self.__osa_index = ZernPol.get_osa_index(self.__m, self.__n)
                            self.__noll_index = ZernPol.get_noll_index(self.__m, self.__n)
                    else:
                        raise ValueError("Fringe index provided not as an integer")
        else:
            raise ValueError("Length of provided key arguments are not equal 1 (OSA/Fringe/Noll index) or 2 (m, n orders)")
        # Also raise the ValueError if the ZernPol hasn't been initialized by orders / indices
        if not self.__initialized:
            raise ValueError("The initialization parameters for Zernike polynomial hasn't been parsed / recognized")
        # Generate warning messages for possible re-usage
        if self.__n > MAX_RADIAL_ORDER_COEFFS:
            self.__warn_mes_r = f"Call for radial order {self.__n} is higher than {MAX_RADIAL_ORDER_COEFFS}"
        else:
            self.warn_mes_r = ""
        if self.__n > MAX_RADIAL_ORDER_COEFFS_dR:
            self.__warn_mes_dr = f"Call for derivative of radial order {self.__n} is > than {MAX_RADIAL_ORDER_COEFFS_dR}"
        else:
            self.warn_mes_dr = ""

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

    def __str__(self) -> str:
        """
        Provide string information about the initialized class.

        Returns
        -------
        str
            String composed of a polynomial name (if specified in the stored file), polynomial's orders and indices.

        """
        name = self.get_polynomial_name()
        pol_signature = f": n={self.__n}, m={self.__m}, OSA={self.__osa_index}, Noll={self.__noll_index}, Fringe={self.__fringe_index}"
        if len(name) > 0:
            return name + pol_signature
        else:
            return "Zernike polynomial"

    def __gt__(self, other) -> bool:
        """
        Implement the method '>' for comparing two Zernike polynomials based on their OSA indices (!).

        Parameters
        ----------
        other : ZernPol
            Instance of ZernPol class.

        Raises
        ------
        ValueError
            If provided object for comparison isn't instance of the class ZernPol.

        Returns
        -------
        bool
            Result of comparison.

        """
        if isinstance(other, ZernPol):
            return self.__osa_index > other.__osa_index
        else:
            raise ValueError("Provided object for comparison isn't instance of the ZernPol class")

    def __eq__(self, other) -> bool:
        """
        Implement the method '==' for comparing two Zernike polynomials based on all their indices and orders equality.

        Parameters
        ----------
        other : ZernPol
            Instance of ZernPol class.

        Raises
        ------
        ValueError
            If provided object for comparison isn't instance of the class ZernPol.

        Returns
        -------
        bool
            Result of comparison.

        """
        if isinstance(other, ZernPol):
            return (self.__osa_index == other.__osa_index and self.__m == other.__m and self.__n == other.__n
                    and self.__noll_index == other.__noll_index and self.__fringe_index == other.__fringe_index)
        else:
            raise ValueError("Provided object for comparison isn't instance of the ZernPol class")

    # %% Polynomial values calculation in various forms
    def polynomial_value(self, r: Union[float, np.ndarray], theta: Union[float, np.ndarray], use_exact_eq: bool = False):
        """
        Calculate Zernike polynomial value(-s) within the unit circle.

        Calculation up to 10th order of Zernike function performed by exact equations from Ref.[2],
        after - using the recurrence equations taken from the Ref.[1], using shortcut of storing
        coefficients for each power of radius (coefficient*R^n) \n
        Surprisingly, the exact equation for polynomials are just pretty fast to use them directly, without
        any need to use the recurrence equations. It can be used by providing the flag as the input parameter. \n
        However, after ~ the 46th radial order due to the high integer values involved in the exact equation
        (factorials) producing ambiguous results (failed simple check that radial polynomial <= 1.0),
        only iterative equations (which along with increasing order become time-consuming and slow)
        could be used. The 40th radial order as the limit for usage of the exact equation is selected due to
        found increasing after this order discrepancy between results of recursive and factorial formulas.

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
            Note about the limit for usage of the exact equation - up to 40th radial order (n).

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
        if isinstance(r, type(np.zeros(1))) and isinstance(theta, type(np.zeros(1))):
            if r.shape != theta.shape:
                raise ValueError("Shape of input arrays r and theta is not equal")
        # Calculation using imported function from submodule depending on radial order, use different eq.
        nTr = normalization_factor(self)*triangular_function(self, theta)
        if not use_exact_eq:
            if self.__n <= 12:  # condition to switch from direct recurrence equation to finding of coeffs. algorithm
                return nTr*radial_polynomial(self, r)
            else:
                # Raise warning about slow calculations for orders more than 50, only once
                if self.__n > 50 and not self.__show_slow_calc_warn:
                    warn_mess = f"ZernPol(m={self.__m}, n={self.__n})" + warn_mess_slow_calc
                    warnings.warn(warn_mess)
                    self.__show_slow_calc_warn = True
                # Returning values using recursive scheme for finding the coefficients for all radial orders (e.g. R^6)
                return nTr*radial_polynomial_coeffs(self, r)
        else:
            if self.__n > MAX_RADIAL_ORDER_COEFFS:
                warnings.warn(self.__warn_mes_r + warn_mess_r_long)
                if isinstance(r, float):
                    return 0.0
                elif isinstance(r, np.ndarray):
                    return np.zeros(shape=r.shape)
            else:
                return nTr*radial_polynomial_eq(self, r)

    def radial(self, r: Union[float, np.ndarray], use_exact_eq: bool = False):
        """
        Calculate R(m, n) - radial Zernike function value(-s) within the unit circle.

        Calculation up to 10th order of Zernike function performed by exact equations from Ref.[2],
        after - using the recurrence equations taken from the Ref.[1], using shortcut of storing
        coefficients for each power of radius (coefficient*R^n) \n
        Surprisingly, the exact equation for polynomials are just pretty fast to use them directly, without
        any need to use the recurrence equations. It can be used by providing the flag as the input parameter. \n
        However, after ~ the 46th radial order due to the high integer values involved in the exact equation
        (factorials) producing ambiguous results (failed simple check that radial polynomial <= 1.0),
        only iterative equations (which along with increasing order become time-consuming and slow)
        could be used. The 40th radial order as the limit for usage of the exact equation is selected due to
        found increasing after this order discrepancy between results of recursive and factorial formulas.

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
            Note about the limit for usage of the exact equation - up to 40th radial order (n).

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
        if not use_exact_eq:
            if self.__n <= 12:  # condition to switch from direct recurrence equation to finding of coeffs. algorithm
                return radial_polynomial(self, r)
            else:
                # Raise warning about slow calculations for orders more than 50, only once
                if self.__n > 50 and not self.__show_slow_calc_warn:
                    warn_mess = f"ZernPol(m={self.__m}, n={self.__n})" + warn_mess_slow_calc
                    warnings.warn(warn_mess)
                    self.__show_slow_calc_warn = True
                # Returning values using recursive scheme for finding the coefficients for all radial orders (e.g. R^6)
                return radial_polynomial_coeffs(self, r)
        else:
            if self.__n > MAX_RADIAL_ORDER_COEFFS:
                warnings.warn(self.__warn_mes_r + warn_mess_r_long)
                if isinstance(r, float):
                    return 0.0
                elif isinstance(r, np.ndarray):
                    return np.zeros(shape=r.shape)
            else:
                return radial_polynomial_eq(self, r)

    def triangular(self, theta: Union[float, np.ndarray]):
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

    def radial_dr(self, r: Union[float, np.ndarray], use_exact_eq: bool = False):
        """
        Calculate derivative of radial Zernike polynomial value(-s) within the unit circle.

        Calculation up to 10th order of Zernike polynomials performed by exact equations,
        after - using the recurrence equations, using shortcut of storing
        coefficients for each power of radius (coefficient*R^n) \n
        The input flag use_exact_eq allows using the exact equation with factorials.
        But note that after 38th radial order the usage of the exact equation is forbidden, because
        after ~ the 44th radial order due to the high integer values associated with factorials and power
        values produced by derivatives leading to ambiguous results, only iterative equations
        (which along with increasing order become time-consuming and slow) could be used. The 38th radial order
        as the limit for usage of the exact equation is selected due to found increasing after this order
        discrepancy between results of recursive and factorial formulas.


        References
        ----------
        Same as for the method "radial" or "polynomial value"

        Parameters
        ----------
        r : float or numpy.ndarray
            Radius (radii) from unit circle or the range [0.0, 1.0], float / array, for which the polynomial is calculated.
        use_exact_eq : bool, optional
            Flag for using the exact equation with factorials. The default is False.
            Note about the limit for usage of the exact equation - up to 38th radial order (n).

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
        if not use_exact_eq:
            if self.__n <= 12:  # condition to switch from direct recurrence equation to finding of coeffs. algorithm
                return radial_derivative(self, r)
            else:
                # Raise warning about slow calculations for orders more than 50, only once
                if self.__n > 48 and not self.__show_slow_calc_warn:
                    warn_mess = f"ZernPol(m={self.__m}, n={self.__n})" + warn_mess_slow_calc
                    warnings.warn(warn_mess)
                    self.__show_slow_calc_warn = True
                # Returning values using recursive scheme for finding the coefficients for all radial orders (e.g. R^6)
                return radial_polynomial_coeffs_dr(self, r)
        else:
            if self.__n > MAX_RADIAL_ORDER_COEFFS_dR:
                warnings.warn(self.__warn_mes_dr + warn_mess_dr_long)
                if isinstance(r, float):
                    return 0.0
                elif isinstance(r, np.ndarray):
                    return np.zeros(shape=r.shape)
            else:
                return radial_derivative_eq(self, r)

    def triangular_dtheta(self, theta: Union[float, np.ndarray]):
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
        Calculate normalization factor for the Zernike polynomial calculated according to the References below.

        References
        ----------
        [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)
        [2] Check also preceding coefficients in the table Zj column: https://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials

        Returns
        -------
        float
            Normalization factor calculated according to the References.

        """
        return normalization_factor(self)

    # %% Static methods: indices transformations
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
        highest_order = 56  # default value, limited by the allowed maximal radial order + 2
        if "osa_index" in kwargs.keys():
            osa_index = kwargs.get("osa_index")
        elif "noll_index" in kwargs.keys():
            noll_index = kwargs.get("noll_index")
        elif "fringe_index" in kwargs.keys():
            fringe_index = kwargs.get("fringe_index")
            highest_order = 250  # to guarantee that right Fringe index found
        # Define m, n orders up to the specified above highest order (radial)
        stop_search = False
        for order in range(0, highest_order):
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

    # %% Static methods: generation parameters, sum of polynomials and plotting
    @staticmethod
    def _sum_zernikes_meshgrid(coefficients: Sequence[float], polynomials: Sequence, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Calculate sum of Zernike polynomials only for the numpy arrays with r and theta.

        For getting the sum of polynomials for float and array or floats of polar coordinates, use the sum_zernikes method.
        This implementation uses the meshgrid for calculation of sums on polar coordinates (see the code).
        It turns out that for large number of points usage of for loops for summation is more effective.

        Parameters
        ----------
        coefficients : Sequence[float]
            Coefficients of Zernike polynomials for calculation of their sum.
        polynomials : Sequence[ZernPol]
            Initialized polynomials as class instances of ZernPol class specified in this module.
        r : np.ndarray
            Radii from a unit circle.
        theta : np.ndarray
            Polar angles from a unit circle.

        Raises
        ------
        ValueError
            If the "polynomials" Sequence doesn't contain the ZernPol() class instances.

        Returns
        -------
        S : 2D numpy.ndarray
            Sum over the provided polar coordinates.

        """
        S = 0.0  # default value - sum
        if len(coefficients) != len(polynomials):
            raise ValueError("Lengths of coefficients and polynomials aren't equal")
        else:
            if not isinstance(r, np.ndarray) or not isinstance(theta, np.ndarray):
                warnings.warn("Requested calculation of surface (mesh) values with"
                              + " provided r or theta as not numpy.ndarray.\n"
                              + "The surface will be generated automatically.")
            else:
                r_size = np.size(r, 0); theta_size = np.size(theta, 0)
                theta_grid, r_grid = np.meshgrid(theta, r); S = np.zeros(shape=(r_size, theta_size))
                for i, coefficient in enumerate(coefficients):
                    if not isinstance(polynomials[i], ZernPol):
                        raise ValueError(f"Variable {polynomials[i]} isn't an instance of ZernPol class")
                    if abs(coefficient) > 0.0:
                        S += coefficient*polynomials[i].polynomial_value(r_grid, theta_grid)
        return S

    @staticmethod
    def sum_zernikes(coefficients: Sequence[float], polynomials: Sequence, r: Union[float, np.ndarray], theta: Union[float, np.ndarray],
                     get_surface: bool = False) -> Union[float, np.ndarray]:
        """
        Calculate sum of Zernike polynomials with their amplitude coefficients (e.g., for plotting over a unit circle).

        Parameters
        ----------
        coefficients : Sequence[float]
            Coefficients of Zernike polynomials for calculation of their sum.
        polynomials : Sequence[ZernPol]
            Initialized polynomials as class instances of ZernPol class specified in this module.
        r : float or numpy.ndarray
            Radius(Radii) from a unit circle.
        theta : float or numpy.ndarray
            Polar angle(-s) from a unit circle.
        get_surface : bool, optional
            If True, it forces to calculate 2D sum of polynomials based on r and theta (as a mesh of polar coordinates).
            The default is False. \n
            Note that if r and theta provided as the numpy ndarrays with different shapes and this flag is False, then
            the result of this method will raise ValueError (because r and theta shapes will be checked for equality).

        Raises
        ------
        TypeError
            If the input parameters aren't iterable (doesn't support len() function), this error will be raised.
        ValueError
            If the lengths of lists (tuples, numpy.ndarrays) aren't equal for coefficients and polynomials. \n
            Or if the list (tuple, numpy.ndarray vector, ...) with Zernike polynomials instances (ZernPol()).

        Returns
        -------
        Sum of Zernike polynomials: float or numpy.ndarray
            Depending on the input values and parameter get_surface - can be: float, 1D or 2D numpy.ndarrays.

        """
        S = 0.0  # default value - sum
        if len(coefficients) != len(polynomials):
            raise ValueError("Lengths of lists with polynomials and their amplitudes aren't equal")
        elif len(coefficients) == 0:
            raise ValueError("Length of list with polynomials is zero")
        else:
            if not get_surface or not isinstance(r, np.ndarray) or not isinstance(theta, np.ndarray):
                for i, coefficient in enumerate(coefficients):
                    if not isinstance(polynomials[i], ZernPol):
                        raise ValueError(f"Variable {polynomials[i]} isn't an instance of ZernPol class")
                    if i == 0:
                        S = coefficient*polynomials[i].polynomial_value(r, theta)  # if even coefficient = 0, gives initial array
                    else:
                        if abs(coefficient) > 0.0:
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
                        if abs(coefficient) > 0.0:
                            if theta_size > r_size:
                                for j in range(r_size):
                                    S[j, :] += coefficient*polynomials[i].polynomial_value(r[j], theta)[:]
                            else:
                                for j in range(theta_size):
                                    S[:, j] += coefficient*polynomials[i].polynomial_value(r, theta[j])[:]
        return S

    @staticmethod
    def gen_polar_coordinates(r_step: float = 0.01, theta_rad_step: float = round(np.pi/240, 7)) -> polar_vectors:
        """
        Generate the named tuple "PolarVectors" with R and Theta - vectors with polar coordinates for an entire unit circle.

        Note that R and Theta are generated as the numpy.ndarrays vectors (shape like (n elements, )). Their shapes are
        defined by the specification of r_step and theta_rad_step parameters.

        Parameters
        ----------
        r_step : float, optional
            Step for generation the vector with radiuses for an entire unit circle. The default is 0.01.
        theta_rad_step : float, optional
            Step for generation the vector with theta angles for an entire unit circle. The default is (np.pi/240).

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
        Rs = np.arange(0.0, 1.0+r_step, r_step); Thetas = np.arange(0.0, 2.0*np.pi+theta_rad_step, theta_rad_step)
        # Check that the last values on the generated ranges appeared not outside of ranges
        if Rs[Rs.shape[0]-1] > 1.0:
            Rs[Rs.shape[0]-1] = 1.0
        if Thetas[Thetas.shape[0]-1] > 2.0*np.pi:
            Thetas[Thetas.shape[0]-1] = 2.0*np.pi
        return polar_vectors(Rs, Thetas)

    @staticmethod
    def gen_equal_polar_mesh(n_points: int = 250) -> polar_vectors:
        """
        Generate the named tuple "PolarVectors" with R and Theta - vectors with polar coordinates for an entire unit circle.

        Note that R and Theta are generated as the numpy.ndarrays vectors (shape like (n elements, )). Their shapes are
        equal and defined by the parameter n_points.

        Parameters
        ----------
        n_points : int, optional
            Number of points between 0.0 ... 1.0 for radii and 0.0 ... 2pi for thetas. The default is 250.

        Returns
        -------
        polar_vectors
            namedtuple("PolarVectors", "R Theta"), where R - vector with radiuses values [0.0, r_step, ... 1.0],
            Theta - vector with theta angles values [0.0, theta_rad_step, ... 2*pi].

        """
        if n_points < 4:
            n_points = 4
        Rs = np.linspace(0.0, 1.0, n_points); Thetas = np.linspace(0.0, 2*np.pi, n_points)
        return polar_vectors(Rs, Thetas)

    @staticmethod
    def plot_profile(polynomial, color_map: str = "coolwarm", show_title: bool = True, use_defaults: bool = True,
                     projection: str = "2d", polar_coordinates: polar_vectors = ()):
        """
        Plot the provided Zernike polynomial (instance of ZernPol class) on the matplotlib figure.

        Note that the plotting function plt.show() creates the plot in non-interactive mode.

        Parameters
        ----------
        polynomial : ZernPol
            Instance of ZernPol class.
        color_map : str, optional
            Color map of the polar plot, common values for representation: coolwarm, jet, turbo, rainbow.
            As alternative - perceptually equal color maps: viridis, plasma. The default is "coolwarm".
            Note that rainbow, jet, turbo - not perceptually equal color maps.
        show_title : bool, optional
            Toggle for showing the name of polynomial on the plot or not (only works for 2D case).
        use_defaults : bool, optional
            Use default parameters for polar coordinates generation. The default is True.
        projection : str, optional
            Either "2d" ("2D") - for 2D profile plot, or "3d" ("3D") - for 3D surface plot. The default is "2d".
        polar_coordinates : polar_vectors, optional
            If the flag 'use_defaults' is False, this named tuple is used for accessing polar coordinates for plotting.
            The default is ().

        Raises
        ------
        ValueError
            If the flag use_defaults is False and polar_coordinates are not provided.

        Returns
        -------
        None.

        """
        if isinstance(polynomial, ZernPol):
            if not use_defaults and len(polar_coordinates) != 2:
                raise ValueError("Polar coordinates isn't provided as a tuple with values R, Theta")
            # Get polar coordinates
            if use_defaults:
                if projection == "3d" or projection == "3D":
                    r, theta = ZernPol.gen_equal_polar_mesh()
                else:
                    r, theta = ZernPol.gen_polar_coordinates()
            else:
                r, theta = polar_coordinates.R, polar_coordinates.Theta
            # Get profile or surface to plot
            amplitudes = [1.0]; zernikes = [polynomial]  # for reusing the sum function of polynomials
            zern_surface = ZernPol.sum_zernikes(amplitudes, zernikes, r, theta, get_surface=True)
            # Select plotting function between 2D and 3D plotting functions
            if projection == "3d" or projection == "3D":
                plot_sum_fig_3d(zern_surface, r, theta, color_map)
            else:
                if show_title:
                    plot_sum_fig(zern_surface, r, theta, title=polynomial.get_polynomial_name(),
                                 color_map=color_map)
                else:
                    plot_sum_fig(zern_surface, r, theta, "", color_map)

    @staticmethod
    def gen_zernikes_surface(coefficients: Sequence[float], polynomials: Sequence, r_step: float = 0.01,
                             theta_rad_step: float = round(np.pi/180, 7),
                             equal_n_coordinates: bool = False, n_points: int = 250) -> zernikes_surface:
        """
        Generate surface of provided Zernike polynomials on the generated polar coordinates used steps.

        Parameters
        ----------
        coefficients : Sequence[float]
            Coefficients of Zernike polynomials for calculation of their sum.
        polynomials : Sequence[ZernPol]
            Initialized polynomials as class instances of ZernPol class specified in this module.
        r_step : float, optional
            Step for generation the vector with radiuses for an entire unit circle. The default is 0.01. \n
            See also the documentation for the method gen_polar_coordinates().
        theta_rad_step : float, optional
            Step for generation the vector with theta angles for an entire unit circle. The default is (np.pi/180). \n
            See also the documentation for the method gen_polar_coordinates().
        equal_n_coordinates: bool, optional
            Switch between generation polar coordinates based on individual steps or on equal number of points.
            The default is False.
        n_points: int, optional
            Number of points used for generation of equal sized polar coordinates r, theta.

        Returns
        -------
        zernikes_surface
            namedtuple("ZernikesSurface", "ZernSurf R Theta") - tuple for storing mesh values for polar coordinates.
            ZernSurf variable is 2D matrix with the sum of the input polynomials on generated polar coordinates (R, Theta).

        """
        if not equal_n_coordinates:
            polar_vectors = ZernPol.gen_polar_coordinates(r_step, theta_rad_step)
        else:
            polar_vectors = ZernPol.gen_equal_polar_mesh(n_points)
        zernikes_sum = ZernPol.sum_zernikes(coefficients, polynomials, polar_vectors.R,
                                            polar_vectors.Theta, get_surface=True)
        return zernikes_surface(zernikes_sum, polar_vectors.R, polar_vectors.Theta)

    @staticmethod
    def plot_sum_zernikes_on_fig(figure: plt.Figure, coefficients: Sequence[float] = (), polynomials: Sequence = (), use_defaults: bool = True,
                                 zernikes_sum_surface: zernikes_surface = (), show_range: bool = True, color_map: str = "coolwarm",
                                 projection: str = "2d") -> plt.Figure:
        """
        Plot a sum of the specified Zernike polynomials by input lists (see function parameters) on the provided figure.

        Note that for showing the plotted figure, one needs to call appropriate functions (e.g., matplotlib.pyplot.show()
        or figure.show()) as the method of the input parameter figure.

        Parameters
        ----------
        figure : plt.Figure
            'Figure' class there the plotting will be done, previous plot will be cleared out.
        use_defaults : bool, optional
            Use for plotting default values for generation of a mesh of polar coordinates and calculation
            of Zernike polynomials sum or Use for plotting provided calculated beforehand surface. The default is True.
        coefficients : Sequence[float], optional
            Coefficients of Zernike polynomials for calculation of their sum. The default is ().
        polynomials : Sequence[ZernPol], optional
            Initialized polynomials as class instances of ZernPol class specified in this module. The default is ().
        zernikes_sum_surface : namedtuple("ZernikesSurface", "ZernSurf R Theta") , optional
            This tuple should contain the ZernSurf calculated on a mesh of polar coordinates R, Theta.
            This tuple could be generated by the call of the static method gen_zernikes_surface().
            Check the method signature for details. The default is ().
        show_range : bool, optional
            Flag for showing range of provided values as the colorbar on the figure. The default is True.
        color_map : str, optional
            Color map of the polar plot, common values for representation: coolwarm, jet, turbo, rainbow.
            As alternative - perceptually equal color maps: viridis, plasma. The default is "coolwarm".
            Note that rainbow, jet, turbo - not perceptually equal color maps.
        projection : str, optional
            Either "2d" ("2D") - for 2D profile plot, or "3d" ("3D") - for 3D surface plot. The default is "2d".

        Raises
        ------
        ValueError
            Check the signature for details. In general, it will be raised if some input parameters are inconsistent.

        Returns
        -------
        figure : plt.Figure
            Matplotlib.pyplot Figure class there the Zernike polynomials sum plotted.

        """
        if use_defaults and len(coefficients) == 0 and len(polynomials) == 0:
            raise ValueError("Input Sequence with coefficients or with polynomials is empty along with the flag 'use_defaults' - True")
        if not use_defaults and len(zernikes_sum_surface) != 3:
            raise ValueError("Zernike surface isn't provided as a tuple with values Sum surface, R, Theta")
        if use_defaults:
            if projection == "3d" or projection == "3D":
                polar_vectors = ZernPol.gen_equal_polar_mesh()
            else:
                polar_vectors = ZernPol.gen_polar_coordinates()
            zernikes_sum = ZernPol.sum_zernikes(coefficients, polynomials, polar_vectors.R,
                                                polar_vectors.Theta, get_surface=True)
            if projection == "3d" or projection == "3D":
                figure = subplot_sum_on_fig_3d(figure, zernikes_sum, polar_vectors.R, polar_vectors.Theta,
                                               show_range_colorbar=show_range, color_map=color_map)
            else:
                figure = subplot_sum_on_fig(figure, zernikes_sum, polar_vectors.R, polar_vectors.Theta,
                                            show_range_colorbar=show_range, color_map=color_map)
        else:
            if projection == "3d" or projection == "3D":
                figure = subplot_sum_on_fig_3d(figure, zernikes_sum_surface.ZernSurf, zernikes_sum_surface.R,
                                               zernikes_sum_surface.Theta, show_range_colorbar=show_range,
                                               color_map=color_map)
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
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1); fig.tight_layout()

    # %% Static methods: parameters checking
    @staticmethod
    def _check_radii(radii: Union[list, tuple, float, int, np.ndarray]) -> Union[float, np.ndarray]:
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
            if isinstance(radii, list) or isinstance(radii, tuple) or isinstance(radii, set):
                radii = np.asarray(radii)  # convert list or tuple to np.array
            else:
                radii = float(radii)  # attempt to convert r to float number, will raise ValueError if it's impossible
        # Checking that radii or radius lie in the range [0.0, 1.0]
        if isinstance(radii, np.ndarray):
            if np.min(radii) < 0.0 or np.max(radii) > 1.0:
                raise ValueError("Minimal or maximal value of radii laying outside unit circle [0.0, 1.0]")
        elif isinstance(radii, float):
            if radii > 1.0 or radii < 0.0:
                raise ValueError("Radius laying outside unit circle [0.0, 1.0]")
        return radii

    @staticmethod
    def _check_angles(angles: Union[list, tuple, float, int, np.ndarray]) -> Union[float, np.ndarray]:
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
            if isinstance(angles, list) or isinstance(angles, tuple) or isinstance(angles, set):
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


# %% Independent functions defs.
def generate_polynomials(max_order: int = 10) -> tuple:
    """
    Generate tuple with ZernPol instances (ultimately, representing Zernike polynomials) indexed using OSA scheme, starting with Piston(m=0,n=0).

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
    polynomials_list = [ZernPol(m=0, n=0)]  # list starting with piston
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
    # Generate random phases image - sum of polynomials on the 2D array of pixels converted to polar coordinates
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
        phases_image[i, :] = ZernPol.sum_zernikes(polynomials_ampl_list, polynomials_list, r, theta)
        # Set again all background pixels to zero, because they can be reset to some other constant (result of line above)
        for j in range(img_width):
            euclidean_dist = np.linalg.norm(center - np.asarray([i, j]))
            if euclidean_dist > img_radius:
                phases_image[i, j] = 0.0
    # Final conversion
    polynomials_list = tuple(polynomials_list)
    return phases_image, polynomials_amplitudes, polynomials_list


def generate_phases_image(polynomials: tuple = (), polynomials_amplitudes: tuple = (),
                          img_width: int = 513, img_height: int = 513) -> np.ndarray:
    """
    Generate phases image (profile) for provided set of polynomials with provided coefficients (amplitudes).

    Parameters
    ----------
    polynomials : tuple, optional
        Initialized ZernPol instances for generation of phases as the sum of all stored in this tuple polynomials. The default is ().
    polynomials_amplitudes : tuple, optional
        Amplitudes of polynomials provided in the tuple 'polynomials'. The default is ().
    img_width : int, optional
        Width of generated image. The default is 513.
    img_height : int, optional
        Height of generated image. The default is 513.

    Returns
    -------
    phases_image : np.ndarray
        2D image with phases calculated as the sum of provided polynomials.

    """
    # Generate random phases image - sum of polynomials on the 2D array of pixels converted to polar coordinates
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
        phases_image[i, :] = ZernPol.sum_zernikes(coefficients=list(polynomials_amplitudes), polynomials=list(polynomials), r=r, theta=theta)
        # Set again all background pixels to zero, because they can be reset to some other constant (result of line above)
        for j in range(img_width):
            euclidean_dist = np.linalg.norm(center - np.asarray([i, j]))
            if euclidean_dist > img_radius:
                phases_image[i, j] = 0.0
    return phases_image


def fit_polynomials(phases_image: np.ndarray, polynomials: tuple, crop_radius: float = 1.0, suppress_warnings: bool = False,
                    strict_circle_border: bool = False, round_digits: int = 4, return_cropped_image: bool = False) -> tuple:
    """
    Fit provided Zernike polynomials (instances of ZernPol class) as the input tuple to the 2D phase image.

    2D phase image implies that phases recorded depending on cartesian coordinates and circle aperture is cropped
    out from this image for fitting procedure. One can check the result of cropping by plotting return_cropped_image as
    True and plotting the second item from the returned tuple.

    Parameters
    ----------
    phases_image : numpy.ndarray
        2D image with recorded phases which should be approximated by the sum of Zernike polynomials.
        Note that image is assumed as the recorded phases on the cartesian coordinates. The circle
        (aperture) containing phases is cropped from the input image during the fitting procedure.
    polynomials : tuple
        Initialized tuple with instances of the ZernPol class that effectively represents target set of Zernike polynomials.
    crop_radius : float, optional
        Allow cropping pixel from range [0.5, 1.0], where 1.0 corresponds to radius of the cropped circle = min image size.
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
    logic_mask, cropped_phases_coordinates = crop_phases_img(phases_image, crop_radius, suppress_warnings, strict_circle_border)
    if return_cropped_image:
        cropped_image = logic_mask*phases_image  # for debugging
    zernike_coefficients = fit_zernikes(cropped_phases_coordinates, polynomials)
    zernike_coefficients = np.round(zernike_coefficients, round_digits)
    if return_cropped_image:
        return zernike_coefficients, cropped_image
    else:
        return zernike_coefficients, None


def fit_polynomials_vectors(polynomials: tuple, phases_vector: np.ndarray, radii_vector: np.ndarray,
                            thetas_vector: np.ndarray, round_digits: int = 4) -> np.ndarray:
    """
    Fit provided Zernike polynomials (instances of ZernPol class) as the input tuple to the provided 1D vectors.

    1D vectors should contain: phases recorded in the related radii and thetas (polar coordinates). For the example
    of phases and coordinates composing, see the module test_fitting in 'tests' sub-folder of the repository.

    Parameters
    ----------
    polynomials : tuple
        Initialized tuple with instances of the ZernPol class that effectively represents target set of Zernike polynomials for fitting.
    phases_vector : numpy.ndarray
        Recorded phases.
    radii_vector : numpy.ndarray
        Related radii - 1st polar coordinates for the recorded phases.
    thetas_vector : numpy.ndarray
        Related angles (thetas) - 2nd polar coordinates for the recorded phases.
    round_digits : int, optional
        Round digits for returned polynomials amplitudes (numpy.round(...) function call). The default is 4.

    Raises
    ------
    AttributeError
        Any of provided vectors (phases_vector, etc.) isn't proper 1D numpy.ndarray (len(array.shape > 1)).

    Returns
    -------
    zernike_coefficients : numpy.ndarray
        1D numpy.ndarray with the fitted coefficients of Zernike polynomials.

    """
    # Checking input data consistency
    if len(phases_vector.shape) > 1 or len(radii_vector.shape) > 1 or len(thetas_vector.shape) > 1:
        raise TypeError("Some of provided vector is not 1D, check the call len(...shape)")
    # Call to fitting procedure
    zernike_coefficients = fit_zernikes((phases_vector, radii_vector, thetas_vector), polynomials)
    zernike_coefficients = np.round(zernike_coefficients, round_digits)
    return zernike_coefficients


def compare_performances(min_order: int, max_order: int) -> tuple:
    """
    Compare performances of radial polynomials calculation by using recursive and exact equations.

    Comparison achieved by simple measuring of time in msec needed for calculation of all radial
    polynomials from minimal radial order up to maximum radial order returned as tuple.

    Parameters
    ----------
    min_order : int
        Minimum radial order of used polynomials (n).
    max_order : int
        Maximum radial order of used polynomials (n).

    Returns
    -------
    tuple
        Composed by time for recursive calculation and time for exact calculation.

    """
    # Generation of orders in OSA/ANSI indexing scheme for initializing Zernike polynomials
    zernpols = []
    for order in range(min_order, max_order+1):
        m = -order; n = order
        zernpols.append(ZernPol(m=m, n=n))
        for n_azimuthals in range(0, order):
            m += 2
            zernpols.append(ZernPol(m=m, n=n))
    # Generation numpy array with radii
    n_points = 251
    test_r = np.zeros(shape=(n_points, ))
    for i in range(n_points):
        test_r[i] = i/(n_points-1)
    test_r = np.round(test_r, 6)
    # Measuring performance of radial polynomials calculations using recursive implementation
    t1 = time.perf_counter()
    for i, polynomial in enumerate(zernpols):
        polynomial.radial(test_r)  # calculate radial polynomials over vector of radii
    t2 = time.perf_counter()
    t_recursive_ms = round(1000*(t2-t1), 3)
    # Measuring performance of radial polynomials calculations using exact implementation
    t1 = time.perf_counter()
    for i, polynomial in enumerate(zernpols):
        polynomial.radial(test_r, use_exact_eq=True)  # calculate radial polynomials over vector of radii
    t2 = time.perf_counter()
    t_exact_ms = round(1000*(t2-t1), 3)
    return (t_recursive_ms, t_exact_ms, f"Used polynomials: {i}", f"Radii: {n_points}")


def _estimate_high_order_calc_times():
    """
    Estimate slowing down of radial polynomial calculation with increasing of radial order.

    Returns
    -------
    None.

    """
    r = 0.55
    high_order_pols = [ZernPol(m=-2, n=46), ZernPol(m=1, n=47), ZernPol(m=2, n=48), ZernPol(m=-1, n=49),
                       ZernPol(m=2, n=50), ZernPol(m=1, n=51), ZernPol(m=-2, n=52)]
    times = []
    # Single polynomials values
    for pol in high_order_pols:
        # calculate radial polynomials over vector of radii
        t1 = time.perf_counter(); pol.radial(r); t2 = time.perf_counter()
        times.append(f"{pol.get_mn_orders()}: {int(round(1000*(t2-t1), 0))} ms")
    # Delete polynomials with too high orders for derivatives calculates
    high_order_pols.pop(len(high_order_pols)-1); high_order_pols.pop(len(high_order_pols)-1)
    high_order_pols.pop(len(high_order_pols)-1)
    print(times); times = []
    # Single derivative polynomials values
    for pol in high_order_pols:
        # calculate derivatives of radial polynomials over vector of radii
        t1 = time.perf_counter(); pol.radial_dr(r); t2 = time.perf_counter()
        times.append(f"Deriv. {pol.get_mn_orders()}: {int(round(1000*(t2-t1), 0))} ms")
    print(times)


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
    # Compare two implementations of Zernike pol-s sum calculation: direct and using meshgrid
    pols = [ZernPol(osa=2), ZernPol(osa=4), ZernPol(osa=7), ZernPol(osa=10), ZernPol(osa=15),
            ZernPol(osa=3), ZernPol(osa=9), ZernPol(osa=12), ZernPol(osa=16), ZernPol(osa=19)]
    ampls = [-0.85, 0.85, 0.24, -0.37, 1.0, 0.1, -1.0, -0.05, 1.1, 0.41]
    radii = np.arange(start=0.0, stop=1.0 + 0.001, step=0.001)
    thetas = np.arange(start=0.0, stop=2.0*np.pi + np.pi/180, step=np.pi/180)
    t1 = time.perf_counter(); ZernPol.sum_zernikes(ampls, pols, radii, thetas, get_surface=True)
    t_direct = int(round(1000*(time.perf_counter() - t1), 0)); t1 = time.perf_counter()
    ZernPol._sum_zernikes_meshgrid(ampls, pols, radii, thetas); t_meshgr = int(round(1000*(time.perf_counter() - t1), 0))
    print(f"Diff. calc. time b/t direct ({t_direct} ms) and meshgrid ({t_meshgr} ms) sums: {t_direct - t_meshgr} ms")
    print("ALL TEST PASSED")


# %% Define default export classes and methods used with import * statement (import * from zernikepol)
__all__ = ['ZernPol', 'fit_polynomials_vectors', 'fit_polynomials', 'generate_phases_image',
           'generate_random_phases', 'generate_polynomials']

# %% Tests
if __name__ == "__main__":
    _test_plots = False  # regulates testing of plotting various plots
    _test_calculations = False  # regulates tests below concerning calculations
    check_conformity()  # testing initialization

    # Testing plotting, the plots will be opened in the additional pop-up windows
    if _test_plots:
        plt.close("all")  # close all previously opened plots
        t1 = time.perf_counter(); zp = ZernPol(m=0, n=2); ZernPol.plot_profile(zp, color_map="jet", show_title=True)  # basic plot
        t2 = time.perf_counter(); print("Plotting of 1 non-zero polynomial takes ms: ", int(round(1000*(t2-t1), 0)))
        coordinates = ZernPol.gen_polar_coordinates(r_step=0.005)
        zp = ZernPol(m=-10, n=30); ZernPol.plot_profile(zp, color_map="jet", show_title=False, polar_coordinates=coordinates)  # high order plot
        zp = ZernPol(m=0, n=0); ZernPol.plot_profile(zp, color_map="turbo", show_title=True)  # plot of piston polynomial

        # Testing 3D surface plotting
        ZernPol.plot_profile(ZernPol(m=0, n=2), color_map="viridis", projection="3d")

        # Testing 3D figure plotting on the externally initialized Figure class
        fig3d = plt.figure(figsize=(6.8, 6.8))
        zern_surface = ZernPol.gen_zernikes_surface([1.0], [ZernPol(m=0, n=2)], equal_n_coordinates=True, n_points=400)
        ZernPol.plot_sum_zernikes_on_fig(figure=fig3d, use_defaults=False, zernikes_sum_surface=zern_surface,
                                         show_range=True, color_map="magma", projection="3D")
        fig3d2 = plt.figure(figsize=(5.8, 5.8))
        ZernPol.plot_sum_zernikes_on_fig(figure=fig3d2, coefficients=[1.0], polynomials=[ZernPol(m=0, n=2)],
                                         show_range=True, color_map="bwr", projection="3D")

        # Testing accelerated plotting / sum calculation
        fig3 = plt.figure(figsize=(3, 3))
        t1 = time.perf_counter(); n_pols = 31; polynomials = []; coefficients = [0.0]*n_pols
        for i in range(n_pols):
            polynomials.append(ZernPol(osa=58+i))
        coefficients[0] = 1.0  # only 1st polynomial will be plotted
        fig3 = ZernPol.plot_sum_zernikes_on_fig(figure=fig3, coefficients=coefficients, polynomials=polynomials,
                                                show_range=False, color_map="turbo")
        fig3.subplots_adjust(0, 0, 1, 1)
        t2 = time.perf_counter(); print("Plotting of 1 non-zero and 30 zero pol-s takes ms: ", int(round(1000*(t2-t1), 0)))

        # Tests with generation / restoring Zernike profiles (phases images)
        phases_image, polynomials_ampls, polynomials = generate_random_phases(img_height=301, img_width=321)
        plt.figure(); plt.axis("off"); plt.imshow(phases_image, cmap="jet"); plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)
        polynomials_amplitudes, cropped_img = fit_polynomials(phases_image, polynomials, return_cropped_image=True,
                                                              strict_circle_border=False, crop_radius=1.0)
        plt.figure(); plt.axis("off"); plt.imshow(cropped_img, cmap="jet")
        plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)

        # Updated test of fitting including piston polynomial
        height = 500; width = 481; crop_r = 1.0; strict_border = True; pols_coeffs = [-0.75, 0.86, 0.41]; fig4 = plt.figure(figsize=(4, 4))
        polynomials = [ZernPol(osa=0), ZernPol(m=0, n=2), ZernPol(m=-3, n=3)]; rs, angles = ZernPol.gen_polar_coordinates()
        phase_profile = ZernPol.sum_zernikes(coefficients=pols_coeffs, polynomials=polynomials, r=rs, theta=angles, get_surface=True)
        # Below - plotting specified polynomials on the polar coordinates
        ZernPol.plot_sum_zernikes_on_fig(figure=fig4, use_defaults=False, zernikes_sum_surface=zernikes_surface(phase_profile, rs, angles),
                                         color_map="jet")
        # Below - generate phases image with the cartesian coordinates
        phases_image2 = generate_phases_image(polynomials=tuple(polynomials), polynomials_amplitudes=tuple(pols_coeffs),
                                              img_height=height, img_width=width)
        plt.figure(); plt.axis("off"); im = plt.imshow(phases_image2, cmap="jet"); plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)
        plt.colorbar(mappable=im)
        # Below - fitting procedure on the provided phases image
        polynomials_amplitudes2, cropped_img2 = fit_polynomials(phases_image2, polynomials, return_cropped_image=True,
                                                                strict_circle_border=strict_border, crop_radius=crop_r)
        print("Difference between used amplitudes and fitted ones:", pols_coeffs-polynomials_amplitudes2)
        plt.figure(); plt.axis("off"); im = plt.imshow(cropped_img2, cmap="jet"); plt.tight_layout(); plt.subplots_adjust(0, 0, 1, 1)
        plt.colorbar(mappable=im)

        plt.show()  # show all images created by plt.figure() calls

    # Testing calculations and their performance comparison
    if _test_calculations:
        # Simple test of two concepts of calculations - exact and recursive equations
        z = ZernPol(n=30, m=-2); print("Diff. between recursive and exact equations:",
                                       round(z.radial(0.85) - z.radial(0.85, use_exact_eq=True), 9))
        z = ZernPol(n=32, l=0); print("Diff. between recursive and exact equations:",
                                      round(z.radial(0.35) - z.radial(0.35, use_exact_eq=True), 9))
        r = 0.955; theta = np.pi/8; z = ZernPol(osa=55)
        print("Diff. between recursive and exact equations:",
              round(z.polynomial_value(r, theta) - z.polynomial_value(r, theta, use_exact_eq=True), 9))
        z = ZernPol(n=35, l=-1); print("Diff. between recursive & exact eq-s for derivatives:",
                                       round(z.radial_dr(0.78) - z.radial_dr(0.78, use_exact_eq=True), 9))
        z = ZernPol(n=38, m=-2); print("Diff. between recursive & exact eq-s for derivatives:",
                                       round(z.radial_dr(0.9) - z.radial_dr(0.9, use_exact_eq=True), 9))
        # Compare performances
        print("Tabular (10th order) / exact calc. times:", compare_performances(1, 10))
        print("Recursive / exact calc. times for high orders:", compare_performances(12, 40))
        # Statement below producing expected warnings, it's used for performance estimation
        _estimate_high_order_calc_times()
