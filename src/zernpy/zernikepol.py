# -*- coding: utf-8 -*-
"""
Main script with the class definition for accessing Zernike polynomial initialization, calculation and plotting.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
from pathlib import Path

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calculations.calc_zernike_pol import normalization_factor
else:
    from .calculations.calc_zernike_pol import normalization_factor

# %% Module parameters
__docformat__ = "numpydoc"


# %% Class def.
class ZernPol:
    """Specify the Zernike polynomial and associated calculation methods."""

    # Pre-initialized class variables
    __initialized: bool = False  # will be set to true after successful construction
    __n: int = 0; __m: int = 0; __osa_index: int = 0; __noll_index: int = 0; __fringe_index: int = 0
    # References for names - see the docstring of the method get_polynomial_name
    polynomial_names: dict = {
        (-1, 1): "Vertical tilt", (1, 1): "Horizontal tilt", (-2, 2): "Oblique astigmatism",
        (0, 2): "Defocus", (2, 2): "Vertical astigmatism", (-3, 3): "Vertical trefoil",
        (-1, 3): "Vertical coma", (1, 3): "Horizontal coma", (3, 3): "Oblique trefoil",
        (-4, 4): "Oblique quadrafoil", (-2, 4): "Oblique secondary astigmatism",
        (0, 4): "Primary spherical", (2, 4): "Vertical secondary astigmatism",
        (4, 4): "Vertical quadrafoil", (-5, 5): "Vertical pentafoil",
        (-3, 5): "Vertical secondary trefoil", (-1, 5): "Vertical secondary coma",
        (1, 5): "Horizontal secondary coma", (3, 5): "Oblique secondary trefoil",
        (5, 5): "Oblique pentafoil", (-6, 6): "Oblique sexfoil",
        (-4, 6): "Oblique secondary quadrafoil", (-2, 6): "Oblique thirdly astigmatism",
        (0, 6): "Secondary spherical", (2, 6): "Vertical thirdly astigmatism",
        (4, 6): "Vertical secondary quadrafoil", (6, 6): "Vertical sexfoil",
        (-7, 7): "Vertical septfoil", (-5, 7): "Vertical secondary pentafoil",
        (-3, 7): "Vertical thirdly trefoil", (-1, 7): "Vertical thirdly coma",
        (1, 7): "Horizontal thirdly coma", (3, 7): "Oblique thirdly trefoil",
        (5, 7): "Oblique secondary pentafoil", (7, 7): "Oblique septfoil"}
    short_polynomial_names: dict = {
        (-1, 1): "Vert. tilt", (1, 1): "Hor. tilt", (-2, 2): "Obliq. astigm.",
        (0, 2): "Defocus", (2, 2): "Vert. astigm.", (-3, 3): "Vert. 3foil",
        (-1, 3): "Vert. coma", (1, 3): "Hor. coma", (3, 3): "Obliq. 3foil",
        (-4, 4): "Obliq. 4foil", (-2, 4): "Obliq. 2d ast.",
        (0, 4): "Spherical", (2, 4): "Vert. 2d ast.", (4, 4): "Vert. 4foil",
        (-5, 5): "Vert. 5foil", (-3, 5): "Vert. 2d 3foil", (-1, 5): "Vert. 2d coma",
        (1, 5): "Hor. 2d coma", (3, 5): "Obliq. 2d 3foil",
        (5, 5): "Obliq. 5foil", (-6, 6): "Obliq. 6foil", (-4, 6): "Obliq.2d 4foil",
        (-2, 6): "Obliq. 3d ast.", (0, 6): "2d spherical", (2, 6): "Vert. 3d ast.",
        (4, 6): "Vert. 2d 4foil", (6, 6): "Vert. 6foil", (-7, 7): "Vert. 7foil",
        (-5, 7): "Vert. 2d 5foil", (-3, 7): "Vert. 3d 3foil", (-1, 7): "Vert. 3d coma",
        (1, 7): "Hor. 3d coma", (3, 7): "Obliq.3d 3foil",
        (5, 7): "Obliq.2d 5foil", (7, 7): "Obliq. 7foil"}

    def __init__(self, **kwargs):
        """
        Initialize of the class with definition of Zernike polynomial.

        Parameters
        ----------
        **kwargs : comma separated parameters
            DESCRIPTION.

        Raises
        ------
        ValueError
            Raised if the specified orders (m, n) or index (OSA, Noll...) have some inconsistencies.

        References
        ----------
        [1] Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials
        [2] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)
        [3] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011)

        Returns
        -------
        ZernPol() class.

        """
        __orders_specified = False; __index_specified = False; key = ""
        # Zernike polynomial specified with key arguments m, n - check firstly for these parameters
        if "n" in kwargs.keys() or "radial_order" in kwargs.keys():
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
                        if not((self.__n - abs(self.__m)) % 2 == 0):  # see [1]
                            raise ValueError("Failed sanity check of orders n and m (n - |m| = even)")
                        # m and n specified correctly, calculate other properties - various indices
                        else:
                            __orders_specified = True  # set the flag to True
                            # Calculation of various indices according to [1]
                            self.__osa_index = ZernPol.get_osa_index(self.__m, self.__n)
                            self.__noll_index = ZernPol.get_noll_index(self.__m, self.__n)
                            self.__fringe_index = ZernPol.get_fringe_index(self.__m, self.__n)
                    else:
                        raise ValueError("Azimuthal order (m) provided not as an integer")
                else:
                    # the n order defined, but m hasn't been found
                    self.__n = 0
            else:
                raise ValueError("Radial order (n) provided not as an integer")
        # OSA / ANSI index used for Zernike polynomial initialization
        elif ("osa_index" in kwargs.keys() or "osa" in kwargs.keys() or "ansi_index" in kwargs.keys()
              or "ansi" in kwargs.keys()):
            if __orders_specified:
                raise ValueError("The polynomial has been already initialized with (m, n) parameters")
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
                    self.__osa_index = kwargs.get(key); __index_specified = True
                    self.__m, self.__n = ZernPol.get_orders(osa_index=self.__osa_index)
                    self.__noll_index = ZernPol.get_noll_index(self.__m, self.__n)
                    self.__fringe_index = ZernPol.get_fringe_index(self.__m, self.__n)
                else:
                    raise ValueError("OSA / ANSI index provided not as an integer")
        # Noll index used for Zernike polynomial initialization
        elif "noll_index" in kwargs.keys() or "noll" in kwargs.keys():
            if __orders_specified:
                raise ValueError("The polynomial has been already initialized with (m, n) parameters")
            else:
                if "noll_index" in kwargs.keys():
                    key = "noll_index"
                elif "noll" in kwargs.keys():
                    key = "noll"
                if isinstance(kwargs.get(key), int):
                    self.__noll_index = kwargs.get(key); __index_specified = True
                    self.__m, self.__n = ZernPol.get_orders(noll_index=self.__noll_index)
                    self.__osa_index = ZernPol.get_osa_index(self.__m, self.__n)
                    self.__fringe_index = ZernPol.get_fringe_index(self.__m, self.__n)
                else:
                    raise ValueError("Noll index provided not as an integer")
        # Fringe / Univ. of Arizona index used for Zernike polynomial initialization
        elif "fringe_index" in kwargs.keys() or "fringe" in kwargs.keys():
            if __orders_specified:
                raise ValueError("The polynomial has been already initialized with (m, n) parameters")
            else:
                if "fringe_index" in kwargs.keys():
                    key = "fringe_index"
                elif "fringe" in kwargs.keys():
                    key = "fringe"
                if isinstance(kwargs.get(key), int):
                    self.__fringe_index = kwargs.get(key); __index_specified = True
                    self.__m, self.__n = ZernPol.get_orders(fringe_index=self.__fringe_index)
                    self.__osa_index = ZernPol.get_osa_index(self.__m, self.__n)
                    self.__noll_index = ZernPol.get_noll_index(self.__m, self.__n)
                else:
                    raise ValueError("Fringe index provided not as an integer")
        else:
            raise ValueError("Key arguments haven't been parsed / recognized, see docs for acceptable values")
        # Also raise the ValueError if the ZernPol hasn't been initialized by orders / indices
        if not __index_specified and not __orders_specified:
            raise ValueError("The initialization parameters for Zernike polynomial hasn't been parsed")

    def get_indices(self):
        """
        Return the tuple with following orders: ((m, n), OSA index, Noll index, Fringe index).

        Returns
        -------
        tuple
            (1 tuple with (azimuthal (m), radial(n)) orders, OSA index, Noll index, Fringe index.
             All indices are integers.
        """
        return (self.__m, self.__n), self.__osa_index, self.__noll_index, self.__fringe_index

    def get_polynomial_orders(self) -> tuple:
        """
        Return tuple with the (azimuthal, radial) orders, i.e. return (m, n).

        Returns
        -------
        tuple
            with the (azimuthal, radial) orders for the initialized Zernike polynomial.

        """
        return (self.__m, self.__n)

    def get_polynomial_name(self, short: bool = False) -> str:
        """
        Return string with the name of polynomial up to 7th order.

        Parameters
        ----------
        short : bool, optional
            If True, this method returns shortened name. The default is False.

        References
        ----------
        [1] Up to 4th order: Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials
        [2] 5th order names: from the website https://www.telescope-optics.net/monochromatic_eye_aberrations.htm
        6th order - 7th order: my guess about the naming

        Returns
        -------
        str
            Name of the initialized polynomial.

        """
        name = ""
        if short:
            if (self.__m, self.__n) in self.short_polynomial_names.keys():
                name = self.short_polynomial_names[(self.__m, self.__n)]
        else:
            if (self.__m, self.__n) in self.polynomial_names.keys():
                name = self.polynomial_names[(self.__m, self.__n)]
        return name

    # %% Calculations
    def get_polynomial_value(self, r, theta):
        pass

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
    def get_orders(**kwargs) -> tuple:
        """
        Return tuple with (m, n) - azimuthal and radial orders of Zernike polynomials.

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
            m, n = ZernPol.get_orders(osa_index=osa_index)
            return ZernPol.get_noll_index(m, n)
        else:
            raise ValueError(f"Provided {osa_index} isn't integer or less than 0")

# %% Test functions for the external call


# %% Tests
if __name__ == "__main__":
    zp = ZernPol(m=-1, n=1)
    print(zp.get_indices())
    zp = ZernPol(m=0, n=2)
    print(zp.get_indices())
    print(zp.get_polynomial_name())
    print(normalization_factor(zp))
    zp = ZernPol(m=-3, n=5)
    print(zp.get_indices())
    zp = ZernPol(osa_index=8)
    print(zp.get_indices())
    print(zp.get_polynomial_name())
    zp = ZernPol(noll_index=6)
    print(zp.get_indices())
    print(zp.get_polynomial_name())
    print(zp.osa2noll(3))
