# -*- coding: utf-8 -*-
"""
Main script with the classes definitions.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports

# %% Local (package-scoped) imports


# %% Class def.
class ZernPol():
    """Specify the Zernike polynomial and associated calculation methods."""

    __initialized = False  # will be set to true after successful construction
    __n = 0; __m = 0; __osa_index = 0; __noll_index = 0; __fringe_index = 0

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
        [2]
        [3]

        Returns
        -------
        ZernPol() class.

        """
        __orders_specified = False; __index_specified = False
        # Zernike polynomial specified with key arguments m, n - check firstly for these parameters
        if "n" in kwargs.keys() or "radial_order" in kwargs.keys():
            # get the actual name of the key for radial order
            if "n" in kwargs.keys():
                key = "n"
            else:
                key = "radial_order"
            if isinstance(kwargs.get(key), int):
                self.__n = kwargs.get(key)  # radial order acknowledged
                if "m" or "l" or "azimuthal_order" or "angular_frequency" in kwargs.keys():
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
                        # m and n specified correctly, calculate other properties - variuos indices
                        else:
                            __orders_specified = True  # set the flag to True
                            self.__osa_index = (self.__n*(self.__n + 2) + self.__m)//2
                            # Calculation of Noll index according to [1]
                            add_n = 1
                            if self.__m > 0:
                                if (self.__n % 4) == 0:
                                    add_n = 0
                                elif ((self.__n - 1) % 4) == 0:
                                    add_n = 0
                            elif self.__m < 0:
                                if ((self.__n - 2) % 4) == 0:
                                    add_n = 0
                                elif ((self.__n - 3) % 4) == 0:
                                    add_n = 0
                            self.__noll_index = (self.__n*(self.__n + 1))//2 + abs(self.__m) + add_n
                    else:
                        raise ValueError("Azimuthal order (m) provided not as integer")
                else:
                    # the n order defined, but m hasn't been found
                    self.__n = 0
            else:
                raise ValueError("Radial order (n) provided not as integer")
        elif "osa_index" or "osa" or "osa_i" or "ansi_index" or "ansi" or "ansi_i" in kwargs.keys():
            if __orders_specified:
                raise ValueError("The polynomial has been already initialized with (m, n) parameters")
            else:
                if "osa_index" in kwargs.keys():
                    key = "osa_index"
        elif "noll_index" or "noll_i" or "noll" in kwargs.keys():
            pass
        elif "fringe_index" or "fringe_i" or "fringe" in kwargs.keys():
            pass
        else:
            raise ValueError("Key arguments haven't been parsed / recognized, see docs for acceptable values")

    def get_indices(self):
        return ((self.__m, self.__n), self.__osa_index, self.__noll_index)


# %% Tests
if __name__ == "__main__":
    zp = ZernPol(m=-1, n=1)
    print(zp.get_indices())
    zp = ZernPol(m=0, n=2)
    print(zp.get_indices())
    zp = ZernPol(m=-3, n=5)
    print(zp.get_indices())
