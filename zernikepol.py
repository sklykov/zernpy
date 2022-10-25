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
    __n = 0; __m = 0

    def __init__(self, **kwargs):
        __orders_specified = False; __index_specified = False
        if "n" in kwargs.keys() or "radial_order" in kwargs.keys():
            __orders_specified = True  # makes an assumption that Zernike pol. will be specified by orders
            # get the actual name of the key for radial order
            if "n" in kwargs.keys():
                key = "n"
            else:
                key = "radial_order"
            if isinstance(kwargs.get(key), int):
                self.__n = kwargs.get(key)
            else:
                raise ValueError
        print(kwargs)


# %% Tests
if __name__ == "__main__":
    zp = ZernPol(m=-1, n=1)
