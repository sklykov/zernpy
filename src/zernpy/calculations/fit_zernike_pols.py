# -*- coding: utf-8 -*-
"""
Fitting of Zernike polynomials to the provided deformations on an image.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
from pathlib import Path
import warnings

# %% Local imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from calc_zernike_pol import define_orders, normalization_factor, radial_polynomial, triangular_function
else:
    from .calc_zernike_pol import define_orders, normalization_factor, radial_polynomial, triangular_function


# %% Functions definitions
def crop_deformations(deformation_image: np.ndarray, crop_radius: float) -> np.ndarray:
    # Sanity checks of provided deformation image
    if not isinstance(deformation_image, np.ndarray):
        raise ValueError("Please provide the deformation image as the numpy ndarray for proper method calls")
    else:
        if deformation_image.ndim == 2:
            rows, cols = deformation_image.shape
            if rows != cols:
                warnings.warn("Deformation image isn't square, results of fitting could be ambiguous")
                img_min_size = min(rows, cols)
            else:
                img_min_size = rows
            if img_min_size % 2 == 0:
                warnings.warn("Deformation image provided with even minimal image size,"
                              + "it's error prone to define image center")
            if img_min_size < 4:
                raise ValueError("Provided image is too small (min size < 4) for producing any meaningful result")
        else:
            raise ValueError("Dimensions of provided image not equal to 2")
    # Sanity checks of provided cropping radius
    if not isinstance(crop_radius, float):
        raise ValueError("Provided radius for cropping is not instance of float")
    if not 0.5 <= crop_radius <= 1.0:
        raise ValueError("Provided radius for cropping is not in the range [0.5, 1.0]")
