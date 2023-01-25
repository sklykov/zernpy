# -*- coding: utf-8 -*-
"""
Fitting of Zernike polynomials to the provided deformations on an image.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
import warnings

# %% Local imports


# %% Module parameters
__docformat__ = "numpydoc"


# %% Functions definitions
def crop_deformations(deformations_image: np.ndarray, crop_radius: float = 1.0,
                      suppress_warns: bool = False, strict_border: bool = False) -> np.ndarray:
    __warn_message = ""  # holder for warning message below
    cropped_logic_mask = None  # initial value for returning it in the case of some incosistency
    # Sanity checks of provided crop radius
    if not isinstance(crop_radius, float):
        crop_radius = float(crop_radius)
    crop_radius = np.round(crop_radius, 4)  # rounding for performing exact comparisons
    if not 0.5 <= crop_radius <= 1.0:
        raise ValueError("Provided radius for cropping is not in the range [0.5, 1.0]")
    # Sanity checks of provided deformation image and cropping the pixels out
    if not isinstance(deformations_image, np.ndarray):
        raise ValueError("Please provide the deformation image as the numpy ndarray for proper method calls")
    else:
        if deformations_image.ndim == 2:
            # Check input image shape
            rows, cols = deformations_image.shape; cropped_logic_mask = np.zeros(shape=deformations_image.shape,
                                                                                 dtype=deformations_image.dtype)
            if rows != cols:
                __warn_message = "Deformation image isn't square, results of fitting could be ambiguous"
                img_min_size = min(rows, cols); img_max_size = max(rows, cols)
                if not suppress_warns:
                    warnings.warn(__warn_message)
            else:
                img_min_size = rows; img_max_size = rows
            if rows % 2 == 0 or cols % 2 == 0:
                __warn_message = ("Deformation image provided with even rows or columns,"
                                  + "it's error prone to define exact image center")
                if not suppress_warns:
                    warnings.warn(__warn_message)
            if img_min_size < 3:
                raise ValueError("Provided image is too small (min size < 3) for producing any meaningful result")
            # Calculate center of the input image - depending on its sizes
            if rows % 2 == 0:
                center_row = (rows // 2) - 0.5
            else:
                center_row = (rows - 1) // 2
            if cols % 2 == 0:
                center_col = (cols // 2) - 0.5
            else:
                center_col = (cols - 1) // 2
            # Define the radius for cropping, different rules applied for avoiding removing many pixels out
            if img_min_size % 2 == 0:
                r = (img_min_size // 2) - 0.5
            else:
                r = (img_min_size - 1) // 2
            if img_max_size - img_min_size < 2:
                if img_max_size % 2 == 0:
                    r = (img_max_size // 2) - 0.5
                else:
                    r = (img_max_size - 1) // 2
            # Definition of accepted for cropping difference between radius and distance to the pixel
            radius_epsilon = 0.001
            if not strict_border and ((rows % 2 == 0 and cols % 2 == 0) or (abs(rows - cols) == 1)):
                radius_epsilon = 0.055  # defined for matrix(4, 4) for including border pixels
                __warn_message = ("Deformation image provided as a square or with 1 pixel difference in dimensions.\n"
                                  + "Note that additional border pixels are included to crop.")
                if not suppress_warns:
                    warnings.warn(__warn_message)
            # Calculate polar coordinates of pixels
            radii = np.zeros(shape=deformations_image.shape); thetas = np.zeros(shape=deformations_image.shape)
            recalibrate_radii = False; recalibration_coeff = 1.0
            for i in range(rows):
                for j in range(cols):
                    radii[i, j] = np.round(np.sqrt(np.power((i - center_row), 2) + np.power((j - center_col), 2))/r, 4)
                    thetas[i, j] = np.arctan2(center_row - i, j - center_col)  # making it counterclockwise
                    if radii[i, j] - crop_radius < radius_epsilon:
                        cropped_logic_mask[i, j] += 1  # should convert to the actual type of image, for it add 1 as integer
                        # Checking that recalibration needed - for cropping only pixels formally from unit radius circle
                        if not strict_border and radii[i, j] - crop_radius > 0.001:
                            recalibrate_radii = True
                            if recalibration_coeff < radii[i, j]:
                                recalibration_coeff = radii[i, j]
                    # else:
                    #     print(i, j, radii[i, j])  # for debugging
                    if thetas[i, j] < 0.0:
                        thetas[i, j] += 2*np.pi  # transforms it from -np.pi min to 2*np.pi max, makes angles continuous
            # Recalibration of special cropping cases to preserve counting of radii from range [0, 1] - unit circle
            if recalibrate_radii:
                radii /= recalibration_coeff
        else:
            raise ValueError("Dimensions of provided image not equal to 2")
    return cropped_logic_mask, (radii, thetas)


def fit_zernikes(deformations_image: np.ndarray, cropped_demormations_mask: np.ndarray,
                 polar_coordinates: tuple, polynomials: tuple) -> np.ndarray:
    # Preparing fitting values - convert 2D images to 1D vectors
    rows, cols = cropped_demormations_mask.shape
    cropped_deformations_vector = np.zeros(shape=(rows*cols,), dtype=deformations_image.dtype)
    cropped_radii_vector = np.zeros(shape=(rows*cols,)); cropped_thetas_vector = np.zeros(shape=(rows*cols,))
    vector_index = 0
    # Resave cropped deformations from input parameters in vectors
    for i in range(rows):
        for j in range(cols):
            if cropped_demormations_mask[i, j] - 1 >= 0:
                cropped_deformations_vector[vector_index] = deformations_image[i, j]
                cropped_radii_vector[vector_index] = polar_coordinates[0][i, j]
                cropped_thetas_vector[vector_index] = polar_coordinates[1][i, j]
                vector_index += 1
    # Cut out all not resaved deformations and their coordinates
    cropped_deformations_vector = cropped_deformations_vector[0:vector_index]
    cropped_radii_vector = cropped_radii_vector[0:vector_index]
    cropped_thetas_vector = cropped_thetas_vector[0:vector_index]
    # Calculate polynomials values in the unit circle defined by polar coordinates
    zernike_values = np.zeros(shape=(vector_index, len(polynomials)))
    try:
        polynomials[0].polynomial_value(0.1, np.pi/4)
    except AttributeError:
        raise AttributeError("1st provided polynomial in polynomials input parameter isn't instance of ZernPol class")
    for j in range(len(polynomials)):
        if polynomials.get_mn_orders()[0] == 0 and polynomials.get_mn_orders()[1] == 0:
            continue  # do not calculate piston value
        else:
            zernike_values[:, j] = polynomials[j].polynomial_value(cropped_radii_vector, cropped_thetas_vector)
    return cropped_deformations_vector, cropped_radii_vector


# %% Basic tests
if __name__ == "__main__":
    deformations_sample = 4*np.ones(shape=(3, 4), dtype="uint8")
    crop_deforms1, polar_coordinates1 = crop_deformations(deformations_sample)
    # crop_deforms_vector1, cropped_radii_vector1 = fit_zernikes(deformations_sample, crop_deforms1, polar_coordinates1,
    #                                                            (1, 2, 3))
    deformations_sample = np.ones(shape=(4, 4), dtype="int16")
    crop_deforms2, polar_coordinates2 = crop_deformations(deformations_sample)
    crop_deforms2a, polar_coordinates2a = crop_deformations(deformations_sample, strict_border=True)
    deformations_sample = np.ones(shape=(6, 6))
    crop_deforms3, polar_coordinates = crop_deformations(deformations_sample)
    deformations_sample = np.ones(shape=(3, 3))
    crop_deforms4, polar_coordinates = crop_deformations(deformations_sample)
    deformations_sample = np.ones(shape=(5, 5))
    crop_deforms5, polar_coordinates = crop_deformations(deformations_sample)
    deformations_sample = np.ones(shape=(4, 6))
    crop_deforms6, polar_coordinates6 = crop_deformations(deformations_sample)
    crop_deforms6a, polar_coordinates6a = crop_deformations(deformations_sample, strict_border=True)
    deformations_sample = np.ones(shape=(7, 6))
    crop_deforms7, polar_coordinates7 = crop_deformations(deformations_sample)
    crop_deforms7a, polar_coordinates7a = crop_deformations(deformations_sample, strict_border=True)
    deformations_sample = np.ones(shape=(5, 5))
    crop_deforms81, polar_coordinates = crop_deformations(deformations_sample)
    crop_deforms82, polar_coordinates = crop_deformations(deformations_sample, crop_radius=0.5)
    thetas_grads = polar_coordinates[1]*(180.0/np.pi)
