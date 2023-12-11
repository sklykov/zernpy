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
def crop_phases_img(phases_image: np.ndarray, crop_radius: float = 1.0, suppress_warns: bool = False,
                    strict_border: bool = False) -> tuple:
    """
    Cropping the circle from input 2D image containing phases with cartesian coordinates.

    This function also calculates polar coordinates, normalizing them to the range [0, 1.0] for radii
    and to the range [0.0, 2.0*pi] for theta angles. The polar coordinates are returned by this function.
    Crop radius allows cropping out pixels from selected unit circle.

    Parameters
    ----------
    phases_image : np.ndarray
        2D image with recorded phases which should be approximated by the sum of Zernike polynomials.
    crop_radius : float, optional
        Allow cropping pixel from range [0.5, 1.0], where 1.0 corresponds to radius of circle = min image size.
        The default is 1.0.
    suppress_warns : bool, optional
        Flag for suppress warnings about the provided 2D image sizes. The default is False.
    strict_border : bool, optional
        Flag for controlling how the border pixels (on the circle radius) are treated: strictly or less strict for
        allowing more pixels to be treated as belonged to a cropped circle. The default is False.

    Raises
    ------
    ValueError
        If input parameters lay outside specific ranges or provided with unexpected types.

    Returns
    -------
    tuple
        Consisting of:
        0th el.: cropped 2D logic mask with values: 0 - laying outside cropped circle pixels, 1 - laying inside,
        1st el.: tuple with cropped phases composed in 1D vector, corresponding polar coordinates,
        calculated radii and angles (thetas).

    """
    __warn_message = ""  # holder for warning message below
    cropped_logic_mask = None  # initial value for returning it in the case of some inconsistency
    # Sanity checks of provided crop radius
    if not isinstance(crop_radius, float):
        crop_radius = float(crop_radius)
    crop_radius = np.round(crop_radius, 4)  # rounding for performing exact comparisons
    if not 0.5 <= crop_radius <= 1.0:
        raise ValueError("Provided radius for cropping is not in the range [0.5, 1.0]")
    # Sanity checks of provided deformation image and cropping the pixels out
    if not isinstance(phases_image, np.ndarray):
        raise ValueError("Please provide the phases image as the numpy ndarray for proper method calls")
    else:
        if phases_image.ndim == 2:
            # Initialize returning values with default shapes
            rows, cols = phases_image.shape; cropped_logic_mask = np.zeros(shape=phases_image.shape, dtype="uint8")
            cropped_phases_vector = np.zeros(shape=(rows*cols,), dtype=phases_image.dtype)
            cropped_radii_vector = np.zeros(shape=(rows*cols,)); cropped_thetas_vector = np.zeros(shape=(rows*cols,))
            # Check input image shape
            if rows != cols:
                __warn_message = "Phases image isn't square, results of fitting could be ambiguous"
                img_min_size = min(rows, cols); img_max_size = max(rows, cols)
                if not suppress_warns:
                    warnings.warn(__warn_message)
            else:
                img_min_size = rows; img_max_size = rows
            if rows % 2 == 0 or cols % 2 == 0:
                __warn_message = ("Phases image provided with even rows or columns, "
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
                radius_epsilon = 0.055  # defined for matrix(4, 4) for including border pixels for not strict cropping
                __warn_message = ("\n Function called with flag that makes not strict cropping border of circle. \n"
                                  + "Phases image provided with even rows and columns"
                                  + " or with 1 pixel difference between them. \n"
                                  + "Note that additional border pixels are included to crop.")
                if not suppress_warns:
                    warnings.warn(__warn_message)
            # Calculate polar coordinates of pixels
            recalibrate_radii = False; recalibration_coeff = 1.0; vector_index = 0
            # Cropping phases, collecting data on cropping process
            for i in range(rows):
                for j in range(cols):
                    radius = np.round(np.sqrt(np.power((i - center_row), 2) + np.power((j - center_col), 2))/r, 4)
                    theta = np.arctan2(center_row - i, j - center_col)  # making it counterclockwise
                    if radius - crop_radius < radius_epsilon:
                        cropped_logic_mask[i, j] += 1  # should be converted to the actual type of image on multiplication
                        cropped_phases_vector[vector_index] = phases_image[i, j]
                        cropped_radii_vector[vector_index] = radius
                        if theta < 0.0:
                            theta += 2.0*np.pi  # transforms it from -np.pi min to 2*np.pi max, makes angles continuous
                        cropped_thetas_vector[vector_index] = theta
                        vector_index += 1
                        # Checking that recalibration needed - for cropping only pixels formally from unit radius circle
                        if not strict_border and radius - crop_radius > 0.001:
                            recalibrate_radii = True
                            if recalibration_coeff < radius:
                                recalibration_coeff = radius
            # Cut out all not transformed phases and their coordinates
            cropped_phases_vector = cropped_phases_vector[0:vector_index]
            cropped_radii_vector = cropped_radii_vector[0:vector_index]
            cropped_thetas_vector = cropped_thetas_vector[0:vector_index]
            # Recalibration of special cropping cases to preserve counting of radii from range [0, 1] - unit circle
            if recalibrate_radii:
                cropped_radii_vector /= recalibration_coeff; cropped_radii_vector = np.round(cropped_radii_vector, 4)
            # Additional check of border condition for a unit circle: the pixel has radius 1.001 but passed by cropping procedure
            if 1.0 < np.max(cropped_radii_vector) < 1.002:
                cropped_radii_vector /= np.max(cropped_radii_vector)
                cropped_radii_vector = np.round(cropped_radii_vector, 4)
        else:
            raise ValueError("Dimensions of provided image not equal to 2")
    return cropped_logic_mask, (cropped_phases_vector, cropped_radii_vector, cropped_thetas_vector)


def fit_zernikes(phases_coordinates_vectors: tuple, polynomials: tuple) -> np.ndarray:
    """
    Fit provided tuple with polynomials (instances of ZernPol class) to the 2D image with phases (e.g., recorded on the interferometer).

    Parameters
    ----------
    phases_coordinates_vectors : tuple
        Results provided by crop_phases_img(...) function composing the cropped phases along with polar coordinates.
        In general, this tuple contains 3 numpy arrays (vectors = 1D arrays) with phases and corresponding polar coordinates.
    polynomials : tuple
        Initialized tuple with instances of the ZernPol class that effectively represents target set of Zernike polynomials for fitting.

    Raises
    ------
    AttributeError
        If the input tuple with polynomials composed not from instances of ZernPol class.
    ValueError
        If the length of the tuple with polynomials is zero or if provided tuple contains repeated polynomials.
    numpy.linalg.LinAlgError
        If the fit procedure doesn't converge (namely, np.linalg.lstsq(...) function doesn't converge).

    Returns
    -------
    zernike_coefficients : numpy.ndarray
        1D array with the amplitudes of fitted Zernike polynomials specified by the input tuple.

    """
    # Unpacking provided vectors
    cropped_phases_vector = phases_coordinates_vectors[0]; cropped_radii_vector = phases_coordinates_vectors[1]
    cropped_thetas_vector = phases_coordinates_vectors[2]
    vector_length, = cropped_phases_vector.shape; zernike_coefficients = None
    # Calculate polynomials values in the unit circle defined by polar coordinates
    zernike_values = np.zeros(shape=(vector_length, len(polynomials)))
    # Checking that all polynomials are unique
    if not len(polynomials) == 0:
        provided_orders = []
        for polynomial in polynomials:
            provided_orders.append(polynomial.get_mn_orders())
        set_provided_orders = set(provided_orders)  # filter out if there is any repeated polynomials provided
        if len(set_provided_orders) != len(provided_orders):
            print(set_provided_orders, provided_orders)
            raise ValueError("Provided repeated polynomials")
    else:
        raise ValueError("Provided zero length tuple with polynomials")
    # Checking the length of provided tuple with polynomials
    if len(polynomials) > 0:
        # Calculation of Zernike polynomials values in the polar coordinates composed in radii, thetas vectors
        for j in range(len(polynomials)):
            if cropped_radii_vector.shape[0] == cropped_thetas_vector.shape[0]:
                zernike_values[:, j] = polynomials[j].polynomial_value(cropped_radii_vector, cropped_thetas_vector)
            else:
                thetas_length = cropped_thetas_vector.shape[0]
                for i in range(cropped_radii_vector.shape[0]):
                    zernike_values[i*thetas_length:(i+1)*thetas_length, j] = polynomials[j].polynomial_value(cropped_radii_vector[i],
                                                                                                             cropped_thetas_vector)
        # Fitting procedure of calculated polynomials values to the provided phases (deformations)
        zernike_coefficients = np.linalg.lstsq(zernike_values, cropped_phases_vector, rcond=None)
        zernike_coefficients = zernike_coefficients[0]  # unpacking fitting results
    else:
        raise ValueError("There isn't any polynomials provided")
    return zernike_coefficients


# %% Basic tests
if __name__ == "__main__":
    phases_sample = 4*np.ones(shape=(3, 4), dtype="uint8")
    crop_deforms1, polar_coordinates1 = crop_phases_img(phases_sample)
    phases_sample = np.ones(shape=(4, 4), dtype="int16")
    crop_deforms2, polar_coordinates2 = crop_phases_img(phases_sample)
    crop_deforms2a, polar_coordinates2a = crop_phases_img(phases_sample, strict_border=True)
    phases_sample = np.ones(shape=(6, 6))
    crop_deforms3, polar_coordinates = crop_phases_img(phases_sample)
    phases_sample = np.ones(shape=(3, 3))
    crop_deforms4, polar_coordinates = crop_phases_img(phases_sample)
    phases_sample = np.ones(shape=(5, 5))
    crop_deforms5, polar_coordinates = crop_phases_img(phases_sample)
    phases_sample = np.ones(shape=(4, 6))
    crop_deforms6, polar_coordinates6 = crop_phases_img(phases_sample)
    crop_deforms6a, polar_coordinates6a = crop_phases_img(phases_sample, strict_border=True)
    phases_sample = np.ones(shape=(7, 6))
    crop_deforms7, polar_coordinates7 = crop_phases_img(phases_sample)
    crop_deforms7a, polar_coordinates7a = crop_phases_img(phases_sample, strict_border=True)
    phases_sample = np.ones(shape=(5, 5))
    crop_deforms81, polar_coordinates = crop_phases_img(phases_sample)
    crop_deforms82, polar_coordinates = crop_phases_img(phases_sample, crop_radius=0.5)
    thetas_grads = polar_coordinates[1]*(180.0/np.pi)
