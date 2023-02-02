# -*- coding: utf-8 -*-
"""
Test the implemented fitting functions for modules "fit_zernike_pols" and "zernikepol" by using the pytest library.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in the command line.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np

# %% Imports from modules
if __name__ != "__main__":
    from ..zernikepol import generate_random_phases, fit_polynomials


# %% Test functions
def test_random_fitting():
    """
    Test difference between randomly generated Zernike polynomials coefficients and fitted ones.

    Returns
    -------
    None.

    """
    # Test generation of random phase profile abs. max value
    for _ in range(4):
        pl1, random_amplitudes, pl2 = generate_random_phases(max_order=3, img_height=11, img_width=11)
        max_src_ampl = round(np.max(np.abs(random_amplitudes)), 6)
        assert max_src_ampl > 0.08, f"Random generated profile max abs. ampl.:{max_src_ampl} < 0.08, that is error, {random_amplitudes}"
    # Test absolute differences, maybe, maximum allowed percentages have been selected unreasonably
    # The parameter "mdp" regulates the maximal percentage as abs. difference between fitted and randomly generated
    # amplitudes divided by the max abs. amplitude of randomly generated amplitude
    for _ in range(2):  # run test several times
        for i in range(4, 8, 1):
            if i == 4:
                height = 147; width = 147; crop_r = 1.0; strict_border = False; mdp = 15.0; stop_warns = False
            elif i == 5:
                height = 160; width = 160; crop_r = 0.98; strict_border = False; mdp = 25.0; stop_warns = True
            elif i == 6:
                height = 201; width = 204; crop_r = 0.95; strict_border = True; mdp = 20.0; stop_warns = True
            elif i == 7:
                # The allowed percentage below is huge because the cropping radius is equal to 80% of image size
                height = 212; width = 214; strict_border = False; crop_r = 0.8; mdp = 33.0; stop_warns = True
            random_phases_image, random_amplitudes, polynomials_tuple = generate_random_phases(max_order=i, img_height=height,
                                                                                               img_width=width)
            fitted_amplitudes, _ = fit_polynomials(random_phases_image, polynomials_tuple, suppress_warnings=stop_warns,
                                                   crop_radius=crop_r, strict_circle_border=strict_border)
            abs_diff_amplitudes = np.abs(fitted_amplitudes - random_amplitudes)
            max_abs_diff = round(np.max(abs_diff_amplitudes), 6); max_src_ampl = round(np.max(np.abs(random_amplitudes)), 6)
            # max_ampl = max(np.max(np.abs(random_amplitudes)), np.max(np.abs(fitted_amplitudes)))  # max abs amplitude
            max_abs_diff_percent = round(100.0*(max_abs_diff/max_src_ampl), 0)
            rmse = np.sqrt(np.mean(np.square(random_amplitudes - fitted_amplitudes)))  # root mean square error
            rmse_percentage = round(100.0*(rmse / np.max(np.abs(random_amplitudes))), 0)  # calculated as % from max abs. amplitudes
            assert max_abs_diff_percent <= mdp, ("Max difference between random and fitted Zernike amplitudes "
                                                 + f"in % of maximum random generated amplitude: {max_abs_diff_percent},"
                                                 + f" for parameters: max src ampl: {max_src_ampl}, max diff: {max_abs_diff}")
            assert rmse_percentage <= (mdp//2) + 1, ("RMSE between fitted and randomly generated polynomials:"
                                                     + f" {rmse_percentage} > assumed value {(mdp//2) + 1}")
    # Test the sign of fitted and randomly generated amplitudes
    for i in range(3):  # run tests 2 times
        random_phases_image, random_amplitudes, polynomials_tuple = generate_random_phases(img_height=101, img_width=101)
        fitted_amplitudes, _ = fit_polynomials(random_phases_image, polynomials_tuple)
        max_fit_ampl = np.max(fitted_amplitudes); max_src_ampl = np.max(random_amplitudes)
        min_fit_ampl = np.min(fitted_amplitudes); min_src_ampl = np.min(random_amplitudes)
        if abs(max_src_ampl) > 0.08 or abs(min_src_ampl) > 0.08:
            if abs(max_src_ampl) < abs(min_src_ampl):
                src_ampl = min_src_ampl; fit_ampl = min_fit_ampl
            else:
                src_ampl = max_src_ampl; fit_ampl = max_fit_ampl
            same_sign = False
            if src_ampl < 0.0 and fit_ampl < 0.0:
                same_sign = True
            elif src_ampl > 0.0 and fit_ampl > 0.0:
                same_sign = True
            assert same_sign, ("\n Fitted and source amplitudes have abs. maximum values with different"
                               + f" signs: source ampl.: {src_ampl}, fitted ampl.: {fit_ampl}")
