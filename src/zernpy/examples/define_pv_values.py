# -*- coding: utf-8 -*-
"""
Example of 'zernpy' script functionality usage.

@author: Sergei Klykov
@licence: MIT, @year: 2024
"""
# %% Global imports
import numpy as np
try:
    from zernpy import generate_polynomials
    zernpy_installed = True
except ModuleNotFoundError:
    print("Install 'zernpy' library")
    zernpy_installed = False

# %% Script functionality
if __name__ == '__main__':
    if zernpy_installed:
        # Define the coefficients for predefined peak-to-valley values
        polynomials_list = generate_polynomials(max_order=7)
        pv_values = [8.0, 8.0,
                     5.1, 5.0, 5.1,
                     3.6, 3.0, 3.0, 3.6,
                     2.0, 1.8, 1.5, 1.8, 2.0,
                     1.3, 1.1, 1.1, 1.1, 1.1, 1.3,
                     1.0, 0.9, 0.9, 0.7, 0.9, 0.9, 1.0,
                     0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7]
        found_coeffs = []
        for j, polynomial in enumerate(polynomials_list):
            if j == 0:
                # print(polynomial.get_polynomial_name())
                continue
            else:
                if j-1 < len(pv_values):
                    for i in range(150):
                        ampl = 0.02*(i+1); zern_surface_tuple2 = polynomial.gen_zernikes_surface([ampl], [polynomial])
                        surface2 = zern_surface_tuple2.ZernSurf
                        pv_value2 = round(np.max(surface2) - np.min(surface2), 1)
                        if pv_value2 > pv_values[j-1]:
                            found_coeffs.append(round(ampl, 1))
                            # print(polynomial.get_polynomial_name(), "P-V value:", pv_value2)
                            print(polynomial.get_mn_orders(), " \t Coefficient:", round(ampl, 1))
                            break
        # Example of peak-to-valley value calculation for including it into README
        print("************************************************************")
        for polynomial in polynomials_list:
            coeff = 1.0; zern_surface_tuple = polynomial.gen_zernikes_surface([coeff], [polynomial])
            surface = zern_surface_tuple.ZernSurf
            pv_value = round(np.max(surface) - np.min(surface), 3)
            print(polynomial.get_mn_orders(), " \t Peak-Valley:", pv_value)
