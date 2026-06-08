# -*- coding: utf-8 -*-
"""
Check the normalization consistency.

@author: Sergei Klykov

@licence: MIT, @year: 2026

"""
# %% Global imports
import numpy as np

try:
    from zernpy import __version__, generate_polynomials
    zernpy_installed = True
except ModuleNotFoundError:
    print("Install 'zernpy' library")
    zernpy_installed = False

# %% Script functionality
if __name__ == '__main__' and zernpy_installed:
    print("zernpy imported version:", __version__)
    polynomials = list(generate_polynomials(max_order=4)); polynomials.pop(0)
    coeff = 1.0  # universal coefficient
    for polynomial in polynomials:
        zern_surface = polynomial.gen_zernikes_surface([1.0], [polynomial])
        phase_profile = zern_surface.ZernSurf
        rho = zern_surface.R[:, None]  # shape conversion: (101, ) -> (101, 1)
        weights = np.ones_like(phase_profile)*rho  # weights recalculated for the whole phase profile on polar coordinates
        # idea below - calculate double integral on polar coordinates on Z**2*r_dr_dphi/r_dr_dphi
        rms = round(np.sqrt(np.sum((phase_profile**2)*weights)/np.sum(weights)), 2)   # now the coefficient is equal to the calculated RMS
        pv = round(np.max(phase_profile) - np.min(phase_profile), 3)  # simple definition
        print(f"{polynomial.get_mn_orders()}: \t", f"RMS: {rms} \t", f"Peak-to-Valley: {pv}")
        # plot of some profile for illustration
        if polynomial.get_mn_orders() == (0, 4):
            polynomial.plot_zernikes_surface(zern_surface)
