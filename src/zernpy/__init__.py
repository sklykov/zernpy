# -*- coding: utf-8 -*-
"""
The "zernpy" package is intended for initialization and calculation attributes / properties of Zernike polynomials.

@author: Sergei Klykov

@licence: MIT, @year: 2025

"""

__version__ = "0.1.1"  # Straightforward way of specifying package version and including it to the package attributes

# Univesal logic for making all main classes and functions available after calling 'from project import *'
from .zernikepol import ZernPol, generate_polynomials, fit_polynomials, generate_random_phases, fit_polynomials_vectors, generate_phases_image
from .zernpsf import ZernPSF, force_get_psf_compilation

__all__ = ["ZernPol", "generate_polynomials", "fit_polynomials", "generate_random_phases", "fit_polynomials_vectors",
           "generate_phases_image", "ZernPSF", "force_get_psf_compilation"]
