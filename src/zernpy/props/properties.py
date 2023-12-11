# -*- coding: utf-8 -*-
"""
Some properties of Zernike polynomials, which better to store in the folder and not directly load to each instance.

@author: Sergei Klykov
@licence: MIT

"""
# Names of Zernike polynomials recorded as dictionary values with keys as (m, n) orders

# References there names are collected from:
# [1] Up to 4th order: Wiki article: https://en.wikipedia.org/wiki/Zernike_polynomials
# [2] 5th order names: from the website https://www.telescope-optics.net/monochromatic_eye_aberrations.htm
# 6th order - 7th order: my guess about the naming

polynomial_names: dict = {
    (0, 0): "Piston",
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
    (0, 0): "Piston",
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

# Warning messages for special cases - usage of exact eq-s with orders higher than 40 (pol. value) or 38 (derivatives)
warn_mess_r_long = (" - that is the highest empirical order (though I found on tests) guarantees"
                    + " stable calculation using the exact equation. Instability emerges because of"
                    + " high integer values produced by used in the exact eq. factorials. \n"
                    + "So, the zero array or float will be returned as the result. "
                    + "Use this method with recursive calculation instead.")

warn_mess_dr_long = (" - that is the highest empirical order (though I found on tests) guarantees"
                     + " stable calculation using the exact equation for derivatives of radial polynomials."
                     + " Instability emerges because of high integer values produced by used in the exact"
                     + "  eq. factorials multiplied additionally by derivation result. \n"
                     + "So, the zero array or float will be returned as the result. "
                     + "Use this method with recursive calculation of derivatives instead.")

# Warning about usage of recursive calculations for high order polynomials
warn_mess_slow_calc = " has actually high radial order, recursive calculations will be slow"
