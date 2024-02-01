# -*- coding: utf-8 -*-
"""
Testing the distance calculation over the meshgrid.

@author: sklykov
"""
import numpy as np


def distance_f(i_px, j_px, i_centre, j_centre):
    return np.round(np.sqrt(np.power(i_px - i_centre, 2) + np.power(j_px - j_centre, 2)), 6)


x = np.arange(start=0, stop=3, step=1)
y = np.arange(start=0, stop=3, step=1)
msh = np.meshgrid(x, y)
distances = distance_f(msh[0], msh[1], 0, 0)
