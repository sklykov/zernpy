# -*- coding: utf-8 -*-
"""
Plotting the Zernike polynomials values over unit circle.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
import matplotlib.pyplot as plt

# %% Module parameters
__docformat__ = "numpydoc"


# %% Function definitions
def plot_sum_fig(polynomials_sum: np.ndarray, r: np.ndarray, theta: np.ndarray, title: str = ""):
    if not plt.isinteractive:
        plt.ion()
    else:
        plt.close("all")  # close all opened figures
    plt.figure(figsize=(4, 4))  # since the figure has the circular shape, better draw it on equal box
    axes = plt.axes(projection='polar'); axes.grid(False)
    plt.pcolormesh(theta, r, polynomials_sum, cmap=plt.cm.coolwarm, shading='nearest')  # fast draw of 2D polar surface
    if isinstance(title, str) and len(title) > 0:
        plt.title(title)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    plt.axis('off'); plt.tight_layout()


# %% Run as a script
if __name__ == "__main__":
    pass
