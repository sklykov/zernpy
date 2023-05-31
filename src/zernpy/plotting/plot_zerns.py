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
def plot_sum_fig(polynomials_sum: np.ndarray, r: np.ndarray, theta: np.ndarray, title: str = "",
                 color_map: str = "coolwarm"):
    """
    Plot Zernike polynomials sum (or single polynomial) on a mesh of polar coordinates (r, theta) as a 2D polar plot.

    Parameters
    ----------
    polynomials_sum : numpy.ndarray
        Values of Zernike polynomials sum or single one on a mesh of polar coordinates.
    r : numpy.ndarray
        Vector with radiuses as polar coordinates.
    theta : numpy.ndarray
        Vector with theta angles as polar coordinates.
    title : str, optional
        Title for representing on the Figure, useful for the single Zernike polynomial plotting. The default is "".
    color_map : str, optional
        Color map of the polar plot, common values for representation: coolwarm, jet, turbo, rainbow.
        As alternative - perceptually equal color maps: viridis, plasma. The default is "coolwarm".
        Note that rainbow, jet, turbo - not perceptually equal color maps.

    Returns
    -------
    None.

    """
    plt.ioff()  # blocks the call of plt.show() for running this script in some IDEs (e.g., PyCharm)
    plt.figure(figsize=(4, 4))  # since the figure has the circular shape, better draw it on equal box
    axes = plt.axes(projection='polar'); axes.grid(False)
    plt.pcolormesh(theta, r, polynomials_sum, cmap=color_map, shading='nearest')  # fast draw of 2D polar surface
    if isinstance(title, str) and len(title) > 0:
        plt.title(title)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    plt.axis('off'); plt.tight_layout()
    plt.show()  # waits in non-interactive mode for closing the plot window


def plot_sum_fig_3d(polynomials_sum: np.ndarray, r: np.ndarray, theta: np.ndarray, color_map: str = "coolwarm"):
    """
    Plot Zernike polynomials sum (or single polynomial) on a mesh of polar coordinates (r, theta) as a 3D surface plot.

    Parameters
    ----------
    polynomials_sum : numpy.ndarray
        Values of Zernike polynomials sum or single one on a mesh of polar coordinates.
    r : numpy.ndarray
        Vector with radiuses as polar coordinates.
    theta : numpy.ndarray
        Vector with theta angles as polar coordinates.
    color_map : str, optional
        Color map of the polar plot, common values for representation: coolwarm, jet, turbo, rainbow.
        As alternative - perceptually equal color maps: viridis, plasma. The default is "coolwarm".
        Note that rainbow, jet, turbo - not perceptually equal color maps.

    Returns
    -------
    None.

    """
    plt.ioff()  # blocks the call of plt.show() for running this script in some IDEs (e.g., PyCharm)
    figure = plt.figure(figsize=(5.4, 5.4))  # since the figure has the circular shape, better draw it on equal box
    axes = figure.add_subplot(projection='3d'); Thetas, Rs = np.meshgrid(theta, r)  # Note the order the meshgrid coordinates!
    X, Y = Rs*np.cos(Thetas), Rs*np.sin(Thetas)  # convert polar coordinates to the cartesian system
    axes.plot_surface(X, Y, polynomials_sum, cmap=color_map)
    # plt.axis('off');
    plt.tight_layout(); plt.show()


def subplot_sum_on_fig(figure: plt.Figure, polynomials_sum: np.ndarray, r: np.ndarray, theta: np.ndarray,
                       show_range_colorbar: bool = False, color_map: str = "coolwarm") -> plt.Figure:
    """
    Plot on the provided 2D figure the sum of Zernike polynomials calculated based on a mesh of polar coordinates r, theta.

    Parameters
    ----------
    figure : plt.Figure
        Figure() class there the plotting will be done, previous plot will be cleared.
    polynomials_sum : numpy.ndarray
        Sum of Zernike polynomials (could be single Zernike polynomial).
    r : numpy.ndarray
        Vector with radiuses as polar coordinates.
    theta : numpy.ndarray
        Vector with theta angles as polar coordinates.
    show_range_colorbar : bool, optional
        Flag for showing range of provided values as the colorbar on the figure. The default is False.
    color_map : str, optional
        Color map of the polar plot, common values for representation: coolwarm, jet, turbo, rainbow.
        As alternative - perceptually equal color maps: viridis, plasma. The default is "coolwarm".
        Note that rainbow, jet, turbo - not perceptually equal color maps.

    Returns
    -------
    figure : plt.Figure
        Figure() class there the Zernike polynomials sum plotted.

    """
    figure.clear()  # clearing the previous plotted figure
    axes = figure.add_subplot(projection='polar')  # axes - the handle for drawing functions
    axes.grid(False)  # demanded by pcolormesh function, if not called - deprecation warning
    # plot the colour map by using the Zernikes values according to Theta, R coordinates
    im = axes.pcolormesh(theta, r, polynomials_sum, cmap=color_map, shading='nearest')
    axes.axis('off')  # off polar coordinate axes
    # below: shows the colour bar with shown on image amplitudes
    if show_range_colorbar:
        figure.colorbar(im, ax=axes)
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1)
    figure.tight_layout()
    return figure


def subplot_sum_on_fig_3d(figure: plt.Figure, polynomials_sum: np.ndarray, r: np.ndarray, theta: np.ndarray,
                          show_range_colorbar: bool = False, color_map: str = "coolwarm") -> plt.Figure:
    """
    Plot on the provided 3D figure the sum of Zernike polynomials calculated based on a mesh of polar coordinates r, theta.

    Parameters
    ----------
    figure : plt.Figure
        Figure() class there the plotting will be done, previous plot will be cleared.
    polynomials_sum : numpy.ndarray
        Sum of Zernike polynomials (could be single Zernike polynomial).
    r : numpy.ndarray
        Vector with radiuses as polar coordinates.
    theta : numpy.ndarray
        Vector with theta angles as polar coordinates.
    show_range_colorbar : bool, optional
        Flag for showing range of provided values as the colorbar on the figure. The default is False.
    color_map : str, optional
        Color map of the polar plot, common values for representation: coolwarm, jet, turbo, rainbow.
        As alternative - perceptually equal color maps: viridis, plasma. The default is "coolwarm".
        Note that rainbow, jet, turbo - not perceptually equal color maps.

    Returns
    -------
    figure : plt.Figure
        Figure() class there the Zernike polynomials sum plotted.

    """
    figure.clear()  # clearing the previous plotted figure
    axes = figure.add_subplot(projection='3d'); Thetas, Rs = np.meshgrid(theta, r)  # Note the order the meshgrid coordinates!
    X, Y = Rs*np.cos(Thetas), Rs*np.sin(Thetas)  # convert polar coordinates to the cartesian system
    im = axes.plot_surface(X, Y, polynomials_sum, cmap=color_map)
    # below: shows the colour bar with shown on image amplitudes
    if show_range_colorbar:
        figure.colorbar(im, ax=axes)
    figure.subplots_adjust(left=0, bottom=0, right=1, top=1)
    figure.tight_layout()
    return figure


# %% Run as a script
if __name__ == "__main__":
    pass
