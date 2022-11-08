# -*- coding: utf-8 -*-
"""
Collection of Zernike polynomial calculation methods.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np


# %% Function definitions
def normalization_factor(zernike_pol) -> float:
    """
    Calculate normalization factor according to the reference (N(m ,n)).

    Parameters
    ----------
    zernike_pol : ZernPol
        ZernPol - class instance of the calling class.

    Reference
    ---------
    Shakibaei B.H., Paramesran R. (2013)

    Returns
    -------
    float
        Normalization factor, depending only on Zernike type.

    """
    (m, n) = zernike_pol.get_polynomial_orders()
    if m == 0:
        return np.sqrt(n + 1)
    else:
        return np.sqrt(2*(n + 1))
