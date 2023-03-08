# -*- coding: utf-8 -*-
"""
Collection of Zernike polynomial calculation methods.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
import math
import time
from pathlib import Path
# from decimal import Decimal

# %% Local (package-scoped) imports
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from find_pols_coeffs import find_coeffs_orders, find_coeffs_orders_dr
else:
    from .find_pols_coeffs import find_coeffs_orders, find_coeffs_orders_dr

# %% Module parameters
__docformat__ = "numpydoc"
# Below - empirically found max radial order allowing stable calculation using recursive coefficients defining
# or using exact equation involving factorials calculation. Approximately, from 46th order the exact equation start
# to violate condition that max of radial polynomial <= 1.0. From 40th order the difference between recurrence eq.
# the exact one start to be too significant
MAX_RADIAL_ORDER_COEFFS = 40
MAX_RADIAL_ORDER_COEFFS_dR = 38


# %% Calc. functions
def define_orders(zernike_pol) -> tuple:
    """
    Return orders as tuple (m, n) for using in the functions below.

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.

    Returns
    -------
    (m, n)
        Azimuthal, radial orders in a tuple.

    """
    # Get azimuthal and radial orders
    if isinstance(zernike_pol, tuple):
        (m, n) = zernike_pol
    else:
        (m, n) = zernike_pol.get_mn_orders()
    return m, n


def normalization_factor(zernike_pol) -> float:
    """
    Calculate normalization factor according to the reference (N(m ,n)).

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.

    References
    ----------
    Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)

    Returns
    -------
    float
        Normalization factor, depending only on Zernike type.

    """
    (m, n) = define_orders(zernike_pol)  # get orders
    # Calculation of the value according to Ref.
    if m == 0:
        return np.sqrt(n + 1)
    else:
        return np.sqrt(2*(n + 1))


def radial_polynomial(zernike_pol, r):
    """
    Calculate radial polynomial R(m, n) value for input r lying in the range [0, 1].

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    r : float or numpy.ndarray
        Radius from the unit circle, float or array of values on which the Zernike polynomial is calculated.

    References
    ----------
    [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)
    [2] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011)
    [3] Andersen T. B. "Efficient and robust recurrence relations for the Zernike
    circle polynomials and their derivatives in Cartesian coordinates" (2018)

    Returns
    -------
    float or numpy.ndarray
        Depending on the type of theta, return float or np.ndarray with calculated values of radial polynomial.

    """
    (m, n) = define_orders(zernike_pol)  # get orders
    # Radial polynomials defined as analytical equations for up to 10th order (check tables from [2])
    # 0th order
    if (m == 0) and (n == 0):
        if isinstance(r, float):
            return 1.0
        elif isinstance(r, np.ndarray):
            return np.ones(shape=r.shape)
    # 1st order
    elif ((m == -1) and (n == 1)) or ((m == 1) and (n == 1)):
        return r
    # 2nd order
    elif ((m == -2) and (n == 2)) or ((m == 2) and (n == 2)):
        return np.power(r, 2)  # r^2
    elif (m == 0) and (n == 2):
        return 2.0*np.power(r, 2) - 1.0  # 2r^2 - 1
    # 3rd order
    elif ((m == -3) and (n == 3)) or ((m == 3) and (n == 3)):
        return np.power(r, 3)  # r^3
    elif ((m == -1) and (n == 3)) or ((m == 1) and (n == 3)):
        return 3.0*np.power(r, 3) - 2.0*r  # 3r^3 - 2r
    # 4th order
    elif ((m == -4) and (n == 4)) or ((m == 4) and (n == 4)):
        return np.power(r, 4)  # r^4
    elif ((m == -2) and (n == 4)) or ((m == 2) and (n == 4)):
        return 4.0*np.power(r, 4) - 3.0*np.power(r, 2)  # 4r^4 - 3r^2
    elif (m == 0) and (n == 4):
        return 6.0*np.power(r, 4) - 6.0*np.power(r, 2) + 1.0  # 6r^4 - 6r^2 + 1
    # 5th order
    elif ((m == -5) and (n == 5)) or ((m == 5) and (n == 5)):
        return np.power(r, 5)  # r^5
    elif ((m == -3) and (n == 5)) or ((m == 3) and (n == 5)):
        return 5.0*np.power(r, 5) - 4.0*np.power(r, 3)  # 5r^5 - 4r^3
    elif ((m == -1) and (n == 5)) or ((m == 1) and (n == 5)):
        return 10.0*np.power(r, 5) - 12.0*np.power(r, 3) + 3.0*r  # 10r^5 - 12r^3 + 3r
    # 6th order
    elif ((m == -6) and (n == 6)) or ((m == 6) and (n == 6)):
        return np.power(r, 6)  # r^6
    elif ((m == -4) and (n == 6)) or ((m == 4) and (n == 6)):
        return 6.0*np.power(r, 6) - 5.0*np.power(r, 4)  # 6r^6 - 5r^4
    elif ((m == -2) and (n == 6)) or ((m == 2) and (n == 6)):
        return 15.0*np.power(r, 6) - 20.0*np.power(r, 4) + 6.0*np.power(r, 2)  # 15r^6 - 20r^4 + 6r^2
    elif (m == 0) and (n == 6):
        return 20.0*np.power(r, 6) - 30.0*np.power(r, 4) + 12.0*np.power(r, 2) - 1.0  # 20r^6 - 30r^4 + 12r^2 - 1
    # 7th order
    elif ((m == -7) and (n == 7)) or ((m == 7) and (n == 7)):
        return np.power(r, 7)  # r^7
    elif ((m == -5) and (n == 7)) or ((m == 5) and (n == 7)):
        return 7.0*np.power(r, 7) - 6.0*np.power(r, 5)  # 7r^7 - 6r^5
    elif ((m == -3) and (n == 7)) or ((m == 3) and (n == 7)):
        return 21.0*np.power(r, 7) - 30.0*np.power(r, 5) + 10.0*np.power(r, 3)  # 21r^7 - 30r^5 + 10r^3
    elif ((m == -1) and (n == 7)) or ((m == 1) and (n == 7)):
        return 35.0*np.power(r, 7) - 60.0*np.power(r, 5) + 30.0*np.power(r, 3) - 4.0*r  # 35r^7 - 60r^5 + 30r^3 - 4r
    # 8th order
    elif ((m == -6) and (n == 8)) or ((m == 6) and (n == 8)):
        return 8.0*np.power(r, 8) - 7.0*np.power(r, 6)  # 8r^8 - 7r^6
    elif ((m == -4) and (n == 8)) or ((m == 4) and (n == 8)):
        return 28.0*np.power(r, 8) - 42.0*np.power(r, 6) + 15.0*np.power(r, 4)  # 28r^8 - 42r^6 + 15r^4
    elif ((m == -2) and (n == 8)) or ((m == 2) and (n == 8)):
        # 56r^8 - 105r^6 + 60r^4 - 10r^2
        return 56.0*np.power(r, 8) - 105.0*np.power(r, 6) + 60.0*np.power(r, 4) - 10.0*np.power(r, 2)
    elif (m == 0) and (n == 8):
        # 70r^8 - 140r^6 + 90r^4 - 20r^2 + 1
        return 70.0*np.power(r, 8) - 140.0*np.power(r, 6) + 90.0*np.power(r, 4) - 20.0*np.power(r, 2) + 1.0
    # 9th order
    elif ((m == -7) and (n == 9)) or ((m == 7) and (n == 9)):
        return 9.0*np.power(r, 9) - 8.0*np.power(r, 7)  # 9r^9 - 8r^7
    elif ((m == -5) and (n == 9)) or ((m == 5) and (n == 9)):
        return 36.0*np.power(r, 9) - 56.0*np.power(r, 7) + 21.0*np.power(r, 5)  # 36r^9 - 56r^7 + 21r^5
    elif ((m == -3) and (n == 9)) or ((m == 3) and (n == 9)):
        # 84r^9 - 168r^7 + 105r^5 - 20r^3
        return 84.0*np.power(r, 9) - 168.0*np.power(r, 7) + 105.0*np.power(r, 5) - 20.0*np.power(r, 3)
    elif ((m == -1) and (n == 9)) or ((m == 1) and (n == 9)):
        # 126r^9 - 280r^7 + 210r^5 - 60r^3 + 5r
        return 126.0*np.power(r, 9) - 280.0*np.power(r, 7) + 210.0*np.power(r, 5) - 60.0*np.power(r, 3) + 5.0*r
    # 10th order
    elif ((m == -8) and (n == 10)) or ((m == 8) and (n == 10)):
        return 10.0*np.power(r, 10) - 9.0*np.power(r, 8)  # 10r^10 - 9r^8
    elif ((m == -6) and (n == 10)) or ((m == 6) and (n == 10)):
        return 45.0*np.power(r, 10) - 72.0*np.power(r, 8) + 28.0*np.power(r, 6)  # 45r^10 - 72r^8 + 28r^6
    elif ((m == -4) and (n == 10)) or ((m == 4) and (n == 10)):
        # 120r^10 - 252r^8 + 168r^6 - 35r^4
        return 120.0*np.power(r, 10) - 252.0*np.power(r, 8) + 168.0*np.power(r, 6) - 35.0*np.power(r, 4)
    elif ((m == -2) and (n == 10)) or ((m == 2) and (n == 10)):
        # 210r^10 - 504r^8 + 420r^6 - 140r^4 + 15r^2
        return 210.0*np.power(r, 10) - 504.0*np.power(r, 8) + 420.0*np.power(r, 6) - 140.0*np.power(r, 4) + 15.0*np.power(r, 2)
    elif (m == 0) and (n == 10):
        # 252r^10 - 630r^8 + 560r^6 - 210r^4 + 30r^2 - 1
        return 252.0*np.power(r, 10) - 630.0*np.power(r, 8) + 560.0*np.power(r, 6) - 210.0*np.power(r, 4) + 30.0*np.power(r, 2) - 1.0
    # Recurrence equations from the [1] and [3] for higher than 10 order polynomials
    elif n > 7 and abs(m) == n:  # simplified recurrence formula from [3]
        return np.power(r, n)
    elif n > 10 and m == 0:  # simplified recurrence formula from [3]
        return 2.0*r*radial_polynomial((1, n-1), r) - radial_polynomial((0, n-2), r)
    elif n > 10 and abs(m) == n-2:  # my guess about overall equation
        return float(n)*np.power(r, n) - float(n-1)*np.power(r, n-2)
    else:
        return (r*(radial_polynomial((abs(m-1), n-1), r) + radial_polynomial((m+1, n-1), r))
                - radial_polynomial((m, n-2), r))  # general recurrence formula from [1]


def radial_derivative(zernike_pol, r):
    """
    Calculate the derivative of radial polynomial dR(m, n)/dr value for input r lying in the range [0, 1].

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    r : float or numpy.ndarray
        Radius from the unit circle, float or array of values on which the Zernike polynomial is calculated.

    Returns
    -------
    float or numpy.ndarray
        Depending on the type of r, return float or np.ndarray with calculated values of radial polynomial derivative.

    """
    (m, n) = define_orders(zernike_pol)  # get orders
    # Derivatives of radial polynomials defined as analytical equations for up to 8th order (check function above)
    # 0th order
    if (m == 0) and (n == 0):
        if isinstance(r, float):
            return 0.0
        elif isinstance(r, np.ndarray):
            return np.zeros(shape=r.shape)
    # 1st order
    elif ((m == -1) and (n == 1)) or ((m == 1) and (n == 1)):
        if isinstance(r, float):
            return 1.0
        elif isinstance(r, np.ndarray):
            return np.ones(shape=r.shape)
    # 2nd order
    elif ((m == -2) and (n == 2)) or ((m == 2) and (n == 2)):
        return 2.0*r  # 2r
    elif (m == 0) and (n == 2):
        return 4.0*r  # 4r
    # 3rd order
    elif ((m == -3) and (n == 3)) or ((m == 3) and (n == 3)):
        return 3.0*np.power(r, 2)  # 3r^2
    elif ((m == -1) and (n == 3)) or ((m == 1) and (n == 3)):
        return 9.0*np.power(r, 2) - 2.0  # 9r^2 - 2
    # 4th order
    elif ((m == -4) and (n == 4)) or ((m == 4) and (n == 4)):
        return 4.0*np.power(r, 3)  # 4r^3
    elif ((m == -2) and (n == 4)) or ((m == 2) and (n == 4)):
        return 16.0*np.power(r, 3) - 6.0*r  # 16r^3 - 6r
    elif (m == 0) and (n == 4):
        return 24.0*np.power(r, 3) - 12.0*r  # 24r^3 - 12r
    # 5th order
    elif ((m == -5) and (n == 5)) or ((m == 5) and (n == 5)):
        return 5.0*np.power(r, 4)  # 5r^4
    elif ((m == -3) and (n == 5)) or ((m == 3) and (n == 5)):
        return 25.0*np.power(r, 4) - 12.0*np.power(r, 2)  # 25r^4 - 12r^2
    elif ((m == -1) and (n == 5)) or ((m == 1) and (n == 5)):
        return 50.0*np.power(r, 4) - 36.0*np.power(r, 2) + 3.0  # 50r^4 - 36r^2 + 3
    # 6th order
    elif ((m == -6) and (n == 6)) or ((m == 6) and (n == 6)):
        return 6.0*np.power(r, 5)  # 6r^5
    elif ((m == -4) and (n == 6)) or ((m == 4) and (n == 6)):
        return 36.0*np.power(r, 5) - 20.0*np.power(r, 3)  # 36r^5 - 20r^3
    elif ((m == -2) and (n == 6)) or ((m == 2) and (n == 6)):
        return 90.0*np.power(r, 5) - 80.0*np.power(r, 3) + 12.0*r  # 90r^5 - 80r^3 + 12r
    elif (m == 0) and (n == 6):
        return 120.0*np.power(r, 5) - 120.0*np.power(r, 3) + 24.0*r  # 120r^5 - 120r^3 + 24r
    # 7th order
    elif ((m == -7) and (n == 7)) or ((m == 7) and (n == 7)):
        return 7.0*np.power(r, 6)  # 7r^6
    elif ((m == -5) and (n == 7)) or ((m == 5) and (n == 7)):
        return 49.0*np.power(r, 6) - 30.0*np.power(r, 4)  # 49r^6 - 30r^4
    elif ((m == -3) and (n == 7)) or ((m == 3) and (n == 7)):
        return 147.0*np.power(r, 6) - 150.0*np.power(r, 4) + 30.0*np.power(r, 2)  # 147r^6 - 150r^4 + 30r^2
    elif ((m == -1) and (n == 7)) or ((m == 1) and (n == 7)):
        return 245.0*np.power(r, 6) - 300.0*np.power(r, 4) + 90.0*np.power(r, 2) - 4.0  # 245r^6 - 300r^4 + 90r^2 - 4
    # 8th order
    elif ((m == -6) and (n == 8)) or ((m == 6) and (n == 8)):
        return 64.0*np.power(r, 7) - 42.0*np.power(r, 5)  # 64r^7 - 42r^5
    elif ((m == -4) and (n == 8)) or ((m == 4) and (n == 8)):
        return 224.0*np.power(r, 7) - 252.0*np.power(r, 5) + 60.0*np.power(r, 3)  # 224r^7 - 252r^5 + 60r^3
    elif ((m == -2) and (n == 8)) or ((m == 2) and (n == 8)):
        # 448r^7 - 630r^5 + 240r^3 - 20r
        return 448.0*np.power(r, 7) - 630.0*np.power(r, 5) + 240.0*np.power(r, 3) - 20.0*r
    elif (m == 0) and (n == 8):
        # 560r^7 - 840r^5 + 360r^3 - 40r
        return 560.0*np.power(r, 7) - 840.0*np.power(r, 5) + 360.0*np.power(r, 3) - 40.0*r
    # 9th order
    elif ((m == -5) and (n == 9)) or ((m == 5) and (n == 9)):
        # 324r^8 - 392r^6 + 105r^4
        return 324.0*np.power(r, 8) - 392.0*np.power(r, 6) + 105.0*np.power(r, 4)
    elif ((m == -3) and (n == 9)) or ((m == 3) and (n == 9)):
        # 756r^8 - 1176r^6 + 525r^4 - 60r^2
        return 756.0*np.power(r, 8) - 1176.0*np.power(r, 6) + 525.0*np.power(r, 4) - 60.0*np.power(r, 2)
    elif ((m == -1) and (n == 9)) or ((m == 1) and (n == 9)):
        # 1134r^8 - 1960r^6 + 1050r^4 - 180r^2 + 5
        return 1134.0*np.power(r, 8) - 1960.0*np.power(r, 6) + 1050.0*np.power(r, 4) - 180.0*np.power(r, 2) + 5.0
    # 10th order
    elif ((m == -6) and (n == 10)) or ((m == 6) and (n == 10)):
        # 450r^9 - 576r^7 + 168r^5
        return 450.0*np.power(r, 9) - 576.0*np.power(r, 7) + 168.0*np.power(r, 5)
    elif ((m == -4) and (n == 10)) or ((m == 4) and (n == 10)):
        # 1200r^9 - 2016r^7 + 1008r^5 - 140r^3
        return 1200.0*np.power(r, 9) - 2016.0*np.power(r, 7) + 1008.0*np.power(r, 5) - 140.0*np.power(r, 3)
    elif ((m == -2) and (n == 10)) or ((m == 2) and (n == 10)):
        # 2100r^9 - 4032r^7 + 2520r^5 - 560r^3 + 30r
        return 2100.0*np.power(r, 9) - 4032.0*np.power(r, 7) + 2520.0*np.power(r, 5) - 560.0*np.power(r, 3) + 30.0*r
    elif (m == 0) and (n == 10):
        # 2520r^9 - 5040r^7 + 3360r^5 - 840r^3 + 60r
        return 2520.0*np.power(r, 9) - 5040.0*np.power(r, 7) + 3360.0*np.power(r, 5) - 840.0*np.power(r, 3) + 60.0*r
    # Recurrence equations from the [1] and [3] for higher than 10 order polynomials
    elif n > 7 and abs(m) == n:
        return float(n)*np.power(r, n-1)  # derivative from the simplified recurrence formula from [3]
    elif n > 8 and abs(m) == n-2:  # derivative from my guess about overall equation
        return float(n)*float(n)*np.power(r, n-1) - float(n-1)*float(n-2)*np.power(r, n-3)
    elif n > 10 and m == 0:
        # derivative from the simplified recurrence formula from [3]
        return 2.0*(radial_polynomial((1, n-1), r) + r*radial_derivative((1, n-1), r)) - radial_derivative((0, n-2), r)
    else:
        # derivative from the general recurrence formula from [1]
        return ((radial_polynomial((abs(m-1), n-1), r) + radial_polynomial((m+1, n-1), r))
                + r*(radial_derivative((abs(m-1), n-1), r) + radial_derivative((m+1, n-1), r))
                - radial_derivative((m, n-2), r))


def radial_polynomial_eq(zernike_pol, r):
    """
    Calculate the radial polynomial R(m, n) using exact equation from the Reference below.

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    r : float or numpy.ndarray
        Radius from the unit circle, float or array of values on which the Zernike polynomial is calculated.

    References
    ----------
    [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)

    Returns
    -------
    float or numpy.ndarray
        Depending on the type of theta, return float or numpy.ndarray with calculated values of radial polynomial.

    """
    if isinstance(r, float):
        value = 0.0
    elif isinstance(r, np.ndarray):
        value = np.zeros(r.shape)
    (m, n) = define_orders(zernike_pol)  # get orders
    for k in range(0, ((n - abs(m))//2) + 1):
        a = (n + m)//2; b = (n - m)//2
        value += ((-1)**k)*(((math.factorial(n-k))/(math.factorial(k)
                                                    * math.factorial(a-k)
                                                    * math.factorial(b-k)))
                            * np.power(r, (n - 2*k)))
    return value


def radial_derivative_eq(zernike_pol, r):
    """
    Calculate the derivative of radial polynomial R(m, n) on r (eq. dR(m, n)/dr).

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    r : float or numpy.ndarray
        Radius from the unit circle, float or array of values on which the Zernike polynomial is calculated.

    Returns
    -------
    float or numpy.ndarray
        Depending on the type of theta, return float or numpy.ndarray with calculated values of radial polynomial derivative.

    """
    if isinstance(r, float):
        value = 0.0
    elif isinstance(r, np.ndarray):
        value = np.zeros(r.shape)
    (m, n) = define_orders(zernike_pol)  # get orders
    if n > 0:
        for k in range(0, ((n - abs(m))//2) + 1):
            a = (n + m)//2; b = (n - m)//2
            if n - 2*k > 0:  # because the derivative is zero for np.power(r, 0), n - 2*k == 0
                value += ((-1)**k)*(((math.factorial(n-k))/(math.factorial(k)
                                                            * math.factorial(a-k)
                                                            * math.factorial(b-k)))
                                    * np.power(r, (n - 2*k - 1)))*(n - 2*k)
    return value


def triangular_function(zernike_pol, theta):
    """
    Return triangular component of the Zernike polynomial.

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    theta : float or numpy.ndarray
        Theta - angle in radians, float or array of angles on which the Zernike polynomial is calculated.
        Note that the theta counting is counterclockwise, as it is default for the matplotlib library.

    References
    ----------
    [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)
    [2] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011)

    Returns
    -------
    float or numpy.ndarray
        Depending on the type of theta, return float or numpy.ndarray with calculated values of triangular function.

    """
    (m, n) = define_orders(zernike_pol)  # get orders
    # Calculation of the value according to Refs.
    if m >= 0:
        return np.cos(m*theta)
    else:
        return -np.sin(m*theta)


def triangular_derivative(zernike_pol, theta):
    """
    Return derivative of triangular function of the Zernike polynomial.

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    theta : float or numpy.ndarray
        Theta - angle in radians, float or array of angles on which the Zernike polynomial is calculated.
        Note that the theta counting is counterclockwise, as it is default for the matplotlib library.

    Returns
    -------
    float or numpy.ndarray
        Depending on the type of theta, return float or numpy.ndarray with calculated values of derivative of triangular function.

    """
    (m_order, n) = define_orders(zernike_pol)  # get orders
    # Calculation of the value according to the analytical derivative of the equation above
    m = float(m_order)
    if m_order >= 0:
        return -m*np.sin(m*theta)
    else:
        return -m*np.cos(m*theta)


def radial_polynomial_coeffs(zernike_pol, r):
    """
    Calculate radial polynomial using recursive finding algorithm of each coefficient for radial component (e.g. R^6).

    After 40 order, this function again uses the recursive equation, because the finding the coefficients becomes
    unstable.

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    r : float or numpy.ndarray
        Radius from the unit circle, float or array of values on which the Zernike polynomial is calculated.

    References
    ----------
    [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)

    Returns
    -------
    r_sum : float or numpy.ndarray
        Depending on the type of theta, return float or np.ndarray with calculated values of radial polynomial.

    """
    m, n = define_orders(zernike_pol)
    if n <= MAX_RADIAL_ORDER_COEFFS:
        pols_coeffs = find_coeffs_orders((m, n))
    # Initial value for sum calculation
    if isinstance(r, float):
        r_sum = 0.0
    elif isinstance(r, np.ndarray):
        r_sum = np.zeros(shape=r.shape)
    # Calculation of sum of orders for not super high orders
    if n <= MAX_RADIAL_ORDER_COEFFS:
        for key, value in pols_coeffs.items():
            if abs(value) > 0:
                r_sum += value*np.power(r, key)
    # Special case - for really high orders
    else:
        # Recurrence equations - stable way to refer previously stably calculated values
        # Simple check that values stably calculated - radial polynomial values shouldn't exceed 1.0
        if abs(m) == n:  # derivative from the simplified recurrence formula from [3]
            r_sum = np.power(r, n)
        elif m == 0:  # simplified recurrence formula from [3]
            r_sum = 2.0*r*radial_polynomial_coeffs((1, n-1), r) - radial_polynomial_coeffs((0, n-2), r)
        elif abs(m) == n-2:  # my guess about overall equation
            r_sum = float(n)*np.power(r, n) - float(n-1)*np.power(r, n-2)
        else:
            r_sum = (r*(radial_polynomial_coeffs((abs(m-1), n-1), r) + radial_polynomial_coeffs((m+1, n-1), r))
                     - radial_polynomial_coeffs((m, n-2), r))  # general recurrence formula from [1]
    return r_sum


def radial_polynomial_coeffs_dr(zernike_pol, r):
    """
    Calculate radial polynomial derivative using recursive finding algorithm of each coefficient for radial component.

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    r : float or numpy.ndarray
        Radius from the unit circle, float or array of values on which the Zernike polynomial is calculated.

    References
    ----------
    [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)

    Returns
    -------
    r_sum : float or numpy.ndarray
        Depending on the type of theta, return float or np.ndarray with calculated values of radial polynomial derivative.

    """
    m, n = define_orders(zernike_pol)
    if n <= MAX_RADIAL_ORDER_COEFFS_dR:
        pols_coeffs = find_coeffs_orders_dr((m, n))
    # Initial value for sum calculation
    if isinstance(r, float):
        r_sum = 0.0
    elif isinstance(r, np.ndarray):
        r_sum = np.zeros(shape=r.shape)
    # Calculation of sum of orders for not super high orders
    if n <= MAX_RADIAL_ORDER_COEFFS_dR:
        for key, value in pols_coeffs.items():
            if abs(value) > 0:
                r_sum += value*np.power(r, key)
    # Special case - for really high orders
    else:
        # Recurrence equations - stable way to refer previously stably calculated values
        # Simple check that values stably calculated - radial polynomial values shouldn't exceed 1.0
        if abs(m) == n:  # simplified recurrence formula from [3]
            r_sum = float(n)*np.power(r, n-1)
        elif m == 0:  # derivative from the simplified recurrence formula from [3]
            r_sum = (2.0*(radial_polynomial_coeffs_dr((1, n-1), r) + r*radial_polynomial_coeffs_dr((1, n-1), r))
                     - radial_polynomial_coeffs_dr((0, n-2), r))
        elif abs(m) == n-2:  # derivative from my guess about overall equation
            r_sum = float(n)*float(n)*np.power(r, n-1) - float(n-1)*float(n-2)*np.power(r, n-3)
        else:
            # derivative from the general recurrence formula from [1]
            r_sum = ((radial_polynomial_coeffs((abs(m-1), n-1), r) + radial_polynomial_coeffs((m+1, n-1), r))
                     + r*(radial_polynomial_coeffs_dr((abs(m-1), n-1), r) + radial_polynomial_coeffs_dr((m+1, n-1), r))
                     - radial_polynomial_coeffs_dr((m, n-2), r))
    return r_sum


# %% Test functions
def compare_radial_calculations(max_order: int) -> np.ndarray:
    """
    Test difference between tabular/recursive and exact equation implementations of radial Zernike polynomials.

    Parameters
    ----------
    max_order : int, optional
        Maximum order of tested Zernike polynomials (not less than 2).

    Returns
    -------
    diff : numpy.ndarray
        Size (N_Zernikes, 20) corresponds to number of tested orders (m, n) calculated for the input
        maximum order and 21 radiuses between [0, 1].
        Note that the checked precision difference is 1E-9 and thus the returned matrix also rounded to 9 numbers after
        floating point.

    """
    # check maximum order
    if not isinstance(max_order, int) and max_order < 2:
        print("NOTE that max_order by default set to 2")
        max_order = 2

    # Generating Zernike orders in OSA/ANSI indexing scheme
    orders_list = [(0, 0)]
    for order in range(1, max_order):
        m = -order; n = order
        orders_list.append((m, n))
        for n_azimuthals in range(0, order):
            m += 2
            orders_list.append((m, n))

    # Generation numpy array with radii
    n_points = 21
    test_r = np.zeros(shape=(n_points, ))
    for i in range(n_points):
        test_r[i] = i/(n_points-1)
    test_r = np.round(test_r, 4)
    # Testing that exact calculation and implementation of tabular / recursive are the same
    diff = np.ones(shape=(len(orders_list), n_points))
    for i, order in enumerate(orders_list):
        diff[i, :] = radial_polynomial(order, test_r) - radial_polynomial_eq(order, test_r)
    diff = np.round(diff, 9)
    assert np.max(np.abs(diff)) < 1E-9, (f"Order {order} has inconsistency between tabular/recursion"
                                         + f" and exact implementations, diff: {np.max(np.abs(diff))}")
    print("Difference between analytical and implemented equations for Zernike pol-s is negligible, test passed")
    return diff


def compare_radial_derivatives(max_order: int) -> np.ndarray:
    """
    Test difference between tabular/recursive and exact equation implementations of derivatives of radial Zernike polynomials.

    Parameters
    ----------
    max_order : int, optional
        Maximum order of tested Zernike polynomials (not less than 2).

    Returns
    -------
    diff : numpy.ndarray
        Size (N_Zernikes, 20) corresponds to number of tested orders (m, n) calculated for the input
        maximum order and 21 radiuses between [0, 1].
        Note that the checked precision difference is 1E-9 and thus the returned matrix also rounded to 9 numbers after
        floating point.

    """
    # check maximum order
    if not isinstance(max_order, int) and max_order < 2:
        print("NOTE that max_order by default set to 2")
        max_order = 2

    # Generating Zernike orders in OSA/ANSI indexing scheme
    orders_list = [(0, 0)]
    for order in range(1, max_order):
        m = -order; n = order
        orders_list.append((m, n))
        for n_azimuthals in range(0, order):
            m += 2
            orders_list.append((m, n))

    # Generation numpy array with radii
    n_points = 21
    test_r = np.zeros(shape=(n_points, ))
    for i in range(n_points):
        test_r[i] = i/(n_points-1)
    test_r = np.round(test_r, 4)
    # Testing that exact calculation and implementation of tabular / recursive are the same
    diff = np.ones(shape=(len(orders_list), n_points))
    for i, order in enumerate(orders_list):
        diff[i, :] = radial_derivative(order, test_r) - radial_derivative_eq(order, test_r)
    diff = np.round(diff, 9)
    assert np.max(np.abs(diff)) < 1E-9, (f"Order {order} has inconsistency between tabular/recursion"
                                         + f" and exact implementations, diff: {np.max(np.abs(diff))}")
    print("Difference between analytical and implemented equations for derivatives of Zernike pol-s is negligible, test passed")
    return diff


def compare_recursive_coeffs_radials() -> np.ndarray:
    """
    Test difference between exact radials and finding pols. coeffs. implementations of radial Zernike polynomials.

    Comparison performed between orders 15 and 41 (more than MAX_RADIAL_ORDER_COEFFS).

    Returns
    -------
    diff : numpy.ndarray
        Size corresponds to number of tested orders (m, n) calculated for the input and 101 radii between [0, 1].
        Note that the checked precision difference is 2E-2 and thus the returned matrix also rounded to 9 numbers after
        floating point.

    """
    # Generating Zernike orders in OSA/ANSI indexing scheme
    orders_list = []
    for order in range(15, MAX_RADIAL_ORDER_COEFFS+2):
        m = -order; n = order
        orders_list.append((m, n))
        for n_azimuthals in range(0, order):
            m += 2
            orders_list.append((m, n))
    # Generation numpy array with radii
    n_points = 101
    test_r = np.zeros(shape=(n_points, ))
    for i in range(n_points):
        test_r[i] = i/(n_points-1)
    test_r = np.round(test_r, 4)
    # Testing that exact calculation and implementation of tabular / recursive are the same
    diff = np.ones(shape=(len(orders_list), n_points))
    for i, order in enumerate(orders_list):
        diff[i, :] = radial_polynomial_eq(order, test_r) - radial_polynomial_coeffs(order, test_r)
        # diff[i, :] = radial_polynomial_coeffs(order, test_r)
        # diff[i, :] = radial_polynomial_eq(order, test_r)
        # if np.max(np.abs(diff)) > 1.0:
        #     print(order, np.max(np.abs(diff)))
    diff = np.round(diff, 9)
    assert np.max(np.abs(diff)) < 2E-2, (f"Order {order} has inconsistency between tabular/recursion"
                                         + f" and exact implementations, diff: {np.max(np.abs(diff))}")
    print("Difference between exact equation and pols. coeffs. finding algorithm is negligible, test passed")
    return diff


def compare_recursive_coeffs_radials_dr() -> np.ndarray:
    """
    Test difference between exact radials and finding pols. coeffs. implementations of radial Zernike polynomials derivatives.

    Comparison performed between orders 15 and 41 (more than MAX_RADIAL_ORDER_COEFFS).

    Returns
    -------
    diff : numpy.ndarray
        Size corresponds to number of tested orders (m, n) calculated for the input and 21 radiuses between [0, 1].
        Note that the checked precision difference is 5E-2 and thus the returned matrix also rounded to 9 numbers after
        floating point.

    """
    # Generating Zernike orders in OSA/ANSI indexing scheme
    orders_list = []
    for order in range(15, MAX_RADIAL_ORDER_COEFFS_dR+2):
        m = -order; n = order
        orders_list.append((m, n))
        for n_azimuthals in range(0, order):
            m += 2
            orders_list.append((m, n))
    # Generation numpy array with radii
    n_points = 101
    test_r = np.zeros(shape=(n_points, ))
    for i in range(n_points):
        test_r[i] = i/(n_points-1)
    test_r = np.round(test_r, 4)
    # Testing that exact calculation and implementation of tabular / recursive are the same
    diff = np.ones(shape=(len(orders_list), n_points))
    for i, order in enumerate(orders_list):
        diff[i, :] = radial_derivative_eq(order, test_r) - radial_polynomial_coeffs_dr(order, test_r)
        # if np.max(np.abs(diff)) > 1.0:
        #     print(order, np.max(np.abs(diff)))
    diff = np.round(diff, 9)
    assert np.max(np.abs(diff)) < 5E-2, (f"Order {order} has inconsistency between tabular/recursion"
                                         + f" and exact implementations, diff: {np.max(np.abs(diff))}")
    print("Difference between exact equation and pols. coeffs. finding algorithm is negligible, test passed")
    return diff


def check_high_orders_recursion() -> np.ndarray:
    """
    Test max of abs value of calculated recursively radial polynomials for orders [41, 45].

    Returns
    -------
    diff : numpy.ndarray.
        Composed radial polynomials values for 51 radii.

    """
    # Generating Zernike orders in OSA/ANSI indexing scheme
    orders_list = []
    for order in range(MAX_RADIAL_ORDER_COEFFS+1, MAX_RADIAL_ORDER_COEFFS+6):
        m = -order; n = order
        orders_list.append((m, n))
        for n_azimuthals in range(0, order):
            m += 2
            orders_list.append((m, n))
    # Generation numpy array with radii
    n_points = 51
    test_r = np.zeros(shape=(n_points, ))
    for i in range(n_points):
        test_r[i] = i/(n_points-1)
    test_r = np.round(test_r, 4)
    # Testing that exact calculation and implementation of tabular / recursive are the same
    diff = np.ones(shape=(len(orders_list), n_points))
    for i, order in enumerate(orders_list):
        diff[i, :] = radial_polynomial_coeffs(order, test_r)
    diff = np.round(diff, 6)
    assert np.max(np.abs(diff)) <= 1.0, f"Order {order} has inconsistency in max abs value: {np.max(np.abs(diff))}"
    print("Abs max in calculated recursively radial polynomials <= 1.0, test passed")
    return diff


def time_radial_pols():
    """
    Measure in the simple way the required time for calculation of radial polynomials with high orders.

    Returns
    -------
    None.

    """
    calc_times_ms = []  # for storing calculation times
    zp1 = (2, 16); zp2 = (0, 18); zp3 = (-2, 20); zp4 = (6, 22); zp5 = (-4, 24); r = 0.425
    zpols = [zp1, zp2, zp3, zp4, zp5]
    for zp in zpols:
        t1 = time.perf_counter()
        radial_polynomial(zp, r)
        t2 = time.perf_counter()
        calc_times_ms.append(int(round(1000*(t2-t1), 0)))
    print("Timed calc. recursive rad. pol.", zpols, ":", calc_times_ms)
    calc_times_ms = []  # refresh stored values
    for zp in zpols:
        t1 = time.perf_counter()
        radial_polynomial_coeffs(zp, r)
        t2 = time.perf_counter()
        calc_times_ms.append(round(1000*(t2-t1), 3))
    print("Timed calc. using finding coeffs. rad. pol.", calc_times_ms)
    calc_times_ms = []  # refresh stored values
    for zp in zpols:
        t1 = time.perf_counter()
        radial_polynomial_eq(zp, r)
        t2 = time.perf_counter()
        calc_times_ms.append(round(1000*(t2-t1), 3))
    print("Timed calc. using exact equation rad. pol.", calc_times_ms)
    calc_times_ms = []  # refresh stored values
    for zp in zpols:
        t1 = time.perf_counter()
        radial_polynomial_coeffs_dr(zp, r)
        t2 = time.perf_counter()
        calc_times_ms.append(round(1000*(t2-t1), 3))
    print("Timed calc. using finding coeffs. deriv. rad. pol.", calc_times_ms)
    # Separate test of exact equation performance
    zp1 = (-14, 18); zp2 = (0, 26); zp3 = (3, 29); zp4 = (0, 31); zp5 = (-2, 36); zp6 = (1, 39); r = 0.425
    zpols = [zp1, zp2, zp3, zp4, zp5, zp6]
    calc_times_ms = []  # refresh stored values
    for zp in zpols:
        t1 = time.perf_counter()
        radial_polynomial_eq(zp, r)
        t2 = time.perf_counter()
        calc_times_ms.append(round(1000*(t2-t1), 3))
    print("Timed calc. using exact equation rad. pol.", zpols, ":", calc_times_ms)


# %% Tests
if __name__ == '__main__':
    R = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; R = np.asarray(R)
    Theta = [i*np.pi/3 for i in range(6)]; Theta = np.asarray(Theta)
    orders = (-2, 2); r = 0.5
    ZR = radial_polynomial(orders, R); ZR1 = radial_polynomial(orders, r)
    TR = triangular_function(orders, Theta)
    diff = compare_radial_calculations(max_order=20)
    diff_deriv = compare_radial_derivatives(max_order=18)
    # Test specific implementations
    orders = (-8, 52); r = 0.95; val = radial_polynomial_coeffs(orders, r)
    time_radial_pols()  # initial estimation of performance of calculations
    diff_coeffs = compare_recursive_coeffs_radials()
    diff_deriv_coeffs = compare_recursive_coeffs_radials_dr()
    high_R = check_high_orders_recursion()
