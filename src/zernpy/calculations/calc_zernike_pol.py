# -*- coding: utf-8 -*-
"""
Collection of Zernike polynomial calculation methods.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np
import math


# %% Function definitions
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
        (m, n) = zernike_pol.get_polynomial_orders()
    return (m, n)


def normalization_factor(zernike_pol) -> float:
    """
    Calculate normalization factor according to the reference (N(m ,n)).

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.

    Reference
    ---------
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
    Calculate radial polynomial R(m, n) value for input r laying in the range [0, 1].

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    r : float or np.ndarray
        Radius from the unit circle, float or array of values on which the Zernike polynomial is calculated..

    Reference
    ---------
    [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)
    [2] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011)
    [3] Andersen T. B. "Efficient and robust recurrence relations for the Zernike
    circle polynomials and their derivatives in Cartesian coordinates" (2018)

    Returns
    -------
    float or np.ndarray
        Depending of the type of theta, return float or np.ndarray with calculated values of radial polynomial.

    """
    (m, n) = define_orders(zernike_pol)  # get orders
    # Radial polynomials defined as analytical equation for up to 7th order (check tables from [3])
    # 0th orders
    if ((m == 0) and (n == 0)):
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
    # Recurrence equations from the [1] and [3]
    elif n > 7 and abs(m) == n:
        return np.power(r, n)
    elif n > 7 and m == 0:
        return 2*r*radial_polynomial((1, n-1), r) - radial_polynomial((0, n-2), r)
    else:
        return (r*(radial_polynomial((abs(m-1), n-1), r) + radial_polynomial((m+1, n-1), r))
                - radial_polynomial((m, n-2), r))

def radial_polynomial_eq(zernike_pol, r):
    value = 0.0
    (m, n) = define_orders(zernike_pol)  # get orders
    for k in range(0, ((n - abs(m))//2) + 1):
        a = (n + m)//2; b = (n - m)//2
        value += ((-1)**k)*(((math.factorial(n-k))/(math.factorial(k)
                                                    * math.factorial(a-k)
                                                    * math.factorial(b-k)))
                            * np.power(r, (n - 2*k)))
    return value


def triangular_function(zernike_pol, theta):
    """
    Return triangular component of the Zernike polynomial.

    Parameters
    ----------
    zernike_pol : ZernPol or tuple with orders (m, n)
        ZernPol - class instance of the calling class (module zernikepol) or tuple with azimuthal and radial orders.
    theta : float or np.ndarray
        Theta - angle in radians, float or array of angles on which the Zernike polynomial is calculated.

    Reference
    ---------
    [1] Shakibaei B.H., Paramesran R. "Recursive formula to compute Zernike radial polynomials" (2013)
    [2] Lakshminarayanan V., Fleck A. "Zernike polynomials: a guide" (2011)

    Returns
    -------
    float or np.ndarray
        Depending of the type of theta, return float or np.ndarray with calculated values of triangular function.

    """
    (m, n) = define_orders(zernike_pol)  # get orders
    # Calculation of the value according to Refs.
    if m >= 0:
        return np.cos(m*theta)
    else:
        return -np.sin(m*theta)


# %% Tests
if __name__ == '__main__':
    R = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; R = np.asarray(R)
    Theta = [i*np.pi/3 for i in range(6)]; Theta = np.asarray(Theta)
    orders = (-2, 2); r = 0.5
    ZR = radial_polynomial(orders, R); ZR1 = radial_polynomial(orders, r)
    print(ZR - radial_polynomial_eq(orders, R))
    print(ZR1 - radial_polynomial_eq(orders, r))
    TR = triangular_function(orders, Theta); TR1 = triangular_function(orders, r)
    orders = (-9, 9)
    ZR2 = radial_polynomial(orders, r)
    print(ZR2 - radial_polynomial_eq(orders, r))
    orders = (0, 8)
    ZR3 = radial_polynomial(orders, r)
    orders = (-1, 9); r = 0.25
    ZR4 = radial_polynomial(orders, r)
    print(ZR4 - radial_polynomial_eq(orders, r))