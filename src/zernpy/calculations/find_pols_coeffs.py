# -*- coding: utf-8 -*-
"""
Calculation of Zernike radial polynomials coefficients using recurrence equation.

@author: Sergei Klykov
@licence: MIT

"""
import time


# %% Gen orders dict.
def make_orders_coeffs(defined_coeff: dict, max_order: int, minus: bool = False) -> dict:
    """
    Generate dictionary with radial orders of polynomials: coefficients as key: value pairs.

    Note that non-zero coefficients will be parsed from defined_coeff input.

    Parameters
    ----------
    defined_coeff : dict
        Dictionary with non-zero integer coefficients (values) for radial orders as keys.
    max_order : int
        Maximum radial order n of polynomials.
    minus : bool, optional
        Used for getting inverse coefficients. The default is False.

    Raises
    ------
    ValueError
        If provided max_order and way how the values/keys specified in defined_coeff is wrong.

    Returns
    -------
    dict
        Keys - radial orders of polynomials (n), values -integers with polynomials coefficients.

    """
    # Initialize dictionary with zero coefficients
    coefficients = {}
    for i in range(max_order+1):
        coefficients[i] = 0
    # Setting provided already defined coefficients to the initialized dictionary
    if len(defined_coeff.keys()) > 0:
        for key, value in defined_coeff.items():
            if key in coefficients.keys():
                if not minus:
                    coefficients[key] = value
                else:
                    coefficients[key] = -value
            else:
                raise ValueError("Generated coefficients don't include specified order / value")
    return coefficients


# %% Initial set of coefficients
global initial_coefficients_test
initial_coefficients_test = {(-3, 7): make_orders_coeffs({7: 21, 5: -30, 3: 10}, 7),
                             (3, 7): make_orders_coeffs({7: 21, 5: -30, 3: 10}, 7),
                             (-1, 7): make_orders_coeffs({7: 35, 5: -60, 3: 30, 1: -4}, 7),
                             (1, 7): make_orders_coeffs({7: 35, 5: -60, 3: 30, 1: -4}, 7),
                             (-4, 8): make_orders_coeffs({8: 28, 6: -42, 4: 15}, 8),
                             (4, 8): make_orders_coeffs({8: 28, 6: -42, 4: 15}, 8),
                             (-2, 8): make_orders_coeffs({8: 56, 6: -105, 4: 60, 2: -10}, 8),
                             (2, 8): make_orders_coeffs({8: 56, 6: -105, 4: 60, 2: -10}, 8),
                             (0, 8): make_orders_coeffs({8: 70, 6: -140, 4: 90, 2: -20, 0: 1}, 8)}
global precalculated_initial_coeffs
precalculated_initial_coeffs = {(-9, 9): make_orders_coeffs({9: 1}, 9), (9, 9): make_orders_coeffs({9: 1}, 9),
                                (-7, 9): make_orders_coeffs({9: 9, 7: -8}, 9),
                                (7, 9): make_orders_coeffs({9: 9, 7: -8}, 9),
                                (-5, 9): make_orders_coeffs({9: 36, 7: -56, 5: 21}, 9),
                                (5, 9): make_orders_coeffs({9: 36, 7: -56, 5: 21}, 9),
                                (-3, 9): make_orders_coeffs({9: 84, 7: -168, 5: 105, 3: -20}, 9),
                                (3, 9): make_orders_coeffs({9: 84, 7: -168, 5: 105, 3: -20}, 9),
                                (-1, 9): make_orders_coeffs({9: 126, 7: -280, 5: 210, 3: -60, 1: 5}, 9),
                                (1, 9): make_orders_coeffs({9: 126, 7: -280, 5: 210, 3: -60, 1: 5}, 9),
                                (-6, 10): make_orders_coeffs({10: 45, 8: -72, 6: 28}, 10),
                                (6, 10): make_orders_coeffs({10: 45, 8: -72, 6: 28}, 10),
                                (-4, 10): make_orders_coeffs({10: 120, 8: -252, 6: 168, 4: -35}, 10),
                                (4, 10): make_orders_coeffs({10: 120, 8: -252, 6: 168, 4: -35}, 10),
                                (-2, 10): make_orders_coeffs({10: 210, 8: -504, 6: 420, 4: -140, 2: 15}, 10),
                                (2, 10): make_orders_coeffs({10: 210, 8: -504, 6: 420, 4: -140, 2: 15}, 10),
                                (0, 10): make_orders_coeffs({10: 252, 8: -630, 6: 560, 4: -210, 2: 30, 0: -1}, 10)}


# %% Other func. defs.
def increase_order_coeffs(coefficients: dict) -> dict:
    """
    Return new dict with all values reassigned to order + 1, required by recurrence equation.

    Parameters
    ----------
    coefficients : dict
        Order n: value as dictionary values.

    Returns
    -------
    dict
        Composed shifted orders and related values.

    """
    # Initialize returning dictionary by defining max order stored in input coefficients
    increased_order_coefficients = {}
    max_order = max(coefficients.keys())
    for i in range(max_order+1):
        increased_order_coefficients[i] = 0
    for key, value in coefficients.items():
        if abs(value) > 0:
            increased_order_coefficients[key+1] = value
    return increased_order_coefficients


def sum_orders_coeffs(max_order: int, *args) -> dict:
    # Initialize dictionary with zero coefficients
    coefficients = {}
    for i in range(max_order+1):
        coefficients[i] = 0
    # Sum on the provided dictionaries
    # print("Call for dict sum: ", *args)
    for input_coeffs in args:
        for key, value in input_coeffs.items():
            coefficients[key] += value
    return coefficients


def check_special_orders(orders: tuple) -> dict:
    """
    Check if the coefficients could be calculated immediately for cases: abs(m) == n and abs(m) == n-2.

    Parameters
    ----------
    orders : tuple
        Orders (m, n) as a tuple.

    Returns
    -------
    dict
        Polynomials coefficients with radial order: value pairs.

    """
    special_coefficients = None
    m, n = orders
    if abs(m) == n:
        special_coefficients = make_orders_coeffs({n: 1}, n)
    elif abs(m) == n-2:
        special_coefficients = make_orders_coeffs({n: n, n-2: -(n-1)}, n)
    return special_coefficients


def find_pols_coeffs(orders: tuple, use_test_dict: bool = False) -> dict:
    m, n = orders
    # print("Orders called with:", m, n)
    if use_test_dict:
        initial_coefficients = initial_coefficients_test
    else:
        initial_coefficients = precalculated_initial_coeffs
    if orders in initial_coefficients.keys():
        # print("Found in initial dict.: ", initial_coefficients[orders])
        return initial_coefficients[orders]  # return stored in the dictionary value
    elif check_special_orders(orders) is not None:
        # Cashing already calculated coefficients in global dictionary specified above
        if not use_test_dict:
            if orders not in precalculated_initial_coeffs.keys():
                precalculated_initial_coeffs[orders] = check_special_orders(orders)
        else:
            if orders not in initial_coefficients_test.keys():
                initial_coefficients_test[orders] = check_special_orders(orders)
        return check_special_orders(orders)  # some special shorthanded cases for polynomials values calculation
    else:
        polm1n1 = find_pols_coeffs((abs(m-1), n-1), use_test_dict)
        polmP1n1 = find_pols_coeffs((m+1, n-1), use_test_dict)
        sum_dict_coeffs = sum_orders_coeffs(n, increase_order_coeffs(polm1n1),
                                            increase_order_coeffs(polmP1n1),
                                            make_orders_coeffs(find_pols_coeffs((m, n-2), use_test_dict),
                                                               max_order=n-2, minus=True))
        # Cashing already calculated coefficients in global dictionary specified above
        if not use_test_dict:
            if (abs(m-1), n-1) not in precalculated_initial_coeffs.keys():
                precalculated_initial_coeffs[(abs(m-1), n-1)] = polm1n1
            if (m+1, n-1) not in precalculated_initial_coeffs.keys():
                precalculated_initial_coeffs[(m+1, n-1)] = polmP1n1
        else:
            if (abs(m-1), n-1) not in initial_coefficients_test.keys():
                initial_coefficients_test[(abs(m-1), n-1)] = polm1n1
            if (m+1, n-1) not in initial_coefficients_test.keys():
                initial_coefficients_test[(m+1, n-1)] = polmP1n1
        return sum_dict_coeffs


def check_equal_coeffs(coeffs1: dict, coeffs2: dict) -> bool:
    """
    Check that 2 dictionaries with calculated coefficients have equal values for the same orders (keys).

    Parameters
    ----------
    coeffs1 : dict
        Calculated coefficients.
    coeffs2 : dict
        Precoded coefficients.

    Returns
    -------
    bool
        Whatever or not dictionary are identical.

    """
    result = True
    if len(coeffs1.keys()) == len(coeffs2.keys()):
        for key, value in coeffs1.items():
            if value != coeffs2[key]:
                print(key, value, coeffs2[key])
                result = False; break
    else:
        result = False
    return result


def test_coeffs_calc():
    """
    Test calculation of polynomials coefficients using tested before 9th order ones.

    Returns
    -------
    None.

    """
    m = -9; n = 9; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = -7; n = 9; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = -5; n = 9; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = -3; n = 9; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = -1; n = 9; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = 1; n = 9; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = 5; n = 9; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = 0; n = 10; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = 2; n = 10; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = 4; n = 10; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = -6; n = 10; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"
    m = 7; n = 9; orders_rec = find_pols_coeffs((m, n), use_test_dict=True)
    assert check_equal_coeffs(orders_rec, precalculated_initial_coeffs[(m, n)]), f"Check coeffs. for {(m, n)}"


def measure_time_high_orders(orders: tuple) -> dict:
    """
    Print out the measured time required for calculation coefficients for specified orders.

    Parameters
    ----------
    orders : tuple
        Orders m, n put in tuple.

    Returns
    -------
    pols_coeffs : dict
        Calculated dictionary with orders (m, n): coefficients as key: value pairs.

    """
    t1 = time.perf_counter()
    pols_coeffs = find_pols_coeffs(orders)
    t2 = time.perf_counter()
    print("Calculation of coefficients takes ms: ", int(round(1000*(t2-t1), 0)))
    return pols_coeffs


# %% Testing
if __name__ == "__main__":
    test_coeffs_calc()
    print("*****Tests passed*****")
    coeffs1 = measure_time_high_orders((0, 30))
    coeffs2 = measure_time_high_orders((-9, 25))
    coeffs3 = measure_time_high_orders((0, 50))
