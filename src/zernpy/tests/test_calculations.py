# -*- coding: utf-8 -*-
"""
Test the implemented classes and methods by using pytest library.

pytest library avalaible on: https://docs.pytest.org/en/latest/contents.html
Note that for the running test without module installation, run this script by pytest module itself:
in command line navigate to this folder and run command pytest.

@author: Sergei Klykov
@licence: MIT

"""
# Importing the written in the modules test functions for letting pytest library automatic exploration
if __name__ != "__main__":
    from ..calculations.calc_zernike_pol import compare_radial_calculations
    from ..zernikepol import check_conformity


# Testing implemented equations for tabular R(m, n) from References by comparing with the exact ones
def test_tabular_orders():
    compare_radial_calculations(max_order=7)


# Testing implemented recursive equations for R(m, n) from References by comparing with the exact ones
def test_recursive_orders():
    compare_radial_calculations(max_order=21)


# Testing initialization of Zernike polynomials
def test_polynomials_initialization():
    check_conformity()
