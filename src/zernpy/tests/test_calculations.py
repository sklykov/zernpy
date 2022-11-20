# -*- coding: utf-8 -*-
"""
Test the modules by using pytest library.

pytest library avalaible on: https://docs.pytest.org/en/latest/contents.html

@author: Sergei Klykov
@licence: MIT

"""

# Importing the written in the modules test functions for letting pytest library automatic exploration
if __name__ != "__main__":
    from ..calculations.calc_zernike_pol import compare_radial_calculations


def test_tabular_orders():
    compare_radial_calculations(max_order=7)


def test_recursive_orders():
    compare_radial_calculations(max_order=21)
