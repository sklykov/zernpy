# -*- coding: utf-8 -*-
"""
Test if the package "zernpy" has been installed.

@author: Sergei Klykov
@licence: MIT
"""
try:
    from zernpy import ZernPol

    def test_initialization():
        zp = ZernPol(m=-2, n=2)
        assert len(zp.get_polynomial_name()) > 0, "Failed simple function call for getting polynomial name"

except ModuleNotFoundError:
    print("The package zernpy isn't installed, install it for passing this basic test")
