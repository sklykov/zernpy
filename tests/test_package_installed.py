# -*- coding: utf-8 -*-
"""
Test the installed package.

@author: Sergei Klykov
@licence: MIT
"""
try:
    from zernpy import ZernPol

    def test_initialization():
        zp = ZernPol(m=-2, n=2)
        assert len(zp.get_polynomial_name()) > 0, "Failed simple function call for getting polynomial name"

except ModuleNotFoundError:
    print("The package zernpy isn't installed, if the test is run as the module")
