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

except ModuleNotFoundError:
    print("The package zernpy isn't installed")
