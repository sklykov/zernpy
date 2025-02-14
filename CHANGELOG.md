# Changelog
Logging of changes between package versions (generated and uploaded to pypi.org).

All notable changes to this project will be documented in this file.    
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).   

### [0.0.15] - 2025-02-14

#### Added
***crop_kernel()*** method of ***ZernPSF*** class for cropping large PSF kernels; 

#### Fixed
- Checking for consistency of provided amplitudes (coefficients) and number of polynomials;
- Better estimation of required kernel size for several polynomials kernel calculation;
- Text for plot namings, warnings in case of several polynomials kernel calculation; 
- Removed images used in README from distribution packages (archives).


### [0.0.14] - 2024-11-09

#### Added
- PSF kernel calculation for several polynomials;
- Acceleration by numba compilation;

#### Fixed
- Issue with type hints for Python 3.8 (tested in the environment with Python 3.8.20).


### [0.0.13] - 2024-06-13
Added the **ZernPSF** class class with methods for calculation, visualization, and convolution
with a 2D PSF kernel, corresponding to the Zernike polynomial.


### [0.0.12] - 2023-12-11
Added this file for storing changes between package versions.
 
#### Added
- Fitting using accounting of the Piston polynomial;
- A few more tests of new fitting procedure;
- ***functools.lru_cache*** annotation for speeding up of recursive calculation of polynomial coefficients.

#### Fixed
- Several docstrings of the methods.
