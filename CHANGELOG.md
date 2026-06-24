# Changelog
Logging of changes between package versions (generated and uploaded to pypi.org).

All notable changes to this project will be documented in this file.    
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  


### [0.1.2] - 2026-06-26

#### Added
- Calculation of Peak-to-Valley (PV) and RMS values for OSA-normalized Zernike polynomials.

#### Changed
- Corrected the conversion of Zernike expansion coefficients to phase units (radians) for PSF kernel calculations. 
Expansion coefficients are now multiplied by `2π / wavelength` instead of being divided by the wavelength only; 
- Revised the PSF kernel size estimation logic. The estimated kernel size is now generally larger for the same input 
parameters, reducing the risk of PSF truncation during calculations.


### [0.1.1] - 2026-04-29

#### Added
- Saving of compiled by numba computational methods and function for cleaning them up;
- Automatic publishing workflow used by GitHub Actions.

#### Changed 
- GitHub Actions tests, added new Python version runtimes;
- Typing definitions after setting up mypy testing rules; 
- Codebase after applying ruff linter rules.


### [0.1.0] - 2026-01-07

#### Added
- Automatic testing workflow for GitHub - run collected tests by ***pytest*** library after merging with the main branch;  
- Bumped up version to a minor release version (0.1.0) for designating persistence of developed API (methods).

#### Fixed
- Returning types of class methods; 
- Formulations in docstrings for class methods.


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
Added the **ZernPSF** class with methods for calculation, visualization, and convolution
with a 2D PSF kernel, corresponding to the Zernike polynomial.


### [0.0.12] - 2023-12-11
Added this file for storing changes between package versions.
 
#### Added
- Fitting using accounting of the Piston polynomial;
- A few more tests of new fitting procedure;
- ***functools.lru_cache*** annotation for speeding up of recursive calculation of polynomial coefficients.

#### Fixed
- Several docstrings of the methods.
