# zernpy
[![Tests](https://github.com/sklykov/zernpy/actions/workflows/test.yaml/badge.svg)](https://github.com/sklykov/zernpy/actions/workflows/test.yaml)
[![PyPI](https://img.shields.io/pypi/v/zernpy)](https://pypi.org/project/zernpy/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Lint](https://img.shields.io/badge/lint-ruff-informational)](https://github.com/astral-sh/ruff)

Python package for:  
- Calculating real-valued Zernike polynomials based on exact (factorial) and recursive equations;
- Converting between radial (n) + azimuthal (m) orders, OSA, Noll, and Fringe indices; 
- Fitting phase profiles with Zernike polynomials;
- Generating and visualizing wavefronts formed by single or sums of polynomials;
- Computing 2D PSF kernels in the imaging plane of an optical system for individual polynomials and polynomial sums.

### Features

- Real-valued Zernike polynomial computation in analytical and recursive forms;
- Derivatives computation (radial and azimuthal);
- Orders / OSA / Noll / Fringe index conversions;
- Wavefront fitting; 
- 2D PSF kernel simulation (image plane);
- Optional Numba acceleration of computations;
- Visualization (plotting) utilities.

Full API documentation: [sklykov.github.io/zernpy](https://sklykov.github.io/zernpy/)  
The recursive form of the equations is used by default. It becomes valuable for high-order polynomials 
(`n > 40`) due to improved numerical stability.

### Setup instructions

#### Installation
```console
pip install zernpy
```
To upgrade:
```console
pip install -U zernpy
```

#### Requirements
The package requires `numpy` and `matplotlib` with no specific version constraints.  
Tests are run with `pytest`. Linting and formatting use `ruff`; non-strict type checking uses `mypy`.

### Examples
#### Minimal Example
```python
from zernpy import ZernPol
from math import pi
zp = ZernPol(m=0, n=4)  # Spherical polynomial initialization
# Similar init. forms: ZernPol(osa=12), ZernPol(noll=11), ZernPol(fringe=9)
indices = zp.get_indices()  # returns tuple: ((m, n), OSA, Noll, Fringe) orders / indices
naming = zp.get_polynomial_name()  # returns str with polynomial name (up to 7th order)
# Coordinates can be provided as real numbers or NumPy arrays
r = 0.5; theta = pi*0.25  # as example of a single polar point
value = zp.polynomial_value(r, theta)  # polynomial value(-s) for radial coordinates
value_r = zp.radial(r)  # radial polynomial value(-s)
value_dr = zp.radial_dr(r)  # derivative on r of radial polynomial value(-s)
value_ang = zp.triangular(theta)  # angular polynomial value(-s)
value_ang_dth = zp.triangular_dtheta(theta)  # derivative on theta of angular value(-s)
normalization = zp.normf()  # OSA or Var(ZP)=1 normalization factor
```
#### Example of a few useful static methods
```python
from zernpy import ZernPol
# Notations conversion
m, n = ZernPol.index2orders(osa_index=10)  # Get azimuthal, radial orders. Same for noll_index, fringe_index
noll = ZernPol.osa2noll(10)  # Also available: noll2osa, osa2fringe, fringe2osa
ZernPol.plot_profile(ZernPol(fringe=11))  # interactive plotting of a single polynomial
# Sum of Zernike polynomials as a surface
zerns = ZernPol(osa=3), ZernPol(osa=6)
zerns_sum_surface = ZernPol.gen_zernikes_surface(coefficients=[0.1, -0.1], polynomials=zerns)
ZernPol.plot_zernikes_surface(zerns_sum_surface)  # interactive matplotlib plot
```
> **Note**  
> Images below are visible on GitHub but not on the PyPI project page.

#### Fitting Zernike polynomials to a 2D phase profile
There are methods implemented for fitting a provided phase profile to a provided set of polynomials. Minimal example:
```python
import matplotlib.pyplot as plt
from zernpy import ZernPol, generate_phases_image, fit_polynomials
polynomials = (ZernPol(m=-1, n=5), ZernPol(m=3, n=3), ZernPol(m=0, n=2))
pols_coeffs = (-0.114, 0.245, 0.403)
phases_image = generate_phases_image(polynomials=polynomials, polynomials_amplitudes=pols_coeffs,
                                     img_height=401, img_width=401)
plt.figure("Initial Phase Profile"); plt.imshow(phases_image, cmap="jet")
plt.axis("off"); plt.tight_layout()
polynomials_amplitudes, cropped_img = fit_polynomials(phases_image, polynomials, return_cropped_image=True)
plt.figure("Fitted Phase Profile"); plt.imshow(cropped_img, cmap="jet")
plt.axis("off"); plt.tight_layout()
```
Initially generated profile:   
![Initial Profile](./src/zernpy/readme_images/Initial_Phase_Profile.png "Random phases profile, 'jet' matplotlib colormap")  

Fitted profile:   
![Fitted Profile](./src/zernpy/readme_images/Fitted_Phase_Profile.png "Fitted polynomials profile, 'jet' matplotlib colormap")  

#### 2D PSF kernel calculation
The 2D PSF kernel is calculated from the diffraction integral over the round pupil plane and described as Zernike polynomial phase 
distribution for the focal point (no Z-axis dependency).  

Initialization and usage of the class instance (basic usage with default calculation parameters, such as the kernel size):    
```python
from zernpy import ZernPSF, ZernPol
zpsf = ZernPSF(ZernPol(m=1, n=3))  # horizontal coma
NA = 0.4; wavelength = 0.55; pixel_physical_size = 0.24*wavelength; expansion_coeff = -0.1
zpsf.set_physical_props(NA, wavelength, expansion_coeff, pixel_physical_size)
kernel = zpsf.calculate_psf_kernel(normalized=True)  # get a kernel as a squared matrix
```
The PSF kernel obtained for horizontal coma with the above parameters is shown below:    
![Horizontal Coma Kernel](./src/zernpy/readme_images/(1,3)_Hor._coma_-0.1.png "Horizontal Coma Kernel")   
Check the API documentation for other available methods.     

#### PSF kernel for several polynomials
Similarly to the code above, it's possible to calculate a PSF kernel associated with a sum profile of several polynomials:   
```python
from zernpy import ZernPSF, ZernPol 
zp1 = ZernPol(m=-1, n=3); zp2 = ZernPol(m=2, n=4); zp3 = ZernPol(m=0, n=4)
pols = (zp1, zp2, zp3); coeffs = (0.5, 0.21, 0.15)
zpsf_pic = ZernPSF(pols); zpsf_pic.set_physical_props(NA=0.65, wavelength=0.6, expansion_coeff=coeffs, 
                                                      pixel_physical_size=0.6/5.0)
zpsf_pic.calculate_psf_kernel(); zpsf_pic.plot_kernel("Sum of Polynomials Profile")
```
The resulting profile is:    

![3 Polynomials Kernel Plot](./src/zernpy/readme_images/Kernel_Sum_Vert_Coma_2nd_Astigm_Spher.png "3 Polynomials Kernel Plot")  

#### Acceleration of kernel calculation by numba
It's possible to accelerate the calculation of a kernel by installing the [numba](https://numba.pydata.org/) library in the 
same Python environment and providing the appropriate flags in a calculation method, similar to the following code snippet:
```python
from zernpy import force_get_psf_compilation, ZernPol, ZernPSF
force_get_psf_compilation()  # optional precompilation for saving and reusing compiled code
NA = 0.95; wavelength = 0.55; pixel_size = wavelength / 4.6; ampl = -0.12
zp = ZernPol(m=0, n=2); zpsf = ZernPSF(zp) 
zpsf.set_physical_props(NA, wavelength, ampl, pixel_size)
zpsf.calculate_psf_kernel(accelerated=True)
```

#### Cropping kernel
By default, the kernel size is automatically estimated and may be overestimated to guarantee that all significant 
points of kernel will be calculated. Also, kernel size is growing with the polynomial orders and its amplitude. 
To reduce the size of kernel, it's possible to call the method ***crop_kernel***, e.g.: 
```python
from zernpy import ZernPSF, ZernPol
zpsf = ZernPSF(zernpol=(ZernPol(m=0, n=4)))  # Spherical aberration
zpsf.set_physical_props(NA=1.25, wavelength=0.5, expansion_coeff=0.25, pixel_physical_size=0.5/5.0)
zpsf.calculate_psf_kernel(accelerated=True, verbose_info=True); zpsf.plot_kernel("Not Cropped")
zpsf.crop_kernel(min_part_of_max=0.025)  #  2.5% of max kernel value is used for a threshold
zpsf.plot_kernel("Cropped")
```
Originally used (estimated) kernel with size (77, 77) for Spherical aberration:

![Original Spherical aber. kernel](./src/zernpy/readme_images/(0,4)_Spherical_Original.png "Original Spherical aber. kernel (77, 77)")

Cropped kernel with size (33, 33) for Spherical aberration:

![Cropped Spherical aber. kernel](./src/zernpy/readme_images/(0,4)_Spherical_Cropped.png "Cropped Spherical aber. kernel (33, 33)")  

If a warning is encountered that estimated kernel size isn't enough for kernel calculation (it can be thrown
in the end of computation), it's possible to manually increase kernel size for next computation, e.g.:
```python
zpsf.set_calculation_props(kernel_size=51)  # as an example
```

### References
The recursive and tabular equations, along with references to the essential information about Zernike polynomials, 
are sourced from:
1. [Honarvar Shakibaei and Paramesran 2013](https://doi.org/10.1364/OL.38.002487)
2. [Lakshminarayanan and Fleck 2011](https://doi.org/10.1080/09500340.2011.554896) 
3. [Andersen 2018](https://doi.org/10.1364/OE.26.018878)

The equations for calculation of PSF are sourced from:
1. Principles of Optics, by M. Born and E. Wolf, 4 ed., 1968
2. Open Source Articles: 
[Lecture](https://wp.optics.arizona.edu/jsasian/wp-content/uploads/sites/33/2016/03/ZP-Lecture-12.pdf), 
[Thesis](https://nijboerzernike.nl/_PDF/JOSA-A-19-849-2002.pdf#[0,{%22name%22:%22Fit%22}]).
3. [Mahajan and Díaz 2013](https://doi.org/10.1364/AO.52.002062)
