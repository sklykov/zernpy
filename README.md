### zernpy - package for calculation Zernike polynomials

This project is intended for calculation of Zernike polynomials parameters / values / properties using tabular and recursive equations, that might be the
faster way to calculate values of high order polynomials in comparison to usage of the exact definition (that used the sum of factorials, 
see the [Wiki article](https://en.wikipedia.org/wiki/Zernike_polynomials) for details).    
Several useful transformations (e.g., from OSA / ANSI index to Noll one) are implemented as the methods of the main class.

### Setup instructions

#### Basic installation
For installation of this package, use the command: ***pip install zernpy***  

#### Running tests for the code from the repository
Using the library *pytest* just run in the root folder for the folder, containing the package: ***pytest***    
It should collect 8 tests and automatically runs them.

#### Requirements
For installation the *numpy* and *matplotlib* libraries are required.  
For running tests, the *pytest* library is required for automatic recognition of tests stored in package folders.  

### A few examples of the library features usage
#### Initialization of base class instance
The useful calculation methods are written as the instance and static methods. The first ones are accessible after initialization of a class instance
by providing characteristic orders (see the Zernike polynomial definition, e.g. in [Wiki](https://en.wikipedia.org/wiki/Zernike_polynomials)):   
```python  # code block for Python code
from zernpy import ZernPol
zp = ZernPol(m=-2, n=2)  
```
Alternative initializations using other indices: ***ZernPol(osa_index=3)***, ***ZernPol(noll_index=5)***, ***ZernPol(fringe_index=6)***   
For details, please, refer to the API Dictionary provided on the GitHub page (see "Documentation" tab on [pypi](https://pypi.org/project/zernpy/)).

#### Some useful class instance methods:
1) For getting all characteristic indices for the initialized polynomial: ***zp.get_indices()***   
This method returns the following tuple: *((azimuthal order, radial order), OSA index, Noll index, Fringe index)*
2) For getting the string name of the initialized polynomial (up to 7th order): ***zp.get_polynomial_name()***
3) For calculating polynomial value for polar coordinates (r, theta): ***zp.polynomial_value(r, theta)***  
Note that *r* and *theta* are accepted as float numbers or numpy.ndarrays with the equal shape.
4) For calculating radial polynomial value for radius (radii) r: ***zp.radial(r)***  
5) For calculating derivative of radial polynomial value for radius (radii) r: ***zp.radial_dr(r)***
6) For calculating triangular function value for angle theta: ***zp.triangular(theta)*** 
7) For calculating derivative of triangular function value for angle theta: ***zp.triangular_dtheta(theta)***   
8) For calculating normalization factor (N): ***zp.normf()*** 

#### Some useful static methods of ZernPol class:
1) For getting tuple as (azimuthal order, radial order) for OSA index i: ***ZernPol.index2orders(osa_index=i)***  
Same for Fringe and Noll indices: ***ZernPol.index2orders(noll_index=i)*** or ***ZernPol.index2orders(fringe_index=i)***
2) Conversion between indices: ***ZernPol.osa2noll(osa_index)***,
with similar signature: ***noll2osa(...)***, ***osa2fringe(...)***, ***osa2fringe(...)***, ***fringe2osa(...)***
3) Calculation of Zernike polynomials sum: ***ZernPol.sum_zernikes(coefficients, polynomials, r, theta, get_surface)***   
It calculates the sum of initialized Zernike polynomials (*ZernPol*) using coefficients and (r, theta) polar coordinates.
The variable *get_surface* allows returning for vector polar coordinates with different shapes the values as for mesh of these coordinates.
The details of acceptable values - see the docstring of this method or the API Dictionary.
4) Plotting the initialized Zernike polynomial (ZernPol): ***ZernPol.plot_zernike_polynomial(polynomial)***  
It plots the Zernike polynomial on unit circle polar coordinates (blocked non-interactive call of *matplotlib.pyplot.show()*)
5) Plotting Zernike polynomials sum:  ***ZernPol.plot_sum_zernikes_on_fig(...)*** - check the list of parameters in the docstring.
By using only default parameters, this method will plot sum of Zernike polynomials specified in the list with their coefficients
on the provided figure (expected as an instance of the class *matplotlib.pyplot.Figure*).
