### zernpy - package for calculation Zernike polynomials

This project is intended for calculation of Zernike polynomials using recursive equations, that may be the
faster way to calculate polynomials with high orders in comparison to calculation using exact equation.   
Several useful transformations (e.g., from OSA / ANSI index to Noll one) are implemented as the methods of the main class.

### Setup instructions

#### Basic installation
For installation of this package, use the command: ***pip install zernpy***   

#### Running tests for the code from the repository
Using the library *pytest* just run in the root folder for this code: ***pytest***   
It should collect 4 tests and automatically runs them.

### A few examples of usage
Initialization of base class instance:   
***from zernpy import ZernPol***   
***zp = ZernPol(m=-2, n=2)***   
Alternative initializations: ***ZernPol(osa_index=3)***, ***ZernPol(noll_index=5)***, ***ZernPol(fringe_index=6)***

Some useful initialized instance methods:
1) For getting all characteristic indices for the initialized polynomial: ***zp.get_indices()***  
This method returns the following tuple: *((azimuthal order, radial order), OSA index, Noll index, Fringe index*   
2) For getting the string name of the initialized polynomial (up to 7th order): ***zp.get_polynomial_name()***   
3) For getting polynomial value for polar coordinates (r, theta): ***zp.get_polynomial_name(r, theta)***    
Note that r and theta are accepted as float numbers or numpy.ndarrays with the same shape.

Some useful static methods of ZernPol class:
1) For getting tuple as (azimuthal order, radial order) for OSA index i: ***ZernPol.get_orders(osa_index=i)***   
Same for Fringe and Noll indices: ***ZernPol.get_orders(noll_index=i)*** or ***ZernPol.get_orders(fringe_index=i)***
2) Conversion between indices: ***ZernPol.osa2noll(osa_index)***,   
with similar signature: ***noll2osa(...)***, ***osa2fringe(...)***, ***osa2fringe(...)***, ***fringe2osa(...)***
3) Calculation of Zernike polynomials sum: ***ZernPol.sum_zernikes(coefficients, polynomials, r, theta, get_surface)***   
It calculates the sum of initialized Zernike polynomials (*ZernPol*) using coefficients and (r, theta) polar coordinates.    
The variable *get_surface* allows returning for vector polar coordinates with different shapes the values as for mesh of these coordinates.
The details of acceptable values - see the docstring of this method.
4) Plotting the initialized Zernike polynomial (ZernPol): ***ZernPol.plot_zernike_polynomial(polynomial)***    
It plots the Zernike polynomial on unit circle polar coordinates (blocked non-interactive call of *matplotlib.pyplot.show()*))  
5) Plotting Zernike polynomials sum:  ***ZernPol.plot_sum_zernikes_on_fig(...)*** - check the list of parameters in the docstring.    
By using only default parameters, this method will plot sum of Zernike polynomials specified in the list with their coefficients
on the provided figure (expected as an instance of the class *matplotlib.pyplot.Figure*).   