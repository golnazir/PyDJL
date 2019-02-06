# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 21:22:54 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from scipy import interpolate
from DJL import DJL

# Specify the parameters of the problem 
# Load sample data file: two columns, depth and temperature
data = numpy.genfromtxt('case_lakeerie.txt', delimiter=',')
zdata = data[:,0]
T = data[:,1]

# Find a value for z=0 by linear extrapolation
f = interpolate.interp1d(zdata, T, fill_value= 'extrapolate')
T = numpy.append(T, f(0))
zdata = numpy.append(zdata, 0)

# Convert to density with linear EOS (note: full density)
rho0 = 1000
T0   = 10
alpha= 1.7e-4
rhodata = rho0*(1 -alpha*(T-T0))

A  = 5000   # APE for wave (kg m/s^2)
L  = 600    # domain width (m)
H  = 16.5   # domain depth (m)
NX = 64     # grid
NZ = 32     # grid

#verbose=1;
#
# Use python's gradient function to find the derivative drho/dz
rhozdata = numpy.gradient(rhodata,zdata)

# Now build piecewise interpolating polynomials from the data,
# and convert them to function handles for the solver
#rho  = lambda z: interpolate.PchipInterpolator(zdata, rhodata )(z)
#rhoz = lambda z: interpolate.PchipInterpolator(zdata, rhozdata)(z)
rho  = lambda z: interpolate.interp1d(zdata, rhodata )(z) 
rhoz = lambda z: interpolate.interp1d(zdata, rhozdata)(z)

# Find the solution
start_time = time.time()
djl =DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0)
djl.refine_solution()

# Increase resolution, iterate to convergence
djl.change_resolution(128, 128)     # NX=128; NZ=128;
djl.refine_solution()

# Increase to the final resolution, iterate to convergence
djl.change_resolution( 512, 512)    #NX=512; NZ=512;
djl.refine_solution()

end_time = time.time()
print('Total wall clock time: %f seconds\n'%(end_time- start_time))
breakpoint()
# Compute and plot the diagnostics
djl.diagnostics()
#djl.plot()
