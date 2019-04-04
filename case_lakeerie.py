# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 21:22:54 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from scipy import interpolate
from DJL import DJL,Diagnostic, plot

# Specify the parameters of the problem 
# Load sample data file: two columns, depth and temperature
data = numpy.genfromtxt('case_lakeerie.txt', delimiter=',')

zdata = numpy.arange(-16.5, 0.5,0.5)
f = interpolate.interp1d(data[:,0], data[:,1], fill_value= 'extrapolate')
T = f(zdata)

# Convert to density with linear EOS (note: full density)
rho0 = 1000
T0   = 10
alpha= 1.7e-4
rhodata = rho0*(1 -alpha*(T-T0))

A  = 5000             # APE for wave (kg m/s^2)
L , H  = 600, 16.5    # domain width (m) and depth (m)
NX, NZ = 64 , 32      # grid

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
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0)

# Increase resolution, iterate to convergence
NX, NZ =128, 128
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, initial_guess = djl)

# Increase to the final resolution, iterate to convergence
NX, NZ =512, 512
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, initial_guess = djl)

end_time = time.time()
print('Total wall clock time: %f seconds\n'%(end_time- start_time))

# Compute and plot the diagnostics
diag = Diagnostic(djl)
plot(djl, diag, 2)

input("Press Enter to continue...")