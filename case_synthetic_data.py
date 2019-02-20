# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:58:38 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from DJL import DJL, Diagnostic, diffmatrix, plot
from scipy import interpolate

# Specify the parameters of the problem 
A  = 1e-4   # APE for wave (m^4/s^2)
L  = 8.0    # domain width (m)
H  = 0.2    # domain depth (m)
NX = 32     # grid
NZ = 32     # grid

# Specify analytic density and velocity profiles
a_d  = 0.02
z0_d = 0.05
d_d  = 0.01
frho = lambda z: 1-a_d * (numpy.tanh((z+z0_d)/d_d))

U0  = 0.10
zj  = 0.5 * H
dj  = 0.4 * H
fUbg= lambda z: U0 * numpy.tanh((z+zj)/dj)

# Now sample the density & velocity profiles to get synthetic mooring data
NDATA = 26
zdata = numpy.linspace(-H, 0, NDATA)
rhodata = frho(zdata)
ubgdata = fUbg(zdata)

# Use numerical derivatives for the gradients of the background fields
Dz  = diffmatrix(H/(NDATA-1), NDATA, 1, 'not periodic')
Dzz = diffmatrix(H/(NDATA-1), NDATA, 2, 'not periodic')
rhozdata = numpy.matmul(Dz, rhodata)
ubgzdata = numpy.matmul(Dz, ubgdata)
ubgzzdata= numpy.matmul(Dzz,ubgdata)

# Now we build piecewise interpolating polynomials from the data,
# and convert them to function handles for the solver  
rho  = lambda z: interpolate.interp1d(zdata, rhodata )(z)
rhoz = lambda z: interpolate.interp1d(zdata, rhozdata)(z)

Utarget   = lambda z: interpolate.interp1d(zdata, ubgdata)(z)
Utargetz  = lambda z: interpolate.interp1d(zdata, ubgzdata)(z)
Utargetzz = lambda z: interpolate.interp1d(zdata, ubgzzdata)(z)

# Find the solution
start_time = time.time()

djl = DJL(A, L, H, NX, NZ, rho, rhoz)

# Now solve DJL, bringing in the background velocity incrementally
for alpha in numpy.linspace(0,1,4):
    # Velocity profiles
    djl.Ubg   = lambda z: alpha * Utarget  (z)
    djl.Ubgz  = lambda z: alpha * Utargetz (z)
    djl.Ubgzz = lambda z: alpha * Utargetzz(z)

    # Use a larger epsilon for these intermediate waves
    # Iterate the DJL solution
    djl.refine_solution(epsilon = 1e-3)

# Increase resolution, restore default epsilon, iterate to convergence
djl.change_resolution(64, 64)   # NX=64; NZ=64; 
djl.refine_solution()         # clear epsilon

# Increase resolution, iterate to convergence
djl.change_resolution(128, 128)     # NX=128; NZ=128;
djl.refine_solution()

# Increase to the final resolution, iterate to convergence
djl.change_resolution(512, 256) # NX=512; NZ=256
djl.refine_solution()

end_time = time.time()
print('Total wall clock time: %f seconds\n'% (end_time - start_time))

# Compute and plot the diagnostics
diag = Diagnostic(djl)
plot(djl, diag, 2)

