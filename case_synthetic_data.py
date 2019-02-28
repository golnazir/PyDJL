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
A  = 1e-4           # APE for wave (m^4/s^2)
L, H  = 8.0, 0.2    # domain width (m) and depth (m)
NX, NZ = 32, 32     # grid

# Specify analytic density and velocity profiles
a_d, z0_d, d_d = 0.02, 0.05, 0.01
frho = lambda z: 1-a_d * (numpy.tanh((z+z0_d)/d_d))

U0  = 0.10
zj, dj  = 0.5 * H, 0.4 * H
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

#djl = DJL(A, L, H, NX, NZ, rho, rhoz)

# Now solve DJL, bringing in the background velocity incrementally
for ii, alpha in enumerate(numpy.linspace(0,1,4)):
    # Velocity profiles
    Ubg   = lambda z: alpha * Utarget  (z)
    Ubgz  = lambda z: alpha * Utargetz (z)
    Ubgzz = lambda z: alpha * Utargetzz(z)
    
    # Use a larger epsilon for these intermediate waves
    # Iterate the DJL solution
    if ii == 0:
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, epsilon = 1e-3, Ubg=Ubg, Ubgz = Ubgz, Ubgzz=Ubgzz)
    else:
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, epsilon = 1e-3, Ubg=Ubg, Ubgz = Ubgz, Ubgzz=Ubgzz, initial_guess=djl)


# Increase resolution, restore default epsilon, iterate to convergence
NX, NZ =64, 64
 # clear epsilon
djl = DJL(A, L, H, NX, NZ, rho, rhoz, Ubg=Ubg, Ubgz = Ubgz, Ubgzz=Ubgzz, initial_guess = djl)

# Increase resolution, iterate to convergence
NX, NZ =128, 128
djl = DJL(A, L, H, NX, NZ, rho, rhoz, Ubg=Ubg, Ubgz = Ubgz, Ubgzz=Ubgzz, initial_guess = djl)

# Increase to the final resolution, iterate to convergence
NX, NZ = 512,256
djl = DJL(A, L, H, NX, NZ, rho, rhoz, Ubg=Ubg, Ubgz = Ubgz, Ubgzz=Ubgzz, initial_guess = djl)

end_time = time.time()
print('Total wall clock time: %f seconds\n'% (end_time - start_time))

# Compute and plot the diagnostics
diag = Diagnostic(djl)
plot(djl, diag, 2)

