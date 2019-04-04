# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:03:45 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from DJL import DJL,Diagnostic, plot

# Specify the parameters of the problem 
A  = 1e-4           # APE for wave (m^4/s^2)
L , H  = 8.0, 0.2   # domain width (m), and depth(m)
NX, NZ = 32 , 32    # grid

# The unitless density profile (normalized by a reference density rho0)
a_d, z0_d, d_d= 0.02, 0.05, 0.01
rho    = lambda z: 1-a_d*numpy.tanh((z+z0_d)/d_d)
intrho = lambda z: z - a_d*d_d*numpy.log(numpy.cosh( (z+z0_d)/d_d ))
rhoz   = lambda z: -(a_d/d_d)*(1.0/numpy.cosh((z+z0_d)/d_d)**2)

# Specify general velocity profile that takes U0 as a second parameter (m/s)
zj, dj= 0.5*H, 0.4*H
fUbg  = lambda z,U0: U0*numpy.tanh((z+zj)/dj)
fUbgz = lambda z,U0: (U0/dj)*(1.0/numpy.cosh((z+zj)/dj)**2)
fUbgzz= lambda z,U0: (-2*U0/(dj*dj))*(1.0/numpy.cosh((z+zj)/dj)**2)*numpy.tanh((z+zj)/dj)

# Find the solution
start_time = time.time()

#  Create DJL object
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho= intrho)

# Start with U0=0, raise it to U0=0.1 over 6 increments
for U0 in numpy.linspace(0, 0.1, 6):
    # Velocity profile for this wave
    djl.Ubg  = lambda z: fUbg  (z, U0)
    djl.Ubgz = lambda z: fUbgz (z, U0)
    djl.Ubgzz= lambda z: fUbgzz(z, U0)
    
    # Find the solution of the DJL equation
    # Use a reduced epsilon for these intermediate waves
    djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho= intrho, epsilon = 1e-3, initial_guess = djl)
 
# Increase the resolution, reduce epsilon, iterate to convergence
NX, NZ = 64, 64
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho= intrho, epsilon =  1e-6, initial_guess = djl)

# Increase the resolution and iterate to convergence
NX, NZ = 512, 256
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho= intrho, epsilon =  1e-6, initial_guess = djl)

end_time = time.time()
print('Total wall clock time: %f seconds\n' %(end_time - start_time))

# Compute and plot the diagnostics
diag = Diagnostic(djl)
plot(djl, diag, 2)

input("Press Enter to continue...")