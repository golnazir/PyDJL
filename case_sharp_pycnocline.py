# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:45:26 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from DJL import DJL, Diagnostic, plot

# Specify the parameters of the problem 
A  = 5e-5          # APE for wave (m^4/s^2)
L, H = 3.0, 0.1    # domain width (m) and depth (m)

# Specify the general density profile which takes d_d as a second parameter
a_d, z0_d = 0.02, 0.025
frho    = lambda z,d_d: 1-a_d*numpy.tanh((z+z0_d)/d_d)
fintrho = lambda z,d_d: z - a_d*d_d*numpy.log(numpy.cosh( (z+z0_d)/d_d ))
frhoz   = lambda z,d_d: -(a_d/d_d)*(1.0/numpy.cosh((z+z0_d)/d_d))**2
    
#  Find the solution 
start_time = time.time()

# Specify resolution and pycnocline parameter d_d according to this schedule
NXlist=[  64,   128,    256,   256,     512]
NZlist=[  32,    64,    128,   128,     256]
ddlist=[0.01, 0.005, 0.0025, 0.001, 0.00075]

for ddindex in range (0, len(ddlist)):
    NX = NXlist[ddindex]
    NZ = NZlist[ddindex]
    
    # Density profile for this wave with specified d_d
    d_d    = ddlist[ddindex]
    rho    = lambda z: frho    (z, d_d)
    intrho = lambda z: fintrho (z, d_d)
    rhoz   = lambda z: frhoz   (z, d_d)
    
    if ddindex == 0:
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho=intrho, relax = 0.15 )
    else:
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho=intrho, relax = 0.15, initial_guess=djl)
    
# Reduce epsilon, iterate to convergence
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho=intrho,relax=0.15, epsilon=1e-5, initial_guess=djl)

# Raise resolution, iterate to convergence
NX, NZ = 2048, 1024
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho=intrho,relax=0.15, epsilon=1e-5, initial_guess=djl)

end_time = time.time()
print('Total wall clock time: %f seconds\n' %(end_time - start_time))

# Compute and plot the diagnostics
diag = Diagnostic(djl)
plot(djl, diag, 2)
