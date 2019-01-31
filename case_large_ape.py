# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:47:25 2019

@author: GolnazIR
"""
import time
import numpy
from DJL import DJL

from IPython import embed

# Specify the parameters of the problem 
Atarget = 2e-3      # APE for wave (m^4/s^2)
A  = Atarget/100
L  = 8.0            # domain width (m)
H  = 0.2            # domain depth (m)
NX = 32             # grid
NZ = 32             #grid

# The unitless density profile (normalized by a reference density rho0)
a_d  = 0.02
z0_d = 0.05
d_d  = 0.01
rho    = lambda z: 1 - a_d * numpy.tanh((z+z0_d)/d_d)
intrho = lambda z: z - a_d*d_d*numpy.log(numpy.cosh( (z+z0_d)/d_d ))
rhoz   = lambda z: -(a_d/d_d)*(1.0/numpy.cosh((z+z0_d)/d_d)**2)

# The velocity profile (zero for this case) (m/s)
Ubg  = lambda z: 0*z
Ubgz = lambda z: 0*z
Ubgzz= lambda z: 0*z

# Find the solution
start_time = time.time()

# Create DJL object
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho, Ubg, Ubgz, Ubgzz)

# Start with 1% of target APE, raise to target APE in 5 steps
for djl.A in numpy.linspace(Atarget/100, Atarget, 5):
    # Find the solution of the DJL equation
    djl.refine_solution()

# Increase the resolution, reduce epsilon, and iterate to convergence
djl.change_resolution(64,64) # NX = 64, NZ = 64
djl.refine_solution(epsilon = 1e-5)

# Increase the resolution, and iterate to convergence
djl.change_resolution(512,256)  # NX0=512,NZ0=256
djl.refine_solution()

end_time = time.time()
print('Total wall clock time: %f seconds\n'% (end_time - start_time));

# Compute and plot the diagnostics
djl.diagnostics()
djl.plot()
