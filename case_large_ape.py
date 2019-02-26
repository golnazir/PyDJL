# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:47:25 2019

@author: GolnazIR
"""
import time
import numpy
from DJL import DJL, Diagnostic, plot

# Specify the parameters of the problem 
A = 2e-3         # APE for wave (m^4/s^2)
L, H = 8.0, 0.2  # domain width (m) and depth (m)
NX, NZ = 32, 32  # grid size

# The unitless density profile (normalized by a reference density rho0)
a_d, z0_d, d_d = 0.02, 0.05, 0.01
rho    = lambda z: 1 - a_d * numpy.tanh((z + z0_d) / d_d)
intrho = lambda z: z - a_d * d_d * numpy.log(numpy.cosh((z + z0_d) / d_d))
rhoz   = lambda z: -(a_d / d_d) * (1.0 / numpy.cosh((z + z0_d) / d_d) ** 2)

# Find the solution
start_time = time.time()

# Start with 1% of target APE, raise to target APE in 5 steps
for ii, Ai in enumerate(numpy.linspace(A / 100, A, 5)):
    if ii == 0:
        djl = DJL(Ai, L, H, NX, NZ, rho, rhoz, intrho=intrho)
    else:
        djl = DJL(Ai, L, H, NX, NZ, rho, rhoz, intrho=intrho, initial_guess=djl)

# Increase the resolution, reduce epsilon, and iterate to convergence
NX, NZ = 64, 64
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho=intrho, epsilon=1e-5, initial_guess=djl)

# Increase the resolution, and iterate to convergence
NX, NZ = 512,256
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho=intrho, epsilon=1e-5, initial_guess=djl)

end_time = time.time()
print('Total wall clock time: %f seconds\n'% (end_time - start_time));

# Compute and plot the diagnostics
diag = Diagnostic(djl)
plot(djl, diag, 2)

