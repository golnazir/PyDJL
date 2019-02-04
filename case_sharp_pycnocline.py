# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:45:26 2019

@author: GolnazIR
"""
import time
import numpy
from DJL import DJL

# Specify the parameters of the problem 
A  = 5e-5   # APE for wave (m^4/s^2)
L  = 3.0    # domain width (m)
H  = 0.1    # domain depth (m)

# Specify the general density profile which takes d_d as a second parameter
a_d = 0.02
z0_d= 0.025

# TO DO: intfrho ??
frho = lambda z,d_d: 1-a_d*numpy.tanh((z+z0_d)/d_d)
frhoz= lambda z,d_d: -(a_d/d_d)*(1.0/numpy.cosh((z+z0_d)/d_d))**2

# The velocity profile (zero for this case) (m/s)
Ubg  = lambda z: 0*z
Ubgz = lambda z: 0*z
Ubgzz= lambda z: 0*z

#  Find the solution 
start_time = time.time()

#  Create DJL object
djl = DJL(A, L, H, NX= 1, NZ = 1, Ubg = Ubg, Ubgz = Ubgz, Ubgzz = Ubgzz)

# Specify resolution and pycnocline parameter d_d according to this schedule
NXlist=[  64,   128,    256,   256,     512]
NZlist=[  32,    64,    128,   128,     256]
ddlist=[0.01, 0.005, 0.0025, 0.001, 0.00075]

for ddindex in range (0, len(ddlist)):
    NX = NXlist[ddindex]
    NZ = NZlist[ddindex]
    if(ddindex == 0):
        djl.prepareGrid(NX,NZ)
    else:
        # Resolution for this wave
        djl.change_resolution(NX,NZ)
    
    # Density profile for this wave with specified d_d
    d_d  = ddlist[ddindex]
    djl.rho  = lambda z: frho (z, d_d)
    djl.rhoz = lambda z: frhoz(z, d_d)
    #Iterate the DJL solution
    djl.refine_solution(relax=0.15) # use strong underrelaxation
    
    # Uncomment to view progress at each step
#    djl.diagnostics()
#    djl.plot ()

# Reduce epsilon, iterate to convergence
djl.refine_solution(epsilon=1e-5, relax=0.15)

# Raise resolution, iterate to convergence
djl.change_resolution(2048,1024)    # NX=2048; NZ=1024;
djl.refine_solution(epsilon=1e-5, relax=0.15)

end_time = time.time()
print('Total wall clock time: %f seconds\n' %(end_time - start_time))

# Compute and plot the diagnostics
djl.diagnostics()
djl.plot()
