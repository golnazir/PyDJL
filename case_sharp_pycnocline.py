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

frho    = lambda z,d_d: 1-a_d*numpy.tanh((z+z0_d)/d_d)
fintrho = lambda z,d_d: z - a_d*d_d*numpy.log(numpy.cosh( (z+z0_d)/d_d ))
frhoz   = lambda z,d_d: -(a_d/d_d)*(1.0/numpy.cosh((z+z0_d)/d_d))**2
    
#  Find the solution 
start_time = time.time()

# Specify resolution and pycnocline parameter d_d according to this schedule
NXlist=[  64,   128,    256,   256,     512]
NZlist=[  32,    64,    128,   128,     256]
ddlist=[0.01, 0.005, 0.0025, 0.001, 0.00075]

NX = NXlist[0]
NZ = NZlist[0]

# Density profile for this wave with specified d_d
d_d    = ddlist[0]
rho    = lambda z: frho    (z, d_d)
intrho = lambda z: fintrho (z, d_d)
rhoz   = lambda z: frhoz   (z, d_d)

#  Create DJL object
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho=intrho)
djl.refine_solution(relax=0.15)     # use strong underrelaxation

for ddindex in range (1, len(ddlist)):
    NX = NXlist[ddindex]
    NZ = NZlist[ddindex]
    print('ddindex = %d, NX = %d, NZ= %d' %(ddindex, NX, NZ))
    
    # Resolution for this wave
    djl.change_resolution(NX,NZ)
    
    # Density profile for this wave with specified d_d
    d_d  = ddlist[ddindex]
    djl.rho    = lambda z: frho    (z, d_d)
    djl.intrho = lambda z: fintrho (z, d_d)
    djl.rhoz   = lambda z: frhoz   (z, d_d)
    
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
#djl.plot()
