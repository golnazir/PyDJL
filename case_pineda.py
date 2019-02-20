# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:28:03 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from scipy import interpolate
from DJL import DJL
from DJL import Diagnostic

import matplotlib.pyplot as plt

# Load density data. The text file contains an approximate reproduction of
# the smoothed curve in Figure 9 (a) from Pineda et al (2015).
# Courtesy of Jorge MagalhÃ£es and JosÃ© da Silva.
data = numpy.genfromtxt('case_pineda_cast1.txt', delimiter=',')
zdata = data[:, 0]
rhodata = data [:, 1]

rho0 = numpy.max (rhodata)  #Used in djles_common to compute N2(z)

# Use MATLAB's gradient function to find the derivative drho/dz
rhozdata = numpy.gradient(rhodata,zdata)

# Now build piecewise interpolating polynomials from the data,
# and convert them to function handles for the solver
#method='pchip';  % pchip respects monotonicity
rho  = lambda z: interpolate.PchipInterpolator(zdata, rhodata )(z)
rhoz = lambda z: interpolate.PchipInterpolator(zdata, rhozdata)(z)
#rho  = lambda z: interpolate.interp1d(zdata, rhodata )(z)
#rhoz = lambda z: interpolate.interp1d(zdata, rhozdata)(z)

L  =   1200   # domain width (m)
H  =     57   # domain depth (m), estimated from Pineda et al's Figure 9 (a)

############################################################################
#### Find the wave showcased in Pineda et al. (2015) Figure 11 #############
############################################################################
start_time = time.time()

# Set initial resolution and large epsilon for intermediate waves
NX = 32
NZ = 32 

A = 1e4
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0)
djl.refine_solution(epsilon = 1e-3)

# Raise amplitude in a few increments
for i in numpy.linspace(8.04e4, 3.62e5, 5):   # APE (kg m/s^2)
    djl.A = i
    djl.refine_solution(epsilon = 1e-3)


# Increase resolution, reduce epsilon, iterate to convergence
djl.change_resolution(64, 64)    # NX=64 NZ=64
djl.refine_solution(epsilon=1e-4)

djl.change_resolution(128, 128)     #NX=128; NZ=128; 
djl.refine_solution(epsilon=1e-5)

djl.change_resolution(256, 256)     #NX=256; NZ=256; 
djl.refine_solution(epsilon = 1e-6)

djl.change_resolution(512, 512)     #NX=512; NZ=512; 
djl.refine_solution(epsilon = 1e-7)

end_time = time.time()
print('Total wall clock time: %f seconds\n'%(end_time-start_time))

# Compute diagnostics, plot wave
diag = Diagnostic(djl)
#djl.plot()
diag.pressure(djl)

# Construct Pineda et al. (2015) Figure 11
plt.figure(11, figsize = (8,11))
#plt.clf()
plt.subplot(2,1,1)
plt.contour(djl.XC-djl.L/2, djl.ZC, diag.density, numpy.arange(1022.3,1024.7, 0.3), colors = 'black')
plt.xlim(-300, 300)
plt.ylim(-djl.H, 0)
plt.title('Density contours')

plt.subplot(4,1,3)
f = interpolate.RectBivariateSpline(djl.xc, djl.zc, diag.u.transpose(), kx =1 , ky = 1)
plt.plot(djl.xc-djl.L/2, f(djl.xc,-djl.H+1), color = 'black')
plt.xlim(-300, 300)
plt.ylim(-0.4, 0)
plt.grid(True)
plt.title('U at 1 mab')

plt.subplot(4,1,4)
f = interpolate.RectBivariateSpline(djl.xc, djl.zc, diag.p.transpose())
plt.plot(djl.xc-djl.L/2, f(djl.xc,-djl.H+1), color = 'black')
plt.xlim(-300, 300)
plt.grid(True) 
plt.title('Pressure minus background pressure')

plt.tight_layout()
plt.ion()
plt.show()

############################################################################
#### Find the solid curves shown in Pineda et al. (2015) Figure 10 #########
############################################################################
start_time = time.time()

#verbose=0;

Alist = numpy.logspace(numpy.log10(1e3), numpy.log10(1e6), 11)
WArec = numpy.zeros(len(Alist))    # wave amplitude
Crec  = numpy.zeros(len(Alist))    # wave phase speed
Urec  = numpy.zeros(len(Alist))    # velocity at 1 metre above bottom
Prec  = numpy.zeros(len(Alist))    # pressure at 1 metre above bottom

# Solve the wave at each amplitude in Alist
# Each wave is re-used as initial guess for subsequent waves

# First case is done individually to create DJL object.
A = Alist[0]
NX = 32
NZ = 32
djl2 = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0)

djl2.refine_solution(epsilon = 1e-3)

djl2.change_resolution(64, 64)      # NX=64; NZ=64
djl2.refine_solution(epsilon=1e-4)

djl2.change_resolution(128, 128)    # NX=128; NZ=128
djl2.refine_solution(epsilon=1e-5)

# Compute and record quantities
diag2 = Diagnostic(djl2)
diag2.pressure(djl2)

WArec[0] = djl2.wave_ampl
Crec[0]  = djl2.c

f1 = interpolate.interp2d(djl2.XC,djl2.ZC,diag2.u)
u1mab = f1(djl2.xc,-djl2.H+1)
val = u1mab.flat[numpy.abs(u1mab).argmax()]
Urec[0] = val

f2 = interpolate.interp2d(djl2.XC,djl2.ZC, diag2.p)
p1mab = f2(djl2.xc,-djl2.H+1);
val = p1mab.flat[numpy.abs(p1mab).argmax()]
Prec[0] = val

for ai in range (1,len(Alist)):
    djl2.A = Alist[ai]

    # Change resolution, reduce epsilon, iterate to convergence
    djl2.change_resolution(32, 32)      # NX=32; NZ=32;
    djl2.refine_solution(epsilon=1e-3)
     
    djl2.change_resolution(64, 64)      # NX=64; NZ=64
    djl2.refine_solution(epsilon=1e-4)

    djl2.change_resolution(128, 128)    # NX=128; NZ=128
    djl2.refine_solution(epsilon=1e-5)

    # Compute and record quantities
    diag3 = Diagnostic(djl2)
    diag3.pressure(djl2)

    WArec[ai] = djl2.wave_ampl
    Crec[ai] = djl2.c

    f1 = interpolate.interp2d(djl2.XC, djl2.ZC, diag3.u)
    u1mab = f1(djl2.xc,-djl2.H+1)
    val = u1mab.flat[numpy.abs(u1mab).argmax()]
    Urec[ai] = val
    
    f2 = interpolate.interp2d(djl2.XC, djl2.ZC, diag3.p)
    p1mab = f2(djl2.xc,-djl2.H+1);
    val = p1mab.flat[numpy.abs(p1mab).argmax()]
    Prec[ai] = val
    
    
end_time = time.time()
print('Total wall clock time: %f seconds\n'%(end_time - start_time))

# Construct Pineda et al. (2015) Figure 10
plt.clf()
plt.figure(10)
plt.subplot(3,1,1)
plt.plot(-WArec, Crec,color = 'black') 
plt.xlim(0, 23)
plt.grid(True)
plt.title('c (m/s)')

plt.subplot(3,1,2)
plt.plot(-WArec, Urec,color = 'black')
plt.xlim(0, 23)
plt.grid(True) 
plt.title('U at 1 mab (m/s)')

plt.subplot(3,1,3)
plt.plot(-WArec, Prec,color = 'black')
plt.xlim(0, 23) 
plt.grid(True)
plt.title('P at 1 mab (Pa)')

plt.tight_layout()
plt.ion()
plt.show()