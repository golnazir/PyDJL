# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:28:03 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from scipy import interpolate
from PyDJL import DJL, Diagnostic, plot
import matplotlib.pyplot as plt

# Load density data. The text file contains an approximate reproduction of
# the smoothed curve in Figure 9 (a) from Pineda et al (2015).
# Courtesy of Jorge MagalhÃ£es and JosÃ© da Silva.
data = numpy.genfromtxt('case_pineda_cast1.txt', delimiter=',')
zdata = data[:, 0]
rhodata = data [:, 1]

rho0 = numpy.max (rhodata)  #Used in DJL module to compute N2(z)

# Use numpy's gradient function to find the derivative drho/dz
rhozdata = numpy.gradient(rhodata,zdata)

# Now build piecewise interpolating polynomials from the data,
# and convert them to function handles for the solver
#method='pchip';  % pchip respects monotonicity
#rho  = lambda z: interpolate.PchipInterpolator(zdata, rhodata )(z)
#rhoz = lambda z: interpolate.PchipInterpolator(zdata, rhozdata)(z)
rho  = lambda z: interpolate.interp1d(zdata, rhodata )(z)
rhoz = lambda z: interpolate.interp1d(zdata, rhozdata)(z)

L, H = 1200, 57   # domain width (m) and depth(m), estimated from Pineda et al's Figure 9 (a)

############################################################################
#### Find the wave showcased in Pineda et al. (2015) Figure 11 #############
############################################################################
start_time = time.time()

# Set initial resolution and large epsilon for intermediate waves
NX, NZ = 32, 32

# Raise amplitude in a few increments
for ii, A in enumerate(numpy.linspace(1e4, 3.62e5, 6)):     # APE (kg m/s^2)
    if ii == 0:
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-3)
    else:
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-3, initial_guess = djl)

# Increase resolution, reduce epsilon, iterate to convergence
NX, NZ = 64, 64
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-4, initial_guess = djl)

NX, NZ = 128, 128
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-5, initial_guess = djl)

NX, NZ = 256, 256
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-6, initial_guess = djl)

NX, NZ = 512, 512
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-7, initial_guess = djl) 

end_time = time.time()
print('Total wall clock time: %f seconds\n'%(end_time-start_time))

# Compute diagnostics, plot wave
diag = Diagnostic(djl)
plot(djl, diag, 2)
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

for ai, A in enumerate (Alist):
     # Change resolution, reduce epsilon, iterate to convergence
    if ai == 0:
        NX, NZ = 32, 32
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-3)
    else:
        NX, NZ = 32, 32
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-3, initial_guess = djl)

    NX, NZ = 64, 64
    djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-4, initial_guess = djl) 
    
#    NX, NZ = 128,128
#    djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon = 1e-5, initial_guess = djl)
#    
    # Compute and record quantities
    diag = Diagnostic(djl)
    diag.pressure(djl)

    WArec[ai] = djl.wave_ampl
    Crec[ai] = djl.c

    f1 = interpolate.interp2d(djl.XC, djl.ZC, diag.u)
    u1mab = f1(djl.xc,-djl.H+1)
    val = u1mab.flat[numpy.abs(u1mab).argmax()]
    Urec[ai] = val
    
    f2 = interpolate.interp2d(djl.XC, djl.ZC, diag.p)
    p1mab = f2(djl.xc,-djl.H+1);
    val = p1mab.flat[numpy.abs(p1mab).argmax()]
    Prec[ai] = val
    
    
end_time = time.time()
print('Total wall clock time: %f seconds\n'%(end_time - start_time))
breakpoint()
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

input("Press Enter to continue...")