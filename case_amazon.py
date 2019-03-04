# -*- coding: utf-8 -*-
"""
Created on Tue Feb  20 21:03:02 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from scipy import interpolate
from DJL import DJL, Diagnostic, plot
import matplotlib.pyplot as plt

# Load the Amazon data
data     = numpy.genfromtxt('case_amazon.txt', delimiter=',') #columns are z,rho,u
zdata    = data[:, 0]
rhodata0 = data[:, 1]
udata    = data[:, 2]

# Compute derivatives
rhozdata0 = numpy.gradient(rhodata0,zdata)
uzdata    = numpy.gradient(udata,zdata)
uzzdata   = numpy.gradient(uzdata,zdata)

# Density adjustments (to ensure that drho/dz is negative)
rhodata     = numpy.copy(rhodata0)  # copy original density
rhodata[-1] = rhodata[-2]*0.9999    # replace surface value to ensure positive N^2

tmp = []
for idx in range(0, len(zdata)):
    tmp += [zdata[idx]>-55 and  zdata[idx]<-30] 
rb = 0.5 * numpy.mean(rhozdata0[tmp])       # define a mean density gradient from z=(-55,-30) m

ri = (zdata<=-55).nonzero()                 # find indexes for z <= -55 m
rhodata[ri[0]] = rb*(zdata[ri[0]]- zdata[ri[0][-1]+1]) + rhodata[ri[0][-1] + 1]   # replace data below z=-55 m with linear density
rhozdata = numpy.gradient(rhodata,zdata)    # adjusted density derivative

# Show original (blue) and adjusted (red) density and first derivative
plt.figure(2)

plt.subplot(1,2,1)
plt.plot(rhodata0 , zdata , 'b', linewidth = 4)
plt.plot(rhodata ,zdata,'r',linewidth = 2)
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(rhozdata0,zdata , 'b', linewidth = 4)
plt.plot(rhozdata,zdata,'r',linewidth = 2)
plt.grid(True)

plt.tight_layout()
plt.savefig('test.png')
plt.ion()
plt.show()

# Build piecewise interpolating polynomials from the data
rho    = lambda z: interpolate.interp1d(zdata,rhodata ,fill_value="extrapolate")(z)
rhoz   = lambda z: interpolate.interp1d(zdata,rhozdata,fill_value="extrapolate")(z)
fUbg   = lambda z: interpolate.interp1d(zdata,udata   ,fill_value="extrapolate")(z)
fUbgz  = lambda z: interpolate.interp1d(zdata,uzdata  ,fill_value="extrapolate")(z)
fUbgzz = lambda z: interpolate.interp1d(zdata,uzzdata ,fill_value="extrapolate")(z)

# Now we have data, prepare for DJLES
L, H = 600, 80   # domain width (m) and depth (m)
rho0 = 1000      # reference density (kg/m^3)
#verbose = 1;

start_time = time.time()

# Start at low resolution/epsilon, no background velocity, and raise amplitude
NX, NZ = 32 , 32
A1, A2 = 1e5, 1e6
for ii, A in enumerate (numpy.linspace(A1, A2, 3)):
    if ii == 0:
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon=1e-3)
    else:
        djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon=1e-3, initial_guess = djl)

# Improve resolution, raise amplitude
NX, NZ = 64 , 64
A1, A2 = 1e6, 5e6
for A in numpy.linspace(A1, A2, 3):
    djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, epsilon=1e-3, initial_guess = djl)

# Now bring in the background velocity incrementally
for alpha in numpy.linspace(0.25,1,4):
    # Velocity profile for this wave
    Ubg  = lambda z: alpha*fUbg(z)
    Ubgz = lambda z: alpha*fUbgz(z)
    Ubgzz= lambda z: alpha*fUbgzz(z)
    djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, Ubg = Ubg, Ubgz= Ubgz, Ubgzz = Ubgzz, epsilon=1e-3, initial_guess = djl)

# Improve resolution and epsilon, raise amplitude
NX, NZ = 128, 128
A1, A2 = 5e6, 7.5e6
for A in numpy.linspace(A1, A2, 4):
    djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, Ubg = Ubg, Ubgz= Ubgz, Ubgzz = Ubgzz, relax=0.4, epsilon=1e-4, initial_guess = djl)

# Final wave: increase resolution, iterate to convergence
NX, NZ = 256, 256
djl = DJL(A, L, H, NX, NZ, rho, rhoz, rho0 = rho0, Ubg = Ubg, Ubgz= Ubgz, Ubgzz = Ubgzz, relax=0.4, epsilon=1e-6, initial_guess = djl)

# Compute diagnostics, plot wave
diag = Diagnostic(djl)
plot(djl, diag, 2)

# The wave is now near its amplitude limit and is beginning to transition
# to a broad flat crested wave. Use a larger domain for larger vales of APE.
end_time = time.time()
print ('Total wall clock time: %f seconds\n'%(end_time - start_time))
