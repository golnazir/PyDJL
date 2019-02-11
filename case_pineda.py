# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:28:03 2019

@author: Golnaz Irannejad
"""
import time
import numpy
from scipy import interpolate
from DJL import DJL

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

# The velocity profile (zero for this case) (m/s)
#Ubg=@(z) 0*z; Ubgz=@(z) 0*z; Ubgzz=@(z) 0*z;

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
djl.refine_solution(epsilon=1e-6)

#djl.change_resolution(128, 128)     #NX=128; NZ=128; 
#djl.refine_solution(epsilon=1e-5)
#
#djl.change_resolution(256, 256)     #NX=256; NZ=256; 
#djl.refine_solution(epsilon = 1e-6)
#
#djl.change_resolution(512, 512)     #NX=512; NZ=512; 
#djl.refine_solution(epsilon = 1e-7)

end_time = time.time()
print('Total wall clock time: %f seconds\n'%(end_time-start_time))

# Compute diagnostics, plot wave
djl.diagnostics()
#djl.plot()
djl.pressure()

# Construct Pineda et al. (2015) Figure 11
#figure(11)
#subplot(4,1,1:2)
#contour(XC-L/2,ZC,density,[1022.3:0.3:1024.7],'k')
#xlim([-1 1]*300); title('Density contours')
#
#subplot(4,1,3)
#plot(xc-L/2,interp2(XC,ZC,u,xc,-H+1),'k')
#xlim([-1 1]*300); grid on; title('U at 1 mab');
#
#subplot(4,1,4)
#plot(xc-L/2,interp2(XC,ZC,p,xc,-H+1),'k')
#xlim([-1 1]*300); grid on; title('Pressure minus background pressure')


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
djl2.diagnostics()
djl2.pressure()
    
WArec[0] = djl2.wave_ampl
Crec[0]  = djl2.c

#u1mab = interp2(XC,ZC,u,xc,-H+1);
#[~,idx] = max(abs(u1mab));
#Urec(ai) = u1mab(idx);
#
#p1mab = interp2(XC,ZC,p,xc,-H+1);
#[~,idx] = max(abs(p1mab));
#Prec(ai) = p1mab(idx);

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
    djl2.diagnostics()
    djl2.pressure()

    WArec[ai] = djl2.wave_ampl
    Crec[ai] = djl2.c

#    u1mab = interp2(XC,ZC,u,xc,-H+1);
#    [~,idx] = max(abs(u1mab));
#    Urec(ai) = u1mab(idx);
#
#    p1mab = interp2(XC,ZC,p,xc,-H+1);
#    [~,idx] = max(abs(p1mab));
#    Prec(ai) = p1mab(idx);

end_time = time.time()
print('Total wall clock time: %f seconds\n'%(end_time - start_time))

# Construct Pineda et al. (2015) Figure 10
#figure(10)
#subplot(3,1,1)
#plot(-WArec, Crec,'k'); 
#xlim([0 23]); 
#grid on; 
#title('c (m/s)');
#
#subplot(3,1,2)
#plot(-WArec, Urec,'k'); 
#xlim([0 23]); 
#grid on; 
#title('U at 1 mab (m/s)');
#
#subplot(3,1,3)
#plot(-WArec, Prec,'k'); 
#xlim([0 23]); 
#grid on; 
#title('P at 1 mab (Pa)');
