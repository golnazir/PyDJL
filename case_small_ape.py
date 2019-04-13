import time
import numpy
from PyDJL import DJL,Diagnostic, plot

# Specify the parameters of the problem
A  = 5e-5             #APE for wave (m^4/s^2)
L , H  = 4.0, 0.2    # domain width (m) and depth (m)
NX, NZ = 32, 32      # grid

# The unitless density profile (normalized by a reference density rho0)
a_d, z0_d, d_d = 0.02, 0.05, 0.01
rho    = lambda z: 1-a_d*numpy.tanh((z+z0_d)/d_d)
intrho = lambda z: z - a_d*d_d*numpy.log(numpy.cosh( (z+z0_d)/d_d ))
rhoz   = lambda z: -(a_d/d_d)*(1.0/numpy.cosh((z+z0_d)/d_d)**2)

################################
################################ 
start_time = time.time()

#  Create DJL object
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho = intrho)

# Increase the resolution, and iterate to convergence
NX, NZ = 512, 512
djl = DJL(A, L, H, NX, NZ, rho, rhoz, intrho = intrho, epsilon=1e-6, initial_guess=djl)

end_time = time.time()

print("Total wall clock time: %f seconds" % (end_time - start_time))

# Compute and plot the diagnostics
diag = Diagnostic(djl)
plot(djl, diag, 2)

input("Press Enter to continue...")