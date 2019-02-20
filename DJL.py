#DJL Class

import numpy
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import scipy.fftpack
import time

#import scipy.io as sio  # TO DO: DELETE THIS LINE
#import inspect
#
#def cmp(v):
#    m = sio.loadmat("C:\\Users\\GolnazIR\\matlabToPythonPrj\\DJLES\\var.mat")
#    v1 = m[v]
#    ccc = inspect.currentframe().f_back.f_locals.items()
#    for ke,v2 in ccc:
#        if ke==v:
#            break
#    e = numpy.max(numpy.abs(v1-v2))
#    print('{} error: {}'.format(v,e))
    
class Diagnostic(object):
    """
    Diagnostic object
    """
    def __init__ (self, djl):
        """
        Constructor
        """
        self.djl = djl
        
        # Compute_diagnostic to assign rhe self variables.
        self.compute_diagnostics(self.djl)
        
        
    #########################################
    #######     shift_grid:         #########
    #########################################
    def shift_grid(self, fc,NX, NZ, symmx = None, symmz = None):
        """
        Shifts the data fc from the interior grid to the endpoint grid
        Specify symmetry as [] to skip shifting a dimension
        """
        if symmx == 'odd':
            # Compute DST-II, trim one coefficient, inverse DST-I, pad zeros at each end
            FC = scipy.fftpack.dst (fc, type = 2 , axis = 1)/(2*NX) #(2*self.NX)
            FC = FC[:, :-1]
            fc = scipy.fftpack.idst(FC, type = 1, axis = 1)
            fc = numpy.concatenate((numpy.zeros([fc.shape[0], 1]), fc, 
                                    numpy.zeros([fc.shape[0], 1])), axis= 1)
        if symmx == 'even':

            FC = scipy.fftpack.dct(fc, type=2, axis = 1)/(2*NX) #(2*self.NX)
            FC = numpy.concatenate((FC, numpy.zeros([FC.shape[0], 1])), axis = 1)
            fc = scipy.fftpack.idct(FC, type=1, axis = 1)       
        
        if symmz == 'odd':
            FC = scipy.fftpack.dst(fc, type = 2, axis = 0)/(2*NZ) #(2*self.NZ)
            FC = FC[:-1, :]
            fc = scipy.fftpack.idst(FC, type = 1, axis = 0)
            fc = numpy.concatenate((numpy.zeros([1, fc.shape[1]]), fc,
                                    numpy.zeros([1, fc.shape[1]])), axis=0)
        if symmz == 'even':
            # Compute DCT-II, pad one zero at end, inverse DCT-I
            FC = scipy.fftpack.dct(fc, type=2, axis = 0)/(2*NZ) #(2*self.NZ)            
            FC = numpy.concatenate((FC, numpy.zeros([1,FC.shape[1]])), axis = 0)
            fc = scipy.fftpack.idct(FC, type=1, axis = 0)
    
        fe = fc
        return fe
    
    #########################################
    #######        wavelength :     #########
    #########################################
    def wavelength(self, eta, NX):
        """
        L_w following via Eq 3.6 in
        Aghsaee, P., Boegman, L., and K. G. Lamb. 2010. "Breaking of shoaling 
        internal solitary waves". J. Fluid Mech. 659 289-317. doi:10.1017/S002211201000248X.
        We return 2*L_w as the wavelength
        """
        iz, ix = numpy.unravel_index(numpy.argmax(numpy.abs(eta)), eta.shape)
        etaL = eta[iz, :]
        w  = self.djl.quadweights(NX) *(self.djl.L/numpy.pi)
        Lw = numpy.sum(w*etaL) / eta[iz,ix]
        # We take wavelength as twice Lw
        self.wavelength = 2*Lw
        return self.wavelength
    
    #########################################
    #######        residual :       #########
    #########################################
    def residual(self, z, eta, c, NX, NZ ):
        """
        DJL residual using Eq 2.32 in (Stastna, 2001)
        """
        # Compute left hand side
        ETA = scipy.fftpack.dstn(eta, type = 2)/(4*NX*NZ)

        LHS = ETA * self.djl.LAP
        lhs = scipy.fftpack.idstn(LHS, type = 2)
        
        # Compute right hand side
        etax, etaz = self.djl.gradient(eta, 'odd', 'odd')
        if self.djl.Ubg is None:
            rhs = - self.djl.N2(z-eta)*eta/(c**2)
        else:
            Umc = self.djl.Ubg(z - eta)-c
            rhs = -(self.djl.Ubgz(z-eta)/Umc)*(1 - (etax**2 + (1-etaz)**2)) - self.djl.N2(z-eta)*eta/(Umc**2)
        
        # Residual
        residual = lhs-rhs
        return residual, lhs, rhs
    
    #########################################
    #######        diagnostics:     #########
    #########################################
    def compute_diagnostics(self, djl):
        """
        Computes a variety of diagnostics from the solved eta and c
        """ 
        # Compute velocities (Via SL2002 Eq 27)
        etax , etaz = djl.gradient(djl.eta, 'odd', 'odd')
        if djl.Ubg is None:
            self.u = djl.c*etaz
            self.w = -djl.c*etax
            self.uwave = self.u
        else:
            self.u = djl.Ubg (djl.ZC - djl.eta) *(1-etaz) + djl.c*etaz
            self.w = -djl.Ubg(djl.ZC - djl.eta) *( -etax) - djl.c*etax
            self.uwave = self.u - djl.Ubg (djl.ZC)
                
        # Wave kinteic energy density (m^2/s^2)
        self.kewave = 0.5*(self.uwave**2 + self.w**2)
        
        # APE density (m^2/s^2)
        self.apedens = djl.compute_apedens(djl.eta, djl.ZC)
       
        # Get gradient of u and w 
        self.ux, self.uz = djl.gradient(self.u, 'even', 'even')
        self.wx, self.wz = djl.gradient(self.w, 'even', 'odd' )

        # Surface strain rate
        uxze = self.shift_grid(self.ux, djl.NX, djl.NZ, symmz = 'even')   # shift ux to z endpoints
        self.surf_strain = -uxze[-1,:]                                    # = -du/dx(z=0)
        
        # Vorticity, density and Richardson number
        self.vorticity =self.uz - self.wx
        self.density = djl.rho(djl.ZC-djl.eta)
        self.ri = djl.N2(djl.ZC-djl.eta)/(self.uz*self.uz)
        
        #Wavelength (currently works only on interior grid)
        self.wavelength = self.wavelength(djl.eta, djl.NX)

        # Residual in DJL equation
        self.residual, LHS, RHS = self.residual(djl.ZC, djl.eta, djl.c, djl.NX, djl.NZ )

        res = numpy.max(numpy.abs(self.residual))
        lhs = numpy.max(numpy.abs(LHS))
        print('Relative residual %e ' % (res/lhs) )
    
    #########################################
    #######        pressure :       #########
    #########################################
    def pressure(self, djl):
        """
        djles_pressure.m - Pressure computations. Specifically, we compute
          a) the hydrostatic background pressure
          b) the hydrostatic pressure due to the internal solitary wave
          c) the non-hydrostatic pressure due to the internal wave
          d) residuals from substituting the results into the governing equations
        
        The pressure solve part is courtesy of Derek Steinmoeller.
        
        Note: This is only tested/verified for cases without a background current.
        ToDo: Make it work with background current cases.
        """
        #### Hydrostatic background pressure ###
        # We break rho(z) into two parts: a constant-N background state 
        # and an odd function of z:  rho(z) = rholin(z) + rhoodd(z)
        rhomax = djl.rho(-djl.H)
        rhomin = djl.rho(0)
        drho = rhomax - rhomin
        rholin = lambda z : rhomin + (z/ - djl.H)*drho
        rhoodd = lambda z : djl.rho(z) - rholin(z)
         
        # We integrate dplin/dz = -g*rholin(z) analytically to get plin(z)
        plin = lambda z: - djl.g *rhomin * z + djl.g*z**2 / (2*djl.H)*drho
  
        # Now we use a sine transform to integrate dpodd/dz = -g*rhoodd(z)
        temp = scipy.fftpack.dst(-djl.g*rhoodd(djl.zc), type = 2, axis = 0)
        tempz = - temp/djl.mso
        tempz  = numpy.roll(tempz, 1, axis = 0)
        tempz[0]  = 0
        podd = scipy.fftpack.idct(tempz, type = 2 , axis = 0)/(2*djl.NZ)
        poddze = self.shift_grid(numpy.asmatrix(podd).transpose(),djl.NX, djl.NZ, symmz = 'even' )    #get podd on z endpoint grid
        podd = podd - poddze[-1,0]     # enforce podd(0)=0
      
        # Total background pressure is plin+podd
        pbg = plin(djl.zc) + podd
        
        # #### Wave pressure ###
        
        # First find the internal solitary wave hydrostatic pressure
        rhowave = djl.rho(djl.ZC -djl.eta) -djl.rho(djl.ZC)
        
        # Integrate dph/dz = -g*rhowave vertically (odd, odd symmetry in x,z)
        temp = scipy.fftpack.dst(-djl.g*rhowave, type = 2, axis = 0)
        tempz = numpy.einsum('ij, i -> ij', temp, -1.0/djl.mso)
        tempz = numpy.roll(tempz, 1, axis = 0)
        tempz[0,:] = 0
        ph = scipy.fftpack.idct(tempz, type = 2 , axis = 0)/(2*djl.NZ)
        phze = self.shift_grid(ph, djl.NX, djl.NZ, symmz = 'even')  #get ph on z endpoint grid
        ph = ph - phze[-1,:]
        
        # Now solve for total wave pressure
        T1 = -((self.u-djl.c)*self.ux + self.w*self.uz)   # odd, even symmetry in x,z
        T2 = -((self.u-djl.c)*self.wx + self.w*self.wz + (djl.g/djl.rho0)*rhowave)   #even, odd symmetry in x,z
        
        # Get divergence of T vector
        Tx, tmpz  = djl.gradient(T1,  'odd', 'even')
        tmpx, Tz  = djl.gradient(T2, 'even',  'odd')
        
        # The Poisson problem has even, even symmetry in x,z
        RHS = scipy.fftpack.dctn(djl.rho0*(Tx+Tz), type = 2)
        P = RHS * djl.INVLAPE
        self.p = scipy.fftpack.idctn(P, type = 2 )/(4*djl.NZ*djl.NX)
        self.p = self.p - self.p[-1,-1]  # Use p(inf, 0) = 0 (no wave pressure in the farfield)
        pnh = self.p - ph                # non-hydrostatic part

        #### Residuals ####
        # Now we verify that the results are a steady state solution to
        # the Boussinesq equations (trend terms and divergence should be zero)
        px   ,pz   = djl.gradient(self.p , 'even', 'even')
        tmpx ,pnhz = djl.gradient(pnh    , 'even', 'even')
        tmpx ,phz  = djl.gradient(ph     , 'even', 'even')
        rwx  ,rwz  = djl.gradient(rhowave, 'odd' , 'odd' )
        
        # Boussinesq equations
        ut  = -px/djl.rho0 -(self.u-djl.c)*self.ux - self.w*self.uz
        wt  = -pz/djl.rho0 -(self.u-djl.c)*self.wx - self.w*self.wz - (djl.g/djl.rho0)*rhowave
        div = self.ux + self.wz
        rt  = -(self.u-djl.c)*rwx - self.w*(djl.rhoz(djl.ZC) + rwz)
        
        # Also verify that the split pressure satisfies the vertical
        # momentum eqn and hydrostatic balance
        wtnh = -pnhz/djl.rho0 -(self.u-djl.c)*self.wx - self.w*self.wz
        hb   = -phz /djl.rho0 -(djl.g/djl.rho0)*rhowave
        
        # Compute relative errors for each
        ue = numpy.max(numpy.abs(ut)) / numpy.max(numpy.abs(-px/djl.rho0))
        we = numpy.max(numpy.abs(wt)) / numpy.max(numpy.abs(-pz/djl.rho0))
        de = numpy.max(numpy.abs(div))/ numpy.max(numpy.abs(self.ux))
        re = numpy.max(numpy.abs(rt)) / numpy.max(numpy.abs(-(self.u-djl.c)*rwx))
        
        wnhe=numpy.max(numpy.abs(wtnh)) / numpy.max(numpy.abs(-pnhz/djl.rho0))
        hbe =numpy.max(numpy.abs(hb))   / numpy.max(numpy.abs(phz))
        
        print('ut, wt, div, rho_t residuals: %.1e, %.1e, %.1e, %.1e\n'%(ue,we,de,re))
        print('wnht, hb, residuals: %.1e, %.1e\n'%(wnhe,hbe))


    
class DJL(object):
    """
    DJL object
    """
    ################################
    #######  djles_common: #########
    ################################
        
    def __init__(self, A, L, H, NX, NZ, rho, rhoz, intrho=None, rho0 = 1, Ubg=None, Ubgz = None, Ubgzz=None):
        """
        Constructor
        """
        self.A = A 	#APE for wave (m^4/s^2)
        self.L = L 	#domain width (m)
        self.H = H 	#domain depth (m)
              

        self.rho = rho
        self.rhoz = rhoz
        self.intrho = intrho

        self.Ubg   = Ubg
        self.Ubgz  = Ubgz
        self.Ubgzz = Ubgzz

        #  Default min and max number of iterations for the iterative procedure.
        self.min_iteration = 10
        self.max_iteration = 2000

        # Default number of Legendre points for Gauss quadrature
        self.NL = 20

        # Underrelaxation factor, 0 < relax <= 1
        # Setting this can help in finding a solution and/or speeding
        # convergence. This value is interpreted as the fraction of the new
        # field to keep after an iteration completes, that is,
        # val = (1-relax)*oldval + (relax)*newval
        # Setting relax=0 prevents the iterations from proceeding, and setting
        # relax=1 simply disables underrelaxation. The default value of 0.5 works
        # well for most cases.
#        self.relax = 0.5

        # Gravitational acceleration constant (m/s^2)
        self.g = 9.81

        # Verbose flag. If >=1, we display a report on solving progress
        # If >=2, display a timing report
#        self.verbose = 1

        # Convergence criteria: Stop iterating when the relative difference between
        # successive iterations differs by less than epsilon
#        self.epsilon = 1e-4

        # If rho0 is not speficied, assume we are using non-dimensional density
        self.rho0 = rho0
        
        # Get Legendre points and weights for Gauss quadrature
        self.zl, self.wl = numpy.polynomial.legendre.leggauss(self.NL)
        self.zl = (self.zl+1)/2
        self.wl = self.wl/2
        
        self.prepareGrid(NX, NZ)
        
        self.eta = None
        self.c = None
        
    def N2(self, z):
        """
        Create N2(z) function
        """
        return (-self.g/self.rho0)*self.rhoz(z)
    
    def prepareGrid(self, NX, NZ):
        """
     	Generate the grid and wavenumbers
     	"""
        self.NX = NX
        self.NZ = NZ
        
        # Prepare the grids: (xc,zc) are cell centred, (xe,ze) are cell edges
        self.dx = self.L/self.NX
        self.dz = self.H/self.NZ
        
        self.xc = numpy.arange(0.5, self.NX, 1)*self.dx
        self.zc = numpy.arange(0.5, self.NZ, 1)*self.dz - self.H
        self.XC, self.ZC = numpy.meshgrid(self.xc,self.zc)
        
        self.xe = numpy.arange(0, self.NX, 1)*self.dx
        self.ze = numpy.arange(0, self.NZ, 1)*self.dz - self.H
        self.XE, self.ZE = numpy.meshgrid(self.xe,self.ze)
        
        # Get 2D weights for sine rectangle quadrature
        self.wsine = self.sinequadrature()

        # DST-II wavenumbers go from 1 to NX-1 or NZ-1 (beacause the function is odd)
        self.kso = (numpy.pi / self.L) * numpy.arange(1, self.NX+1)
        self.mso = (numpy.pi / self.H) * numpy.arange(1, self.NZ+1)
        
        self.kse = (numpy.pi / self.L) * numpy.arange(0, self.NX)
        self.mse = (numpy.pi / self.H) * numpy.arange(0, self.NZ)

        ksm = numpy.tile(self.kso,(self.NZ ,1))
        msm = numpy.tile(self.mso,(self.NX, 1)).transpose()
        self.LAP = -ksm**2 - msm**2
        self.INVLAP = 1/self.LAP

        ksme = numpy.tile(self.kse,(self.NZ ,1))
        msme = numpy.tile(self.mse,(self.NX, 1)).transpose()
        self.LAPE = -ksme**2 - msme**2
        self.LAPE[0,0] = 1
        self.INVLAPE = 1/self.LAPE
        self.INVLAPE[0,0] = 0
        
    #########################################
    #######     sinequadrature:      ########
    #########################################
    def sinequadrature(self):
        """
        Gets the weights for 2D quadrature over the domain
        """
        wtx = self.quadweights(self.NX)*(self.L/numpy.pi)
        wtz = self.quadweights(self.NZ)*(self.H/numpy.pi)
        w = numpy.einsum('i,j->ij',  wtz, wtx)
        return w
    
    #########################################
    #######          quadweights:    ########
    #########################################
    def quadweights(self, N):
        """
        Quadrature weights for integrating an odd periodic function over a
        half period using the interior grid. From John Boyd's Chebyshev and
        Fourier Spectral Methods, 2nd Ed, Pg 568, Eq F.32.
        """
        w = numpy.zeros(N)
        for j in range(N):
            xj = (2*(j+1) -1) * numpy.pi/ (2* N)
            m = numpy.array([i for i in range(1, N)])
            s = numpy.sum( numpy.sin(m*xj)*(numpy.sin(m*numpy.pi/2)**2)/m)
            w[j] = (2/N/N)*numpy.sin(N*xj)*(numpy.sin(N*numpy.pi/2)**2) + (4/N)*s
        return w
    
    #########################################
    #######     initial_guess:      #########
    #########################################
    def initial_guess(self):
        """
        Finds an initial guess for eta and c via weakly nonlinear theory
        """
        eta = 0
        Dz   = diffmatrix(self.dz, self.NZ, 1, 'not periodic');
        Dzz  = diffmatrix(self.dz, self.NZ, 2, 'not periodic');
        Dzzc = Dzz[1:-1, 1:-1] # cut the endpoints for Dirichlet conditions

        #get n2, u and uzz data
        tmpz   = self.zc[1:-1]      # column vector
        n2vec  = self.N2(tmpz) 

        #create diagonal matrices
        N2d  = numpy.diag(n2vec )
        
        if (self.Ubg is None):
            #setup quadratic eigenvalue problem
            B0 = sparse.csr_matrix(N2d)
            B1 = sparse.csr_matrix((self.NZ-2, self.NZ-2))
            B2 = sparse.csr_matrix(Dzzc)
        else:
            uvec   = self.Ubg(tmpz)
            uzzvec = self.Ubgzz(tmpz)
            
            Ud   = numpy.diag(uvec  )
            Uzzd = numpy.diag(uzzvec)
            
            #setup quadratic eigenvalue problem
            B0 = sparse.csr_matrix(N2d + Ud*Ud*Dzzc - Ud*Uzzd)
            B1 = sparse.csr_matrix(-2*Ud*Dzzc + Uzzd)
            B2 = sparse.csr_matrix(Dzzc)
      
        #Solve eigenvalue problem; extract first eigenvalue & eigenmode
        Z = sparse.csr_matrix((self.NZ-2, self.NZ-2))
        I = sparse.eye(self.NZ-2)
        AA =sparse.bmat([[-B0, Z], [Z, I]], format = 'csr')
        BB =sparse.bmat([[B1, B2], [I, Z]], format = 'csr' )
        cc, V = scipy.linalg.eig(AA.todense() , BB.todense())
        ii = numpy.argmax(cc)
        V2 = V[:self.NZ-2, ii]
        
        #largest eigenvalue is clw
        clw = cc[ii].real
        
        #Add boundary conditions
        phi = numpy.pad(V2, (1,1), 'constant')
        if self.Ubg is None:
            uvec = numpy.zeros(phi.shape)
        else:
            uvec = numpy.pad(uvec, (1,1), 'constant', constant_values = (self.Ubg(-self.H), self.Ubg(0)))
        
        #Compute E1, normalise
        E1 = numpy.divide(clw*phi , clw-uvec)
        E1 = numpy.abs(E1) / numpy.max(numpy.abs(E1))
        
        # Compute r10 and r01
        E1p = numpy.matmul(Dz, E1)
        E1p2 = E1p**2
        E1p3 = E1p**3
        bot = numpy.sum((clw-uvec)*E1p2)
        r10 = (-0.75/clw)*numpy.sum((clw-uvec)*(clw-uvec)*E1p3)/bot
        r01 = -0.5*sum((clw-uvec)*(clw-uvec)*E1*E1)/bot

        #Now optimise the b0, lambda parameters
        tmpE1 = numpy.asmatrix(E1).transpose()
        E = numpy.matlib.repmat(tmpE1, 1, self.NX)
        E[:, 0] = 0
        E[:,-1] = 0
        b0 = numpy.sign(r10)*0.05*self.H  #Start b0 as 5% of domain height
        la = numpy.sqrt( -6*r01 / (clw * r10 * b0) )
        c = (1 + (2/3) * r10 * b0)*clw
        
        flag = 1
        while flag > 0 :
            flag = flag + 1
            eta0 = -b0 * numpy.asarray(E) * (1.0/numpy.cosh((self.XC - self.L/2) / la))**2
            eta0[0, :] = 0
            eta0[:, 0] = 0
            eta0[:,-1] = 0
            eta0[-1,:] = 0
            
            # Find the APE (DSS2011 Eq 21 & 22)
            self.apedens = self.compute_apedens(eta0, self.ZC)
            F = numpy.sum(self.wsine * self.apedens)

            # New b0 by rescaling
            afact = numpy.max([numpy.min([self.A/F , 1.05]), 0.95])
            b0 = b0 * afact
            
            # New c, lambda
            c = (1 + (2/3) * r10 * b0) * clw
            la = numpy.sqrt( -6 * r01 / (clw * r10 * b0) )
            if not numpy.isreal(la):
                print('!problem finding new lambda-la !!')
                
            #Stop conditions: the wave gets too big, or we get matching APE
            if numpy.abs(b0) > 0.75*self.H or numpy.abs(afact-1) < 0.01 :
                eta = eta0
                flag= 0
        self.eta = eta
        self.c = c
    
    #########################################
    #######   compute_apedens:      #########
    #########################################
    def compute_apedens(self, eta, z ):
        """
        Use Gauss quadrature to find ape density
        """        
        if self.intrho is None:
            A1 = self.rho(z - eta)
            A2 = 0.0
            for ii in range (0, numpy.size(self.wl)):
                A2 -= self.wl[ii] *self.rho(z-self.zl[ii]*eta)
            apedens = (A1+A2)*eta
        else:
            B1 = self.rho(z-eta)*eta
            B2 = self.intrho(z-eta) - self.intrho(z)
            apedens = B1 + B2

        return (self.g * apedens)
 
    #########################################
    #####     change_resolution:    #########
    #########################################
    def change_resolution(self, NX0, NZ0):
        """
        Changes the resolution of eta to NZxNX
        Then update grid info
        """
        # Change eta in NX dim
        if not NX0 == self.NX:
            ETA = scipy.fftpack.dst(self.eta, type=2, axis = 1)/(2*self.NX)
            # Increase resolution: pad with zeros
            if NX0 > self.NX :
                ETA0 = numpy.concatenate([ETA, numpy.zeros([self.NZ, NX0-self.NX])], axis= 1)
            # Decrease resolution: drop highest coefficients
            if NX0 < self.NX:
                ETA0 = ETA[:,:NX0]
            # Inverse DST-II
            self.eta = scipy.fftpack.idst(ETA0,type=2, axis = 1)
        self.prepareGrid(NX0, self.NZ)

        #Change eta in NZ dim
        if not NZ0 == self.NZ:
            ETA = scipy.fftpack.dst(self.eta, type=2, axis = 0)/(2*self.NZ)
            # Increase resolution: pad with zeros
            if NZ0 > self.NZ:
                ETA0 = numpy.concatenate([ETA, numpy.zeros([NZ0-self.NZ, self.NX])], axis = 0)
            # Decrease resolution: drop highest coefficients
            if NZ0 < self.NZ:
                ETA0 = ETA[:NZ0, :]
            # Inverse DST-II
            self.eta = scipy.fftpack.idst(ETA0,type=2, axis = 0)
        self.prepareGrid(self.NX, NZ0)    

    #########################################
    #######       gradient:         #########
    #########################################
    def gradient(self, f, symmx, symmz): 
        """
        Computes gradient using the specified symmetry and grid type
        """
        if (symmx == 'odd'):            
            # x derivative
            ftrx = scipy.fftpack.dst (f , type = 2, axis = 1)
            Fpx = numpy.einsum('ij,j-> ij', ftrx, self.kso)
            # Now we have DCT-II coefficients. However we're missing the first
            # one (k=0), so we need to prepend a zero to the start. 
            #To do so, we roll right by 1 and reset first value to zero
            Fpx = numpy.roll(Fpx,1, axis = 1)
            Fpx[:, 0] = 0            
            fx = scipy.fftpack.idct(Fpx, type = 2, axis = 1)/(2*self.NX)
        else:
            ftrx = scipy.fftpack.dct(f, type = 2, axis = 1)         
            Fpx = numpy.einsum('ij,j-> ij', ftrx, -self.kse)

            # Now we have DCT-II coefficients, we need to drop the first one
            Fpx = numpy.roll(Fpx,-1, axis = 1)
            fx  = scipy.fftpack.idst(Fpx, type = 2, axis = 1)/(2*self.NX)

        if (symmz == 'odd'):
            # z derivative
            ftrz = scipy.fftpack.dst (f , type = 2, axis = 0)
            Fpz  = numpy.einsum('ij, i -> ij' , ftrz , self.mso)
            Fpz  = numpy.roll(Fpz, 1, axis = 0)
            Fpz[0, :]  = 0
            fz = scipy.fftpack.idct(Fpz, type = 2 , axis = 0)/(2*self.NZ)
            
        else : 
            ftrz = scipy.fftpack.dct(f, type = 2, axis = 0)
            Fpz = numpy.einsum('ij, i -> ij' , ftrz , -self.mse)
            Fpz = numpy.roll(Fpz,-1, axis = 0 )
            fz  = scipy.fftpack.idst(Fpz, type= 2, axis = 0)/(2*self.NZ)

        return fx, fz

        
    #########################################
    #######     refine_solution:    #########
    #########################################
    def refine_solution(self ,epsilon = 1e-4, relax = 0.5):
        """
        Iterates eta to convergence following the procedure described
        in Stastna and Lamb, 2002 (SL2002) and also in Dunphy, Subich 
        and Stastna 2011 (DSS2011).
        """
        #Get initial guess from WNL theory if needed
        if self.eta is None:
            print ("well, eta ISNOT defined after all!")
            self.initial_guess()
        else:
            print ("sure, eta is defined.")
        
        #Check for nonzero velocity profile, save time below if it's zero
      
        flag = True; iteration = 0;
        while flag: 
            # Iteration shift
            iteration = iteration + 1
            eta0 = self.eta
            c0   = self.c
            la0  = self.g * self.H / (self.c * self.c)
            # Compute S (DSS2011 Eq 19)
            S = self.N2(self.ZC - eta0) * eta0/(self.g*self.H)

            # Compute R, assemble RHS
            if self.Ubg is not None:
                eta0x, eta0z = self.gradient(eta0, 'odd', 'odd')
                uhat  = self.Ubg (self.ZC - eta0)/c0
                uhatz = self.Ubgz(self.ZC - eta0)/c0
                R = (uhatz / (uhat -1)) * ( 1 - (eta0x**2 + (1-eta0z)**2))
                rhs = - ( la0 * (S/((uhat-1)**2))  + R )
            else:
                rhs = - la0 * S  # RHS of (DSS2011 Eq 18)
              
            #Solve the linear Poisson problem (DSS2011 Eq 18)
            t0 = time.time()

            # Compute DST-II
            RHS = scipy.fftpack.dstn(rhs, type = 2)
            NU  = RHS * self.INVLAP
            # Inverse 
            nu = scipy.fftpack.idstn(NU, type = 2) / (4 * self.NX * self.NZ)
            t1 = time.time()
            t_solve = t1 - t0

            # Find the APE (DSS2011 Eq 21 & 22)
            t0 = time.time()
            self.apedens = self.compute_apedens(eta0, self.ZC)
            F = numpy.sum(self.wsine[:]* self.apedens[:])

            #Compute S1, S2 (components of DSS2011 Eq 20)
            S1 = self.g * self.H * self.rho0 * numpy.sum( self.wsine[:]* S[:]* nu[:])
            S2 = self.g * self.H * self.rho0 * numpy.sum( self.wsine[:]* S[:]* eta0[:])
            t1 = time.time()
            t_int = t1 - t0
            
            # Find new lambda (DSS2011 Eq 20)
            la = la0 * (self.A - F + S2)/S1

            # check if lambda is OK
            if (la < 0):
                print('New lambda has wrong sign --> nonconvergence of iterative procedure\n')
                print('new lambda = %1.6e\n'% (la))
                print('   A = %1.10e, F = %1.10e\n' %(self.A, F))
                print('   S1 = %1.8e, S2 = %1.8e, S2/S1 = %1.8e\n'% (S1,S2,S2/S1))
                break
  
            # Compute new c, eta
            self.c   = numpy.sqrt(self.g * self.H / la)      # DSS2011 Eq 24
            self.eta = (la/la0)*nu                           # (DSS2011 Eq 23)
        
            # Apply underrelaxation factor
            self.eta = (1-relax)*eta0 + relax*self.eta

            # Find wave amplitude
            self.wave_ampl = self.eta.flat[numpy.abs(self.eta).argmax()]
            
            # Compute relative difference between present and previous iteration
            reldiff = numpy.max ( numpy.abs(self.eta - eta0)) / numpy.abs(self.wave_ampl)
        
        #    % Report on state of the operation
        #    if (verbose >=1)
        #        fprintf('Iteration %4d:\n',iteration);
        #        fprintf(' A       = %+.10e, wave ampl = %+16.10f m\n',A,wave_ampl);
        #        fprintf(' F       = %+.10e, c         = %+16.10f m/s\n',F,c);
        #        fprintf(' reldiff = %+.10e\n\n',reldiff);
        #    end
        #
            # Stop conditions
#            if iteration == 3: 
#                flag = False
            if (iteration >= self.min_iteration) and (reldiff < epsilon):
                flag = False
            if (iteration >= self.max_iteration):
                flag = False
                print("Reached maximum number of iterations (%d >= %d)\n" %(iteration,self.max_iteration))
           
        #
        #t.stop = clock; t.total = etime(t.stop, t.start);
        #% Report the timing data
        #if (verbose >= 2)
        #    fprintf('Poisson solve time: %6.2f seconds\n', t.solve);
        #    fprintf('Integration time:   %6.2f seconds\n', t.int);
        #    fprintf('Other time:         %6.2f seconds\n', t.total - t.solve - t.int);
        #    fprintf('Total:              %6.2f seconds\n', t.total);
        #end
        #

        print('Finished [NX,NZ]=[%3dx%3d], A=%g, c=%g m/s, wave amplitude=%g m\n' % (self.NX, self.NZ, self.A, self.c, self.wave_ampl))
    
        
#########################################
#######            plot :       #########
#########################################
def plot( djl, diag, plottype):
    plt.clf()
    
    axis_array = []
    if (plottype == 1 ):
        # Plot eta and density(z-eta)
        axis_array += [plt.subplot(1,2,1)]
        plt.pcolormesh(djl.XE, djl.ZE, djl.eta)
        plt.title('eta (m)')
        
        axis_array += [plt.subplot(1,2,2)]
        plt.pcolormesh(djl.XE, djl.ZE, diag.density)
        plt.title('density')
            
        for ca in axis_array:
            plt.sca(ca)
            plt.gca().invert_yaxis()
            plt.xlim(0, djl.L)
            plt.ylim(-djl.H, 0)
            plt.colorbar(aspect = 50) 
            plt.xlabel('x (m)')
            plt.ylabel('z (m)')
            
        
    elif (plottype == 2): 
        # Plot eight fields
        axis_array += [plt.subplot(4,2,1)]
        plt.pcolormesh(djl.XE, djl.ZE, djl.eta)
        plt.title('eta (m)')
        
        axis_array += [plt.subplot(4,2,2)]
        plt.pcolormesh(djl.XE, djl.ZE, diag.density)
        plt.title('density')
        
        axis_array += [plt.subplot(4,2,3)]
        c = numpy.max(numpy.abs(diag.uwave))
        plt.pcolormesh(djl.XE, djl.ZE,diag.uwave, cmap = 'seismic', vmin = -c, vmax = c)
        plt.title('u (wave) (m/s)')
            
        axis_array += [plt.subplot(4,2,4)]
        c = numpy.max(numpy.abs(diag.w))
        plt.pcolormesh(djl.XE, djl.ZE,diag.w, cmap = 'seismic',vmin = -c, vmax = c)
        plt.title('w (m/s)')
    
        axis_array += [plt.subplot(4,2,5)]
        plt.pcolormesh(djl.XE, djl.ZE,diag.kewave, cmap= 'hot', vmin =0, vmax = numpy.max(numpy.abs(diag.kewave)))
        plt.title('kewave (m^2/s^2)')
    
        axis_array += [plt.subplot(4,2,6)]
        plt.pcolormesh(djl.XE, djl.ZE,diag.apedens, cmap= 'hot', vmin =0, vmax = numpy.max(numpy.abs(diag.apedens)))
        plt.title('ape density');
        
        axis_array += [plt.subplot(4,2,7)]
        plt.pcolormesh(djl.XE, djl.ZE,diag.ri, vmin = 0.2, vmax = 1)
        plt.title('Ri')
        
        axis_array += [plt.subplot(4,2,8)]
        c = numpy.max(numpy.abs(diag.vorticity))
        plt.pcolormesh(djl.XE, djl.ZE,diag.vorticity,cmap = 'seismic', vmin = -c, vmax = c)
        plt.title('vorticity (1/s)')
        
        for ca in axis_array:
            plt.sca(ca)
            plt.gca().invert_yaxis()
            plt.xlim(0, djl.L)
            plt.ylim(-djl.H, 0)
            plt.colorbar(aspect = 5) 
            plt.xlabel('x (m)')
            plt.ylabel('z (m)')

    plt.tight_layout()
    plt.ion()
    plt.show()



#########################################
#######        diffmatrix:      #########
#########################################
def diffmatrix(dx, N, order, ends):
    """
    Constructs a centered differentiation matrix with up/down wind
    at the ends
    - order can be 1 (1st deriv) or 2 (2nd deriv)
    """
    a = [0,0,0]
    a [order] = 1
    dx2 = dx*dx
    
    #Upwind 2nd order coefs
    matu  =[[1    , 1    , 1],
            [-2*dx,-dx   , 0],
            [2*dx2, dx2/2, 0]]
    coefsu = numpy.linalg.solve(matu,a)
    
    #Centered 2nd order coefs
    matc = [[1    , 1, 1     ],
            [-dx  , 0, dx    ],
            [dx2/2, 0, dx2/2 ]]
    coefsc = numpy.linalg.solve(matc, a)

    #Downwind 2nd order coefs
    matd=[[1, 1    , 1    ],
          [0, dx   , 2*dx ],
          [0, dx2/2, 2*dx2]]
    coefsd = numpy.linalg.solve(matd, a)

    D = numpy.zeros((N,N))
    
    #Centered in the interior
    for ii in range (1,N-1):
        D[ii, [ii-1, ii+0, ii+1]] = coefsc
    
    if ends == 'periodic':
        #use centered at the ends (wrap around)
        D[0  , [N-1, 0  , 1]] = coefsc
        D[N-1, [N-2, N-1, 0]] = coefsc
    else:
        #Downwind at the first point
        D[0, [0, 1, 2]] = coefsd
        #Upwind at the last point
        D[N-1, [N-3, N-2, N-1]] = coefsu            
    return D
