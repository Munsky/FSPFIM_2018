import sys
sys.path.append('../')
from scipy.sparse import spdiags,csc_matrix
import numpy as np
from expv import expv
import generic_solvers as gs
#reload(gs)
from generic_solvers import GenericODE,GenericFSP,GenericSSA,GenericMoments
from scipy.linalg import expm
from sympy import Matrix,zeros,Symbol 
from scipy.special import gamma

class BirthDeath:
    '''
    Generic birth/death biochemical reaction network.
    params: kr is the production rate, g is the degradation rate. 
    '''
    def __init__(self,params):
        kr,g = params
        self.params = np.array(params)
        self.observables = np.array([0])
        self.kr = kr
        self.g = g
        # for now, use some default parameters
        self.N = 1  
        self.ti = 0
        self.tf = 10
        self.ptimes = 15
        self.xi = np.array([[0]]) 
        self.type = 'linear'
        self.N_pars = 2
        self.tv = False
        self.Nc = 100

    def get_S(self):
        '''
        get the stoichiometry matrix
        '''
        self.S = np.array([ [ 1,-1 ] ])
        
    def get_W(self): 
        ''' 
        get propensity matrices w0 and w1
        '''
        self.W1 = np.array([ [0],[self.g]])
        self.W0 = np.array([ [self.kr],[0]])
   
    def get_S_sym(self):
        '''
        get symbolic stoichiometry matrix
        '''     
        self.S_sym = Matrix([[1,-1]])
    
    def get_W_sym(self):
        '''
        get symbolic W matrices.
        '''
        kr = Symbol('kr')
        g = Symbol('g')
        self.theta_vec = Matrix([kr,g])
        self.sym_pars = self.theta_vec
        self.W1_sym = Matrix([[0],[g]])
        self.W0_sym = Matrix([[kr],[0]])
    
class BirthDeathFSP(GenericFSP):    
    '''
    Generic birth/death biochemical reaction network, 
    except easy for doing finite state projection analsyis.
    params: kr is the production rate, g is the degradation rate. 
    '''
    def __init__(self,params):
        '''
        N is FSP dimensions, i.e. the number of RNA in the system. 
        '''
        kr,g = params
        self.kr = kr
        self.g = g
        # for now, use some default parameters
        self.N = 101  
        self.ti = 0
        self.tf = 10
        self.ptimes = 15
        self.p0  = np.zeros((self.N,1))
        self.p0[0] = 1
        self.type = 'linear'
        self.N_pars = len(params)
        self.tv = False
        
    def getA(self):
        '''
        Gets the infinitesimal generator for the FSP.
        '''
        d1 = np.arange(self.N) * self.g 
        d2 = np.tile(self.kr,self.N)
        dm = -d1-d2
        return spdiags([d1,d2,dm],[1,-1,0],self.N,self.N) 
    
    def getA_kr(self):
        '''
        Gets the derivative of A with respect to kr. 
        '''
        d1 = np.ones(self.N)   
        return spdiags([d1, -d1],[-1,0],self.N,self.N)

    def getA_g(self):
        '''
        Gets the derivative of A with respect to g. 
        '''
        d1 = np.arange(self.N)
        return spdiags([d1, -d1],[1,0],self.N,self.N)

    def solve(self):
        '''
        Solve the FSP.
        '''
        self.A = self.getA()
        self.soln = expv(self.tf,self.A,self.p0)

    def getQ(self):
        '''
        Build the full matrices to find the sensitivities.
        '''
        A =  self.getA().toarray()
        A_kr =  self.getA_kr().toarray()
        A_g =  self.getA_g().toarray()
        As = [A,A_kr,A_g]
        self.Qs = []
        for i in range(self.N_pars):
            Q = np.zeros( ( 2*self.N , 2*self.N))
            Q[:self.N,:self.N] =  As[0]
            Q[self.N:,:self.N] = As[i+1]
            Q[self.N:,self.N:] = As[0]
            self.Qs.append(Q)
    
    def get_sensitivity(self,tstart=0):
        '''
        Build the full matrices to find the sensitivities.
        '''
        if self.tv: 
            self.ODE = lambda x,t: np.ravel( np.dot( self.getQtv(t),np.array([ x]).T ) )
        else:
            self.getQ()
        self.ss = []
        for i in range(self.N_pars):    
            self.A = csc_matrix(self.Qs[i])
            self._solve(tol=1e-12)
            self.p = self.soln[:self.N,tstart:]
            self.ss.append(self.soln[self.N:,tstart:])
        self.s1,self.s2 = self.ss
        self.S = np.vstack( ( self.s1.ravel() , self.s2.ravel() ) ).T
        # threshold on p to avoid overflow error. 
        small_p = self.p<1e-8
        self.p[small_p] = 1e-2
        z = 1.0/self.p
        z[small_p] = 0.0
        self.P_diag = np.diag(z.ravel())
        self.p[small_p] = 0.0

    def get_FIM(self,tstart=0):
        '''
        get the FIM for the full FSP model.
        analytical version is only for steady state solution. 
        '''
        self.xi = np.zeros(2*self.N)
        self.xi[0] = 1.0
        self.pi = self.xi
        self.tvec=self.gettvec()
        self.get_sensitivity(tstart)
        q = np.diag(self.P_diag)
        Nt = len(self.tvec[tstart:])
        npars = self.S.shape[1] 
        self.FIM = np.zeros((npars,npars))
        for i in range(Nt):
            qt = q[i*self.N:i*self.N+self.N]
            St = self.S[i*self.N:i*self.N+self.N,:]
            for j in range(npars):
                for k in range(npars): 
                    self.FIM[j,k] += self.Nc*np.sum(qt*St[:,j].ravel()*St[:,k].ravel())

class BirthDeathSSA(BirthDeath,GenericSSA):
    '''
    Solve the SSA for models of bursting gene expression.
    '''
    def solve(self,n=1):
        '''
        solve the SSA for this model. 
        '''
        self.get_W()
        self.get_S()
        self._solve(n)
        self.get_dist()

class  BirthDeathMoments(BirthDeath,GenericMoments):
    '''
    get the moments for a bursting gene expression model.
    '''
    def solve(self,order=1,ss=False):
        '''
        solve for the moments.
        '''
        self.get_W()
        self.get_S()
        self.xi = np.zeros(self.N) 
        self.N = 1 

        # Solve moment equations.
        self.get_W_sym()
        self.get_moments(order,ss)
        if order == 1:
            if self.data:
               self.block_vars = self.data['block_vars']
            else:
                print("Did not have data! \n Please supply data for 1st moment FIM.")
        if order == 2:
            self.block_vars = np.zeros((self.N*self.ptimes,self.N*self.ptimes))
            self.block_var_inv = np.zeros((self.N*self.ptimes,self.N*self.ptimes))
            for i in range(self.ptimes):
                self.block_vars[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = self.covariances[:,:,:,i]
                try:
                    self.block_var_inv[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = np.linalg.inv(self.covariances[:,:,:,i])
                except:
                    self.block_var_inv[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = np.linalg.pinv(self.covariances[:,:,:,i])
                    #print('Unable to take inverse at time index %d, used pseudo-inverse instead.' %i)

    def get_sym_vars(self):
        '''
        get a symbolic matrix/vector for the means.
        '''
        mu1 = Symbol('mu1')
        self.mu_sym = Matrix([mu1])
        v11 = Symbol('v11')
        self.var_sym = Matrix([[v11]])
        self.var_upper = Matrix([v11])

    def sensitivity_ODE(self,x,t):
        '''
        Gets the odes that describe the sensivity.
        '''
        # parse the first bits of x into means and variances
        _MU = x[:self.N]
        _SIG = np.reshape(x[self.N:self.N+self.N**2],(self.N,self.N))
        m1 = _MU
        va11 = _SIG[0,0]; 
    
        # Build diagonal matrix
        inner = np.dot(self.W1,np.array([_MU]).T)+self.W0
        self.diag_mat = np.diag(np.ravel(inner),k=0)

        # Compute RHS
        RHS_vars = np.ravel(np.dot(np.dot(self.S,self.W1),_SIG) + np.dot(np.dot(_SIG,self.W1.T),self.S.T) + np.dot(np.dot(self.S,self.diag_mat),self.S.T))
        RHS_means = np.ravel(np.dot(self.S,(np.dot(self.W1,np.atleast_2d(_MU).T)))+np.dot(self.S,self.W0))

        # unfortunately, have to redefine symbols... definitely a way round this.
        mu1 = Symbol('mu1')
        v11 = Symbol('v11')
        tmp =  self.K_subs_i.subs([(mu1,m1),(v11,va11)])
        K_subs = np.ravel(np.array(tmp.tolist()).astype(np.float64))
        return np.concatenate((np.concatenate((RHS_means,RHS_vars)),np.ravel(np.dot(self.J_subs,x[self.N+self.N**2:]) + K_subs )))

    def get_FIM(self,order=1,tstart=0):
        '''
        Obtain the Fisher Information Matrix for the order specified
        by order.
        tstart is the index of tvec to keep for FIM computation. 
        '''
        # define number of cells
        #self.Nc = 1

        self.solve(order=2)
        self.xi = np.concatenate((self.xi,np.zeros(self.N+self.N**2)))
        self.get_S_sym()
        self.get_W_sym()
        self.get_variance_ODE_sym()
        self.get_sym_vars() # weird scope with symbolic junk.

        # redefine all symbols.
        kr = Symbol('kr')
        g = Symbol('g')

        # substitute and convert jacobian to a numpy array.
        J_subs = self.J.subs([(kr,self.kr),(g,self.g)])
        self.J_subs = np.array(J_subs.tolist()).astype(np.float64)

        # substitute parameters in K, but not mu and v.
        K_subs= self.K.subs([(kr,self.kr),(g,self.g)])

        # get the interpolated functions.
#        self.get_interp_functions()

        # solve sensitivities.
        ntimes = len(self.tvec[tstart:])
        nblock = ntimes*self.N
        self.tvec_fim = self.tvec[tstart:]

        self.dm_dtheta = np.zeros((nblock,2))
        self.dv_dtheta = np.zeros((nblock,nblock,2))
        self.FIM = np.zeros((2,2))
        self.ODE = self.sensitivity_ODE

        s_mu = self.N+self.N**2
        s_v = s_mu+self.N
        for i in range(2):
            # get the parameter derivatives.
            self.K_subs_i = K_subs[:,i]
            self._solve()
            self.dm_dtheta[:,i] = np.ravel(self.soln[s_mu:s_v,tstart:].T)
            tmp = np.reshape(self.soln[s_v:,tstart:],(self.N,self.N,ntimes))
            for j in range(ntimes):
                self.dv_dtheta[j*self.N:j*self.N+self.N,j*self.N:j*self.N+self.N,i] = tmp[:,:,j]

        # get mean/covariances at correct times.
        self.mean = np.zeros((self.N,ntimes))
        self.covariances = np.zeros((self.N,self.N,ntimes))
        for t in range(ntimes):
            self.mean[:,t] = self.soln[:self.N,tstart+t]
            self.covariances[:,:,t] =  np.reshape(self.soln[self.N:self.N+self.N**2,tstart+t],(self.N,self.N))

        # convert covariance to blocks.
        self.block_vars = np.zeros((self.N*ntimes,self.N*ntimes))
        self.block_var_inv = np.zeros((self.N*ntimes,self.N*ntimes))
        for i in range(ntimes):
            self.block_vars[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = self.covariances[:,:,i]
            try:
                self.block_var_inv[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = np.linalg.inv(self.covariances[:,:,i])
            except:
                self.block_var_inv[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = np.linalg.pinv(self.covariances[:,:,i])
                #print('Unable to take inverse at time index %d, used pseudo-inverse instead.' %i)

        # compute the FIM
        for i in range(2):
            for j in range(2):
                if order == 1:
                    self.FIM[i,j] = self.Nc*np.dot(np.dot(self.dm_dtheta[:,i].T,self.block_var_inv),self.dm_dtheta[:,j])

                elif order == 2:
                     # make a block diagonal from each covariance at each time.
                     self.FIM[i,j] = self.Nc * (np.dot(np.dot(self.dm_dtheta[:,i].T,self.block_var_inv),self.dm_dtheta[:,j]) + .5 * np.trace(np.dot(np.dot(self.block_var_inv,self.dv_dtheta[:,:,i]),np.dot(self.block_var_inv,self.dv_dtheta[:,:,j]))))

    def get_sample_FIM(self,tstart=0,rna_only=False):
        '''
        get the Fisher information from the sample mean and variance.
        only will work for RNA only = True for now. 
        '''
        # get the Gaussian approximation of the FIM
        # self.get_FIM(order=2,tstart=tstart,rna_only=rna_only)
        # get the centered moments up to order 4.      
        self.solve(order=4)
        # get the Gaussian approximation of the FIM
        self.get_FIM(order=2,tstart=tstart)
        self.higher_moments = np.copy(self.solutions)
        self.FIM = np.zeros(self.FIM.shape)
        for t in range(self.ptimes-1,self.ptimes):
            for i in range(2):
                for j in range(2):
                    self.FIM[i,j] += (self.dm_dtheta[t,i] * self.dm_dtheta[t,j] / self.covariances[0,0,t] + (self.covariances[0,0,t]*self.dv_dtheta[t,t,i] - self.dm_dtheta[t,i]*self.higher_moments[3,t])*(self.covariances[0,0,t]*self.dv_dtheta[t,t,j] - self.dm_dtheta[t,j]*self.higher_moments[3,t]) / ( self.covariances[0,0,t]**2 * ( self.higher_moments[4,t] - self.covariances[0,0,t]**2) - self.covariances[0,0,t] * self.higher_moments[3,t]**2))

        self.FIM = self.Nc*self.FIM


