import sys
sys.path.append('../')
#from importlib import reload
from scipy.sparse import spdiags,csc_matrix
import numpy as np
import generic_solvers
#reload(generic_solvers)
from generic_solvers import GenericODE,GenericFSP,GenericSSA,GenericMoments
from scipy.linalg import expm
from scipy.interpolate import interp1d
from sympy import Matrix,zeros,Symbol

class BurstingGene:
    def __init__(self,params):
        self.params = np.array(params)
        kon,koff,kr,g = params
        self.kr = kr
        self.g = g
        self.kon = kon
        self.koff = koff
        self.xi = np.array([[0,0]]).T
        self.ti = 0
        self.tf = 100
        self.dpars = np.array([1,2]) #free parameters.
        self.ptimes = 10
        self.type = 'linear'
        self.Nc = 100
        self.data = None 
        self.tv = False
        self.observables = 1
        self.N_observables = 1
    
    def get_S(self):
        '''
        Get the stoichiometry matrix for the bursting gene model. 
        Assumes the species order is goff, gon, rna
        '''        
        self.S  =  np.array([ [ 1, 0],
                              [-1, 0],
                              [ 0 , 1],
                              [ 0 ,-1]]).T
                        
    def get_W(self):
        '''
        Get affine linear propensity vectors. 
        Assumes the species order is goff, gon, rna
        '''
        self.W1 = np.array([ [-self.kon, 0],
                             [self.koff,0],
                             [self.kr, 0],
                             [ 0, self.g] ])
        self.W0 = np.zeros((4,1))
        self.W0[0,0] = self.kon

    def get_S_sym(self):
        '''
        get symbolic matrix stoichiometry
        '''    
        self.S_sym = Matrix([[1,-1,0,0],
                            [0,0,1,-1]])

    def get_W_sym(self):
        '''
        Get symbolic W matrices.
        '''
        kon = Symbol('kon')
        koff = Symbol('koff')
        kr = Symbol('kr')   
        g = Symbol('g')
        self.sym_pars = [kon,koff,kr,g]
        self.param_dict = {kon:self.kon,koff:self.koff,kr:self.kr,g:self.g}
        self.theta_vec = Matrix([kon,koff,kr,g])
        self.W1_sym = Matrix([[-kon,0],
                             [koff,0],
                             [kr,0],
                             [0,g]]) 
        self.W0_sym = zeros(4,1)
        self.W0_sym[0,0] = kon
         
class BurstingGeneFSP(GenericFSP):
#class BurstingGeneFSP(GenericODE):
    def __init__(self,params):
        kon,koff,kr,g = params
        self.params = params
        self.kr = kr
        self.g = g
        self.kon = kon
        self.koff = koff
        # for now, use some default parameters
        N = 40
        self.N = N+1  
        self.Nt = 2*self.N
        self.ti = 0
        self.tf = 10
        self.ptimes = 10
        self.p0 = np.zeros((self.Nt,1))
        self.p0[0] = 1
        self.N_pars = len(params)
        self.dpars = [1,2] #free parameters.
        self.xi = np.zeros((self.N_pars+1)*self.Nt)
        self.xi[0] = 1
        self.tv = False
        self.Nc = 100

    def getA(self):
        self.params = [self.kon,self.koff,self.kr,self.g]
        d0 = np.tile(np.array([0,self.koff]),self.N)
        d1 = np.tile(np.array([self.kon,0]),self.N)
        d2 = self.g*np.repeat(np.arange(self.N),2)
        d3 = np.tile([0,self.kr],self.N)
        dm = -d1-d2-d3-d0
        return spdiags([d0,d1,d2,d3,dm],[1,-1,2,-2,0],2*self.N,2*self.N)
    
    def getA_kon(self):
        d1 = np.tile(np.array([1.0,0]),self.N)
        return spdiags([d1, -d1],[-1,0],self.Nt,self.Nt)

    def getA_koff(self):
        d1 = np.tile(np.array([0.0,1.0]),self.N)
        return spdiags([d1, -d1],[1,0],self.Nt,self.Nt)

    def getA_kr(self):
        d1 = np.tile([0,1.0],self.N)
        return spdiags([d1, -d1],[-2,0],self.Nt,self.Nt)
        
    def getA_g(self):
        d1 = np.repeat(np.arange(self.N),2)
        return spdiags([d1, -d1],[2,0],self.Nt,self.Nt)

    def solve(self):
        self.params = [self.kon,self.koff,self.kr,self.g]
        self.Nt = 2*self.N
        self.pi = np.zeros((self.Nt))
        self.pi[0] = 1
        self.A = csc_matrix(self.getA())
        self._solve()
        self.marginal = np.zeros((self.N,self.ptimes))
        for i in range(self.N):
            self.marginal[i,:] =  np.sum( self.soln[2*i:2*i+2,:], axis=0)
        self.p = self.marginal 

    def getQ(self):
        A =  self.getA().toarray()
        A_kr =  self.getA_kr().toarray()
        A_g =  self.getA_g().toarray()
        A_kon = self.getA_kon().toarray()
        A_koff = self.getA_koff().toarray()
        As = [A, A_kon, A_koff, A_kr, A_g] 
        self.Qs = []
        for i in range(self.N_pars):
            Q = np.zeros( ( 2*self.Nt ,2*self.Nt))
            Q[:self.Nt,:self.Nt] = As[0]
            Q[self.Nt:,:self.Nt] = As[i+1] 
            Q[self.Nt:,self.Nt:] = As[0]
            self.Qs.append(Q)
#        for i in range(len(As)):
#            Q[i*self.Nt:i*self.Nt+self.Nt,:self.Nt] = As[i]
#            # Main diag is A
#            Q[i*self.Nt:i*self.Nt+self.Nt,i*self.Nt:i*self.Nt+self.Nt] = A
#        self.Q = Q  
#        return Q
        
    def get_sensitivity(self,tstart=0,rna_only = False,log=False):
        '''
        Solve the sensitivity matrix for the system. 
        '''
        self.getQ()
        self.ss = []
        for i in range(self.N_pars):
#            self.ODE = lambda x,t: np.ravel( np.dot(self.Qs[i],np.array([ x]).T ) )
            self.A = csc_matrix(self.Qs[i])
            self._solve() 
            self.p = self.soln[:self.Nt,tstart:]
            if log:
                self.ss.append(self.soln[self.Nt:,tstart:]*self.params[i])
            else:
                self.ss.append(self.soln[self.Nt:,tstart:])
        self.s1,self.s2,self.s3,self.s4 = self.ss
#        self.s1 = self.soln[self.Nt:2*self.Nt,tstart:]
#        self.s2 = self.soln[2*self.Nt:2*self.Nt+self.Nt,tstart:]
#        self.s3 = self.soln[3*self.Nt:3*self.Nt+self.Nt,tstart:]
#        self.s4 = self.soln[4*self.Nt:4*self.Nt+self.Nt,tstart:]
        
        ntimes = len(self.tvec[tstart:])
        self.marginal = np.zeros((self.N,ntimes))

        # get the marginal distributions for what we can measure. 
        for i in range(self.N):
            self.marginal[i,:] = np.sum(self.p[2*i:2*i+2,:],axis=0)

        if rna_only:
            s1 = np.zeros((self.N,len(self.tvec[tstart:])))
            s2 = np.zeros((self.N,len(self.tvec[tstart:])))
            s3 = np.zeros((self.N,len(self.tvec[tstart:])))
            s4 = np.zeros((self.N,len(self.tvec[tstart:])))
            for i in range(self.N):
                s1[i,:] = np.sum(self.s1[2*i:2*i+2,:],axis=0)
                s2[i,:] = np.sum(self.s2[2*i:2*i+2,:],axis=0)
                s3[i,:] = np.sum(self.s3[2*i:2*i+2,:],axis=0)
                s4[i,:] = np.sum(self.s4[2*i:2*i+2,:],axis=0)
            self.s1 = s1
            self.s2 = s2
            self.s3 = s3
            self.s4 = s4
            self.p = self.marginal
                    
        # Make the full S matrix 
        self.S = np.vstack( ( self.s1.ravel() , self.s2.ravel(),self.s3.ravel(),self.s4.ravel()) ).T

        # threshold on p to avoid overflow error. 
        small_p = self.p<1e-6
        self.p[small_p] = 1e-2
        z = 1.0/self.p
        z[small_p] = 0.0
        self.P_diag = np.diag(z.ravel())
        self.p[small_p] = 0.0
            
    def get_moments(self):
        '''
        Get moments from distributions.
        '''
        #self.get_sensitivity() 
        num_p,num_t = self.marginal.shape
        self.mean = np.zeros(num_t)
        self.dd2 = np.zeros(num_t)
        for t in range(num_t):
            self.dd2[t] = np.sum( ( np.arange( 0, self.N )**2 ) * self.marginal[:,t] )
            self.mean[t] = np.sum( ( np.arange( 0, self.N ) * self.marginal[:,t] ))
        self.covariances = self.dd2-self.mean**2

    def get_DTD(self,tstart=0,rna_only=False):
        '''
        Find E{D^T D} using theory.
        '''
        print('Using {0} cells'.format(self.Nc))
        ntimes = len(self.tvec[tstart:]) 
        if rna_only:
            msize = self.N*ntimes
        else:
            msize = self.Nt*ntimes
        #p_ravel = self.p.T.ravel() 
        p_ravel = self.p.ravel() 
        # Assign the diagonal.
        pr2 = p_ravel**2
        DTD_diag = (self.Nc**2)*pr2+self.Nc*p_ravel-self.Nc*pr2
        self.DTD= (self.Nc**2-self.Nc)*np.dot(np.array([p_ravel]).T,np.array([p_ravel]))
        np.fill_diagonal(self.DTD,DTD_diag)
        return self.DTD

    def get_FIM(self,tstart=0,rna_only=False,log=False):  
        '''
        get the FIM for the full FSP model.
        '''
        self.tvec = self.gettvec()
        self.Nt = self.N*2
        # hardcoded for now.
        self.xi = np.zeros(2*self.Nt)
        self.xi[0] = 1.0
        self.pi = self.xi
        self.get_sensitivity(tstart=tstart,rna_only=rna_only,log=log)
        # Use only dpars for sensitivity. 
        self.S = self.S[:,self.dpars]

        # modify to use new analysis.
#        DTD = self.get_DTD(tstart,rna_only)
#        tmp1 = np.dot(self.S.T,self.P_diag)
#        tmp2 = np.dot(tmp1,DTD)
#        tmp3 = np.dot(tmp2,self.P_diag)
#        self.FIM = np.dot(tmp3,self.S)
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

class BurstingGeneSSA(BurstingGene,GenericSSA):
    '''
    Solve the SSA for models of bursting gene expression. 
    '''
    def solve(self,n=1):
        '''
        solve SSA stuffs.
        '''
        self.get_W()
        self.get_S()
        self._solve(n)
        self.get_dist() 

    def get_dist(self):
        '''
        build distribution (non-normalized and pdf)
        of rna for the model)
        '''
        n_specs, n_times, n_traj = self.data.shape
        max_rna = np.max(self.data[1,:,:])
        self.pdf = np.zeros((n_times,int(max_rna)+1))
        self.fdist = np.zeros((n_times,int(max_rna)+1))
        for i in range(n_times):
            for j in range(n_traj):
                self.fdist[i,int(self.data[1,i,j])] +=1
            self.pdf[i,:] = self.fdist[i,:] / np.sum(self.fdist[i,:])

    def get_means(self):
        '''
        get the first moment.
        '''       
        n_specs, n_times, n_traj = self.data.shape
        max_rna = np.max(self.data[2,:,:])+1
        self.means = np.zeros(n_times)
        for i in range(n_times):
            self.means[i] = np.sum(np.arange(max_rna)*self.pdf[i,:])

    def get_variances(self):
        '''
        get the second moment.
        '''
        self.get_dist()
        n_specs, n_times, n_traj = self.data.shape
        max_rna = np.max(self.data[1,:,:])+1
        self.variances = np.zeros(n_times)
        self.covariances = np.zeros((n_specs,n_specs,n_times))
        for i in range(n_times):
            self.variances[i] = np.sum((np.arange(max_rna)**2)*self.pdf[i,:])-(np.sum(np.arange(max_rna)*self.pdf[i,:])**2)
            self.covariances[:,:,i] = np.cov(self.data[:,i,:])

class BurstingGeneMoments(BurstingGene,GenericMoments):
    '''
    get the first couple of moments for a bursting gene expression model. 
    '''
    def solve(self,order=2,ss=False,rna_only=True):
        '''
        solve for the moments. 
        '''
        self.get_W()
        self.get_S()
#        self.xi  = np.ravel(self.xi.T)
#        self.N = len(self.xi)
        self.params = np.array([self.kon,self.koff,self.kr,self.g])
        # Solve moment equations. 
        self.get_moments(order=order,ss=ss)
        #self.var = self.covariances.ravel()
#        if rna_only:
#            self.mean = self.mean[1,:]
#            self.var = self.covariances[1,1,:]
#        if order == 1: 
#            if self.data:
#               self.block_vars = self.data['block_vars'] 
#            else:
#                print("Did not have data! \n Please supply data for 1st moment FIM.")    
#        if order == 2: 
##            self.xi = np.zeros(2*self.N+2*self.N**2)
#            self.block_vars = np.zeros((self.N*self.ptimes,self.N*self.ptimes))
#            self.block_var_inv = np.zeros((self.N*self.ptimes,self.N*self.ptimes))
#            for i in range(self.ptimes):
#                self.block_vars[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = self.covariances[:,:,i]
#                try:
#                    self.block_var_inv[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = np.linalg.inv(self.covariances[:,:,i])
#                except:
#                    self.block_var_inv[i*self.N:i*self.N+self.N,i*self.N:i*self.N+self.N] = np.linalg.pinv(self.covariances[:,:,i])
#                    print('Unable to take inverse at time index %d, used pseudo-inverse instead.' %i)

    def get_sym_vars(self):
        '''
        get a symbolic matrix/vector for the means. 
        '''
        mu1 = Symbol('mu1')
        mu2 = Symbol('mu2')
        self.mu_sym = Matrix([mu1,mu2])
        v11 = Symbol('v11')        
        v22 = Symbol('v22')        
        v12 = Symbol('v12')        
        v21 = Symbol('v21')
        self.species_dict = {mu1:1,mu2:0,v11:0,v12:0,v21:0,v22:0}
        self.species_list = [mu1,mu2,v11,v12,v21,v22]
        self.var_sym = Matrix([[v11,v12],[v12,v22]])
        self.var_upper = Matrix([v11,v22,v12])
    
#    def sensitivity_ODE(self,x,t):
#        '''
#        Gets the odes that describe the sensivity. 
#        '''
#        # evaluate the K vector at the correct time. 
##        m1,m2,va11,va22,va12 = self.get_m_v(t) 
#
#        # parse the first bits of x into means and variances
#        _MU = x[:self.N]
#        _SIG = np.reshape(x[self.N:self.N+self.N**2],(self.N,self.N))
#        m1,m2 = _MU
#        va11 = _SIG[0,0]; va22 = _SIG[1,1]; va12 = _SIG[0,1]
#
#        # Build diagonal matrix
#        inner = np.dot(self.W1,np.array([_MU]).T)+self.W0
#        self.diag_mat = np.diag(np.ravel(inner),k=0)
#
#        # Compute RHS
#        RHS_vars = np.ravel(np.dot(np.dot(self.S,self.W1),_SIG) + np.dot(np.dot(_SIG,self.W1.T),self.S.T) + np.dot(np.dot(self.S,self.diag_mat),self.S.T))
#        RHS_means = np.ravel(np.dot(self.S,(np.dot(self.W1,np.atleast_2d(_MU).T)))+np.dot(self.S,self.W0))
#
#        # unfortunately, have to redefine symbols... definitely a way round this. 
#        mu1 = Symbol('mu1')
#        mu2 = Symbol('mu2')
#        v11 = Symbol('v11')        
#        v22 = Symbol('v22')        
#        v12 = Symbol('v12')        
#        tmp =  self.K_subs_i.subs([(mu1,m1),(mu2,m2),(v11,va11),(v22,va22),
#        (v12,va12)])
#        K_subs = np.ravel(np.array(tmp.tolist()).astype(np.float64))
#        return np.concatenate((np.concatenate((RHS_means,RHS_vars)),np.ravel(np.dot(self.J_subs,x[self.N+self.N**2:]) + K_subs )))
    
    def get_m_v(self,t):
        '''
        evaluate K as a function of time. 
        '''
        vals = [self.interp_m1(t),
        self.interp_m2(t),
        self.interp_v11(t),
        self.interp_v22(t),
        self.interp_v12(t)]
        return vals
    
    def get_interp_functions(self):
        '''
        takes the solutions from the moment functions and generates the necessary functions. 
        '''
        self.interp_m1 = interp1d(self.tvec,self.mean[0,:])
        self.interp_m2 = interp1d(self.tvec,self.mean[1,:])
           
        self.interp_v11 = interp1d(self.tvec,self.covariances[0,0,:])
        self.interp_v22 = interp1d(self.tvec,self.covariances[1,1,:])
        self.interp_v12 = interp1d(self.tvec,self.covariances[0,1,:])

#    def get_FIM(self,order=1,tstart=0,rna_only=False):
#        '''
#        Obtain the Fisher Information Matrix for the order specified 
#        by order. 
#        '''
#        if rna_only: 
#            obvs = [1]
#            nobvs = 1
#        else:
#            obvs = [0,1]
#            nobvs = 2
#        self.xi = np.zeros(self.N+self.N**2)
#        self.solve(order=order)
#        print('solved moment equations')
#        #self.xi = np.concatenate((self.xi,np.zeros(self.N+self.N**2)))
#        self.xi = np.zeros(2*(self.N+self.N**2))
#        print(self.xi.shape)
#        self.get_S_sym()
#        self.get_W_sym()
#        self.get_variance_ODE_sym()
#        self.get_sym_vars() # weird scope with symbolic junk.
#
#        # redefine all symbols. 
#        kon = Symbol('kon')
#        koff = Symbol('koff')
#        kr = Symbol('kr')   
#        g = Symbol('g')
#
#        # substitute and convert jacobian to a numpy array. 
#        J_subs = self.J.subs([(kon,self.kon),(koff,self.koff),(kr,self.kr),(g,self.g)])
#        self.J_subs = np.array(J_subs.tolist()).astype(np.float64)
#
#        # substitute parameters in K, but not mu and v. 
#        K_subs= self.K.subs([(kon,self.kon),(koff,self.koff),(kr,self.kr),(g,self.g)])
#
#        # get the interpolated functions. 
##        self.get_interp_functions()
#
#        # solve sensitivities.  
#        ntimes = len(self.tvec[tstart:])
#        nblock = ntimes*nobvs
#        self.tvec_fim = self.tvec[tstart:]
#
#        self.n_free_pars = len(self.dpars)
#        self.dm_dtheta = np.zeros((nblock,self.n_free_pars))
#        self.dv_dtheta = np.zeros((nblock,nblock,self.n_free_pars))
#        self.FIM = np.zeros((self.n_free_pars,self.n_free_pars))
#        self.ODE = self.sensitivity_ODE
#
#        s_mu = self.N+self.N**2
#        s_v = s_mu+self.N
#        for i in range(self.n_free_pars):
#            # get the parameter derivatives.
#            self.K_subs_i = K_subs[:,self.dpars[i]]
#            self._solve()
#            tmp = self.soln[s_mu:s_v,tstart:]
#            self.dm_dtheta[:,i] = np.ravel(tmp[obvs,:].T)
#            tmp = np.reshape(self.soln[s_v:,tstart:],(self.N,self.N,ntimes))
#            tmp2 = tmp[obvs,:,:]
#            tmp3 = tmp2[:,obvs,:]
#            for j in range(ntimes):
#                self.dv_dtheta[j*nobvs:j*nobvs+nobvs,j*nobvs:j*nobvs+nobvs,i] = tmp3[:,:,j]
#
#        # get mean/covariances at correct times.
#        self.mean = np.zeros((self.N,ntimes)) 
#        self.covariances = np.zeros((nobvs,nobvs,ntimes))
#        for t in range(ntimes):
#            a = self.soln[:self.N,t+tstart]
#            self.mean[:,t]  = a[obvs]
#            tmp  =  np.reshape(self.soln[self.N:self.N+self.N**2,t+tstart],(self.N,self.N))
#            tmp2 = tmp[obvs,:]
#            tmp3 = tmp2[:,obvs] 
#            self.covariances[:,:,t] = tmp3
#
#        # convert covariance to blocks. 
#        self.block_vars = np.zeros((nobvs*ntimes,nobvs*ntimes))
#        self.block_var_inv = np.zeros((nobvs*ntimes,nobvs*ntimes))      
#        for i in range(ntimes):
#            self.block_vars[i*nobvs:i*nobvs+nobvs,i*nobvs:i*nobvs+nobvs] = self.covariances[:,:,i]
#            try:
#                self.block_var_inv[i*nobvs:i*nobvs+nobvs,i*nobvs:i*nobvs+nobvs] = np.linalg.inv(self.covariances[:,:,i])
#            except:
#                self.block_var_inv[i*nobvs:i*nobvs+nobvs,i*nobvs:i*nobvs+nobvs] = np.linalg.pinv(self.covariances[:,:,i])
#                print('Unable to take inverse at time index %d, used pseudo-inverse instead.' %i)
#
#        # compute the FIM  
#        for i in range(self.n_free_pars):
#            for j in range(self.n_free_pars):
#                if order == 1:
#                    self.FIM[i,j] = np.dot(np.dot(self.dm_dtheta[self.N:,i].T,self.block_vars),self.dm_dtheta[self.N:,j]) 
#                elif order == 2: 
#                     # make a block diagonal from each covariance at each time. 
#                    self.FIM[i,j] = self.Nc * (np.dot(np.dot(self.dm_dtheta[:,i].T,self.block_var_inv),self.dm_dtheta[:,j]) + .5 * np.trace(np.dot(np.dot(self.block_var_inv,self.dv_dtheta[:,:,i]),np.dot(self.block_var_inv,self.dv_dtheta[:,:,j]))))

    def get_sample_FIM(self,tstart=0,rna_only=False,log=False):
        '''
        get the Fisher information from the sample mean and variance.
        only will work for RNA only = True for now. 
        '''
        # get the Gaussian approximation of the FIM
        # self.get_FIM(order=2,tstart=tstart,rna_only=rna_only)
        # get the centered moments up to order 4.      
        self.n_free_pars = 2
        self.get_W_sym()
        self.get_S()
        m = self.get_M(4)
        self.get_arb_RHS(m)
        self.get_FIM(order=2,tstart=tstart)
        self.solve(order=4)
        # get the Gaussian approximation of the FIM
        self.higher_moments = np.copy(self.solutions)
        self.FIM = np.zeros(self.FIM.shape)
        self.covariances = self.covariances.ravel()
        for t in range(self.ptimes-1):
            tind = t+1
            for i in range(self.n_free_pars):
                for j in range(self.n_free_pars):
                    #self.FIM[i,j] += (self.dm_dtheta[tind,i] * self.dm_dtheta[tind,j] / self.covariances[1,1,t] + (self.covariances[1,1,t]*self.dv_dtheta[tind,tind,i] - self.dm_dtheta[tind,i]*self.higher_moments[8,t])*(self.covariances[1,1,t]*self.dv_dtheta[tind,tind,j] - self.dm_dtheta[tind,j]*self.higher_moments[8,t]) / ( self.covariances[1,1,t]**2 * ( self.higher_moments[10,t] - self.covariances[1,1,t]**2) - self.covariances[1,1,t] * self.higher_moments[8,t]**2))
                    self.FIM[i,j] += self.Nc[0]*(self.dm_dtheta[t,i] * self.dm_dtheta[t,j] / self.covariances[t] + (self.covariances[t]*self.dv_dtheta[t,t,i] - self.dm_dtheta[t,i]*self.higher_moments[8,tind])*(self.covariances[t]*self.dv_dtheta[t,t,j] - self.dm_dtheta[t,j]*self.higher_moments[8,tind]) / ( self.covariances[t]**2 * ( self.higher_moments[10,tind] - self.covariances[t]**2) - self.covariances[t] * self.higher_moments[8,tind]**2))
        if log:
            for i in range(self.n_free_pars):
                for j in range(self.n_free_pars):
                    self.FIM[i,j] = self.FIM[i,j]*self.params[self.dpars][i]*self.params[self.dpars][j]

