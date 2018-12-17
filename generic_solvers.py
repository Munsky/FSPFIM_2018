#-*- coding: utf-8 -*-
import numpy as np
from scipy.sparse.linalg import expm
from expv import expv
from expokitpy import dgexpv
from scipy.integrate import ode,odeint
from scipy.misc import comb 
import scipy as sp
import scipy.optimize
import random as random
from sympy import Matrix,Symbol,zeros,diag,MatrixSymbol,lambdify,expand,diff
from scipy.special import gamma
import time
from scipy.sparse.linalg import onenormest

class GenericSSA:
    def __init__(self,type='linear'):
        '''
        Simulate biochemical reactions with the stochastic simulation algorithm.
        Parameters: 
        xi: initial state of the system
        ti: initial time of the system
        S: Stoichiometry matrix
        atype: linear or nonlinear. use nonlinear for time-varying.
        ptimes: number of print times in time-vector. 
        W0: propensity matrix that is independent of state
        W1: propensity matrix that is linear with state. W = W1*x+W0
        P: for non-linear/time-varying propensities. P must be a function of the state x and time t.
        '''
        self.xi=np.array([])
        self.ti=None
        self.tf=None
        self.S=np.array([])
        self.type=type
        self.ptimes=100
        self.params={}
        if type=='linear':
            self.W0=np.array([])   
            self.W1=np.array([])
        if atype == 'nonlinear':
            self.fast_rxn = 0.5
            self.P=lambda x,t:None 

    def gettvec(self):
        '''
        Generate a vector of times to store the state between ti and tf.
        '''
        return np.linspace(self.ti,self.tf,self.ptimes)
        
    def _run_trajectory(self):
        '''
        Simulate SSA using the direct method.
        '''
        x=self.xi
        t=self.ti
        __n=len(x)
        self.time=self.gettvec()
        data=np.zeros((len(self.xi),self.ptimes))
        ip=0
        if self.atype=='linear':
            while t<self.tf:
                rate=np.atleast_2d(np.dot(self.W1,x))+self.W0
                rate=np.cumsum(rate)
                t=(t-np.log(np.random.rand(1))/rate[-1])
                ro=rate[-1]*np.random.rand()
                while t>self.time[ip]:
                    if t>self.tf:
                        b = len(self.time[ip:])
                        fill = np.repeat(x,b)
                        data[:,ip:]=fill.reshape(__n,b)
                        return data
                    else:
                        data[:,ip]=x.reshape(__n)
                        ip=ip+1
                for i in range(len(rate)):
                    if rate[i]>=ro:
                        event=i
                        break
                x=x+np.atleast_2d(self.S[:,event]).T

        elif self.atype=='nonlinear':
            x = np.concatenate((np.array([0]),self.xi))
            __n=len(x)
            self.time=self.gettvec()
            data=np.zeros((len(self.xi),self.ptimes))
            a,b = self.S.shape
            S = np.vstack((np.zeros(b),self.S))
            S = np.hstack((np.zeros((a+1,1)),S))
            while t<self.tf:
                trate=self.get_P(x[1:],t)
                rate = np.concatenate((np.array([self.fast_rxn]),trate))
                rate=np.cumsum(rate)
                t=(t-np.log(np.random.rand(1))/rate[-1])
                ro=rate[-1]*np.random.rand()
                while t>self.time[ip]:
                    if t>self.tf:
                        b = len(self.time[ip:])
                        fill = np.repeat(x[1:],b)
                        data[:,ip:]=fill.reshape(__n-1,b)
                        return data
                    else:
                        #data[:,ip]=x.reshape(__n)
                        data[:,ip]=x[1:]
                        ip=ip+1
                for i in range(len(rate)):
                    if rate[i]>=ro:
                        event=i
                        break
                x=x+S[:,event].ravel()
        else:
            'Error'
        self.data=data
        return data

    def _solve(self,n):
        '''
        Solve the SSA for n trajectories and 
        store the results
        '''
        __data=np.zeros((len(self.xi),self.ptimes,n))
        for i in range(n):
            __d=self._run_trajectory()
            __data[:,:,i]=__d
        self.data = __data
        return __data
        
    def get_dist(self,specID=0):
        '''
        build distribution (non-normalized and pdf)
        of rna for the model)
        '''
        n_specs, n_times, n_traj = self.data.shape
        max_rna = int(np.max(self.data[specID,:,:]))
        self.pdf = np.zeros((n_times,max_rna+1))
        self.fdist = np.zeros((n_times,max_rna+1))
        for i in range(n_times):
            for j in range(n_traj):
                self.fdist[i,int(self.data[specID,i,j])] +=1
            self.pdf[i,:] = self.fdist[i,:] / np.sum(self.fdist[i,:])

    def get_means(self,specID=0):
        '''
        get the first moment.
        '''
        n_specs, n_times, n_traj = self.data.shape
        max_rna = np.max(self.data[specID,:,:])+1
        self.means = np.zeros(n_times)
        for i in range(n_times):
            self.means[i] = np.sum(np.arange(max_rna)*self.pdf[i,:])

    def get_variances(self,specID=0):
        '''
        get the second moment.
        '''
        self.get_dist()
        n_specs, n_times, n_traj = self.data.shape
        max_rna = np.max(self.data[specID,:,:])+1
        self.variances = np.zeros(n_times)
        self.covariances = np.zeros((n_specs,n_specs,n_times))
        for i in range(n_times):
            self.variances[i] = np.sum((np.arange(max_rna)**2)*self.pdf[i,:])-(np.sum(np.arange(max_rna)*self.pdf[i,:])**2)
            self.covariances[:,:,i] = np.cov(self.data[:,i,:]) 

class GenericFSP:
    def __init__(self,ti=[],tf = [],xi=[],A=np.array([]),ptimes=100):
        '''
        Simulate biochemical reactions with the stochastic simulation algorithm.
        Parameters: 
        xi: initial state of the system
        ti: initial time of the system
        ptimes: number of print times in time-vector. 
        A: infinitesimal generator for the system. 
        '''
        self.ti=ti
        self.tf=tf
        self.xi=xi
        self.A=A
        self.errors=np.empty(ptimes)
        self.ptimes=ptimes
        self.params={}
        
    def gettvec(self):
        '''
        get vector of times 
        '''
        return np.linspace(0,self.tf-self.ti,self.ptimes)

    def _solve(self,tol=1e-9):
        '''
        solve the FSP at each time. 
        tol specifies the acceptable error tolerance in solutions. 
        '''
        try:
            self.tvec.any()
            self.ptimes = len(self.tvec)
        except:
            self.tvec = self.gettvec()

        self.soln = np.zeros((len(self.pi),len(self.tvec)))
        N = len(self.pi)
        self.soln[:,0] = self.pi
        pv = self.pi
        n=int(N)
        m=30
        w = np.ones(n,dtype=np.float64)
        anorm = onenormest(self.A)
        wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.float64)
        iwsp = np.zeros(m+2,dtype=np.int32)
        for i in range(1,len(self.tvec)):
            try:
                # Solve the system using the  wrapped fortran expv function
                pv,tol0,iflag0 = dgexpv(30,self.tvec[i]-self.tvec[i-1],pv,1e-7,anorm,wsp,iwsp,self.A.dot,0)
            except:
                # use the python expv function
                pv,m,v = expv(self.tvec[i]-self.tvec[i-1],self.A,pv,tol = tol,m=30)
            self.soln[:,i] = pv 
        return self.soln

class GenericODE:
    def __init__(self,ti=None,tf=None,ODE=None,ptimes=50,xi=None):
        '''
        Simulate biochemical reactions with the stochastic simulation algorithm.
        Parameters: 
        xi: initial state of the system
        ti: initial time of the system
        ODE: function of x,t to be integrated. 
        '''
        self.ti=ti
        self.tf=tf
        self.ODE=ODE
        self.ptimes=ptimes
        self.xi=xi
        self.tvec=None
        self.soln=None
        self.parameters={}
    
    def gettvec(self):
        '''
        get vector of times
        '''
        return np.linspace(self.ti,self.tf,self.ptimes)

    def nest_expansion(self,tvec,n_expansions):
        '''
        get an expanded tvec, for when smaller time steps are needed 
        with some integration methods. 
        '''
        if n_expansions == 0:
            return tvec
        else:
            tvec_old = tvec
            for i in range(n_expansions):
                tvec_new = np.zeros(2*len(tvec_old)-1)
                tvec_new[::2] = tvec_old
                tvec_new[1::2] = .5*(tvec_old[1:]+tvec_old[:-1])
                tvec_old = tvec_new
            return tvec_new

    def _solve(self):
        '''
        solve the ODEs at each time
        '''
        self.tvec=self.gettvec()
        nexp_max = 5
        nexp = 0 
        while nexp<nexp_max+1:
            tvec_for_integration = self.nest_expansion(self.tvec,nexp)
            try: 
                try: 
                    solution=odeint(self.ODE,self.xi,tvec_for_integration,mxstep = 100000,Dfun=self.jacobian)       
                except:
                    solution=odeint(self.ODE,self.xi,tvec_for_integration,mxstep = 100000)       
                self.soln = solution[::int(2**(nexp)),:].T
                return self.soln
            except: 
                print( "integration failed! increasing time points for integrator.")
                nexp += 1
                 
        print('Unable to integrate by expanding tvec. Returning -1')
        return -1
    
class GenericMoments(GenericODE):
    '''
    This is a class to get moments for
    models with affine-linear propensities.
    Simulate biochemical reactions with the stochastic simulation algorithm.
    Parameters: 
    xi: initial state of the system
    ti: initial time of the system
    S: Stoichiometry matrix
    atype: linear or nonlinear. use nonlinear for time-varying.
    ptimes: number of print times in time-vector. 
    W0: propensity matrix that is independent of state
    W1: propensity matrix that is linear with state. W = W1*x+W0
    N: number of species. 
    '''
    def mean_ODE(self,x,t):
        '''
        set of ODEs for the means
        '''
        x = np.array([x]).T
        return np.ravel(np.dot(np.dot(self.S,self.W1),x) + np.dot(self.S,self.W0) )
     
    def get_mean_SS(self):
        '''
        get the steady state mean values for the system
        '''
        M1 = -1*np.linalg.inv(np.dot(self.S,self.W1))
        return np.dot(M1,np.dot(self.S,self.W0))
        
    def get_var_SS(self):
        '''
        get the steady state variance for the system
        by solving the lyapunov fn.
        '''
        A = np.dot(self.S,self.W1)
        inner = np.dot(self.W1,self.mu_ss)+self.W0
        diag_mat = np.diag(np.ravel(inner),k=0)
        Q = np.dot(np.dot(self.S,diag_mat),self.S.T)
        return solve_lyapunov(A,-Q)
  
    def variance_ODE(self,x,t):
        '''
        build variance/covariance ODEs.
        '''
        # dSIG = S*W1*SIG + SIG*W1'*S' + S*diag((W1*MU + W0)')*S'
        _MU = x[:self.N]
        _SIG = np.reshape(x[self.N:],(self.N,self.N))
        # Build diagonal matrix
        inner = np.dot(self.W1,np.array([_MU]).T)+self.W0
        self.diag_mat = np.diag(np.ravel(inner),k=0)
        # Compute RHS
        RHS_vars = np.ravel(np.dot(np.dot(self.S,self.W1),_SIG) + np.dot(np.dot(_SIG,self.W1.T),self.S.T) + np.dot(np.dot(self.S,self.diag_mat),self.S.T))
        RHS_means = np.ravel(np.dot(self.S,(np.dot(self.W1,np.atleast_2d(_MU).T)))+np.dot(self.S,self.W0))
        return np.concatenate((RHS_means,RHS_vars)) 

    def get_variance_ODE_sym(self):
        '''
        build a symbolic version of the variance ODE. 
        does not solve anything.
        Requires model to have symbolic S, W0, W1 stored as
        self.S_sym, self.W0_sym, self.W1_sym. 
        '''
        # Supply empty mu and sigma. 
        self.get_sym_vars()
        # Build diagonal matrix
        inner = self.W1_sym*self.mu_sym+self.W0_sym
        diag_mat = diag(*inner)
        # Compute RHS
        A1 = self.S_sym*self.W1_sym
        A2 = A1*self.var_sym
        B1= self.var_sym*self.W1_sym.T
        B2 = B1*self.S_sym.T
        C1 = self.S_sym*diag_mat
        C2 = C1*self.S_sym.T
        RHS_vars = A2+B2+C2
        #RHS_vars = np.dot(np.dot(self.S,self.W1),self.var_sym) + np.dot(np.dot(_self.var_sym,self.W1.T),self.S.T) + np.dot(np.dot(self.S,self.diag_mat),self.S.T)
        W1 = self.S_sym*self.W1_sym*self.mu_sym + self.S_sym*self.W0_sym
           
       # RHS_means = np.ravel(np.dot(self.S,(np.dot(self.W1,np.atleast_2d(__MU).T)))+np.dot(self.S,self.W0))
        W2 = self.ravel_sym(RHS_vars)
        self.W = W1.col_join(W2) 
        self.var_vec = self.ravel_sym(self.var_sym)
        self.Y = self.mu_sym.col_join(self.var_vec)
        self.J = self.W.jacobian(self.Y) 
        self.K = self.W.jacobian(self.theta_vec)
        return 1 

    def ravel_sym(self,M):
        '''
        turns the matrix M with dim NxN into a (N**2)x1 
        matrix, similar to numpy's ravel function.  
        '''
        a,b = M.shape               
        blnk = []
        # put each row into a list
        for i in range(a):
            blnk.append(M[i,:])
        M2 = zeros(a**2,1)
        for i in range(len(blnk)):
            M2[a*i:a*i+a,:] = blnk[i].T
        return M2
        
    def get_mean(self):
        '''
        get the 1st moment of the stochastic system by 
        integrating the mean ODE. 
        '''
        self.ODE = self.mean_ODE    
        self.mean = self._solve()
    
    def get_var(self):
        '''
        Get the means and covariance matrix. 
        uses an "observables" variable, which means that it will
        only return variables of certain species, specified by 
        observables.
        '''
        self.xi = np.zeros(self.N+self.N**2)
        self.ODE = self.variance_ODE
        solns = self._solve()
        foo,ntimes = solns.shape
        self.mean = np.zeros((self.N,ntimes)) 
        self.covariances = np.zeros((self.N,self.N,ntimes))
        for t in range(ntimes):
            self.mean[:,t] = solns[:self.N,t]
            self.covariances[:,:,t] =  np.reshape(solns[self.N:,t],(self.N,self.N))
        # only keep the species specified by the observables array.
        self.mean = self.mean[self.observables,:]
        self.covariances = np.array([self.covariances[self.observables,:,:]])[:,self.observables,:]
    
    def get_moments(self,order=1,ss=False):
        '''
        This is a convenience function to get whatever moments you want.
        ''' 
        if order == 1:
            if ss:
                self.mean = self.get_mean_SS()
                return
            if not ss:
                self.get_mean()
                return
        elif order == 2: 
            if ss:
                self.mean = self.get_mean_SS()
                self.var = self.get_var_SS()
                return
            if not ss:
        #         self.xi = np.concatenate((self.xi,np.zeros(self.N**2)))
                self.get_var()
                return

        elif order > 2:
            m = self.get_M(order)
            self.xi = np.zeros(m.shape[0])
            self.get_arb_RHS(m)
            self.A_real = np.array(self.A(*self.params)).astype(np.float64)
            self.B_real = np.array(self.B(*self.params)).astype(np.float64)
            self.jacobian = lambda y,t: self.A_real
            self.ODE = self.arb_RHS_wrap
            solns = self._solve()
            self.uncentered = solns
            self.create_centered_moments_func(m)
            self.solutions = np.zeros((m.shape[0],self.ptimes))
            for t in range(self.ptimes):
                self.solutions[:,t] = self.convert_to_centered(*solns[:,t]).ravel()
    
    def get_arbitrary_moments(self,order):
        '''
        get the moments up to some arbitrary 
        moments. Ordering should be written 
        down somewhere. 
        '''
        M = self.get_M(order)
        RHS = self.make_arb_RHS(M)
        self.create_centered_moments_func()

    def get_M(self,order):
        '''
        obtain the B matrix, which contains the 
        orders for each moment up to ORDER.  
        '''
        # obtain all possible combinations (not unique)
        b_pos = np.zeros(((order+1)**self.N,self.N)) 
        b = np.zeros(self.N)
        for i in range((order+1)**self.N-1):
            b[0]+=1
            j = 0
            while b[j] == order+1:
                b[j] = 0
                b[j+1] += 1 
                j+=1
            b_pos[i+1:] = b

        # find those combos which are unique, rank 
        # the rows. 
        m = np.sum(b_pos,axis=1)
        indsort = np.argsort(m)
        sortm = np.sort(m)
        max_ind = np.where(sortm == order)[0][-1]
        M = b_pos[indsort,:]
        M = M[:max_ind+1,:]
        return M
    
    def get_arb_RHS(self,M):
        '''
        find moment equations of arbitrary order 
        for the RHS. 
        '''
        nx,ns = M.shape
        # make some symbolic variables. 
        x = []
        for m in range(ns):
            x.append(Symbol('x'+str(m)))
        x = np.array([x]).T
        nun,n_rxn = self.S.shape
        RHS = zeros(nx,1) 
        for i in range(nx):
            for j in range(n_rxn):
                w = self.W0_sym[j]+np.dot(np.array([self.W1_sym[j,:]]),x)
                s = np.array([self.S[:,j]],dtype=np.float64).T
                if j==0:
                    f = w*( np.prod( np.power((x+s).T,M[i,:])) - np.prod( np.power(x.T,M[i,:])))
                else:
                    f+=w*( np.prod( np.power((x+s).T,M[i,:])) - np.prod( np.power(x.T,M[i,:])))
            RHS[i] = f
        # lambdify x
        RHS_func = lambdify(x,RHS)            
        # Convert RHS to a linear system. 
        self.B = zeros(nx,1)
        self.A = zeros(nx,nx)
        for i in range(nx):
            self.B[i] = RHS_func(*np.zeros((ns,1)))[i,0]
            for j in range(nx):
                tmp = RHS[i]
                for k in range(ns): 
                    tmp = 1/gamma(M[j,k]+1) * diff(tmp,x[k,0],int(M[j,k]))
                tmpf = lambdify(x,tmp)
                self.A[i,j] = tmpf(*np.zeros((ns,1)))

        #lambdify A and B in terms of model paramters. 
        self.A = lambdify(self.sym_pars,self.A)
        self.B = lambdify(self.sym_pars,self.B) 
        # convert to a linear system     
#        self.A = np.array(self.A).astype(np.float64)
#        self.B = np.array(self.B).astype(np.float64)
        self.RHS = RHS

    def arb_RHS_wrap(self,x,t):
        '''
        wrap the RHS functions
        so it can be integrated.
        '''
        x = np.array([x]).T
        tmp = np.dot(self.A_real,x)+self.B_real
        return tmp.ravel()
        
    def create_centered_moments_func(self,M):
        '''
        take the uncentered moments and get centered ones. 
        '''
        # get a symbolic vector for the uncentered moments
        nx,ns = M.shape 
#        u = MatrixSymbol('u',nx,1)
#        uu = Matrix(u)
#        uu[0,0] = 1.0
        # make other symbolic vector
        u = []
        for i in range(nx):
            u.append(Symbol('m'+str(i+1)))
        u = np.array([u]).T
        u[0]=1.0
        # Find the centered moments in terms of the 
        # uncentered moments, for those moments greater than 1.
        c = zeros(nx,1)
        for i in range(1,ns+1):
            c[i,0]=u[i]
        for i in range(1+ns,nx):
            b = M[i,:]
            c[i,0] = u[i,0]
            for ik in range(0,i)[::-1]: 
                bdiff = b-M[ik,:]
                if np.min(bdiff)>=0:
                    pref = u[ik,:] 
                    for ij in range(ns):
                        pref = pref*((-1)**bdiff[ij])*comb(b[ij],bdiff[ij])*u[ij+1]**bdiff[ij]
                    c[i] = c[i]+pref
        self.c = c
        u[0] = Symbol('m1')
        self.convert_to_centered = lambdify(u.ravel().tolist(),c) 
 
    def sensitivity_ODE(self,x,t):
        '''
        Gets the odes that describe the sensivity.
        '''
        # parse the first bits of x into means and variances
        _MU = x[:self.N]
        _SIG = np.reshape(x[self.N:self.N+self.N**2],(self.N,self.N))

        if not self.tv:
            inner = np.dot(self.W1,np.array([_MU]).T)+self.W0
            K_subs = self.K_subs_i(*x[:self.N+self.N**2])
            J_subs = self.J_subs
        else:
            # get_W(t) automatically update tv_dict
            self.get_W(t)
            inner = np.dot(self.W1,np.array([_MU]).T)+self.W0
            # time varying substitution for J
            J_subs = self.J_subs(*self.tv_vals)
            # time varying substitution for K
            changing_things = np.concatenate((x[:self.N+self.N**2],self.tv_vals))
            K_subs = self.K_subs_i(*changing_things)


        # Build diagonal matrix
        self.diag_mat = np.diag(np.ravel(inner),k=0)

        # Compute RHS
        RHS_vars = np.ravel(np.dot(np.dot(self.S,self.W1),_SIG) + np.dot(np.dot(_SIG,self.W1.T),self.S.T) + np.dot(np.dot(self.S,self.diag_mat),self.S.T))
        RHS_means = np.ravel(np.dot(self.S,(np.dot(self.W1,np.atleast_2d(_MU).T)))+np.dot(self.S,self.W0))
        return np.concatenate((np.concatenate((RHS_means,RHS_vars)),np.ravel(np.dot(J_subs,x[self.N+self.N**2:]) + np.ravel(K_subs) ))) 

    def get_FIM(self,order=1,tstart=0,log=False):
        '''
        Obtain the Fisher Information Matrix for the order specified
        by order.
        tstart is the index of tvec to keep for FIM computation.
        '''
        self.xi = np.zeros(self.N+self.N**2)
        self.solve(order=order)
        if self.tvec.any():
            self.ti = self.tvec[0]
            self.tf = self.tvec[-1]
            self.ptimes = len(self.tvec)
        else: 
            self.tvec = self.gettvec()
        self.get_S_sym()
        self.get_W_sym()
        self.get_variance_ODE_sym()
        self.params_to_keep = self.dpars
        self.N_free_params = len(self.params_to_keep)
        self.N_params = len(self.params)
        self.xi = np.zeros(2*(self.N+self.N**2))
        # get number of cells vector. 
        try:
            len(self.Nc) 
            if len(self.Nc) != self.ptimes:
                print( "Number of cells != number of times in data." )
                return
        except:
            self.Nc = np.repeat(self.Nc,self.ptimes) 
        self.Nc = self.Nc[tstart:]
        
        # substitute and convert jacobian to a numpy array.
        J_subs = self.J.subs(self.param_dict)
        if not self.tv:
            self.J_subs = np.array(J_subs.tolist()).astype(np.float64)
            changing_vars = self.species_list[:] 
#            for key in self.species_dict:
#                changing_vars.append(key)
            
        else:
            tv_syms = []
            self.tv_vals = []
            changing_vars = []
            for key in self.species_dict:
                changing_vars.append(key)
            for key in self.tv_dict:
                changing_vars.append(key)
                tv_syms.append(key)
                self.tv_vals.append(self.tv_dict[key])
            self.J_subs = lambdify(tv_syms,J_subs)
        # substitute parameters in K, but not mu and v.
        K_subs = self.K.subs(self.param_dict)
        
        # solve sensitivities.
        ntimes = len(self.tvec[tstart:])
        nblock = ntimes*self.N_observables
        self.tvec_fim = self.tvec[tstart:]
        
        self.dm_dtheta = np.zeros((nblock,self.N_free_params))
        self.dv_dtheta = np.zeros((nblock,nblock,self.N_free_params))
        self.FIM = np.zeros((len(self.params_to_keep),len(self.params_to_keep)))
        self.FIM1 = np.zeros((len(self.params_to_keep),len(self.params_to_keep)))
        self.ODE = self.sensitivity_ODE
        
        s_mu = self.N+self.N**2
        s_v = s_mu+self.N
        for i in range(self.N_free_params):
#            print( 'Integrating sensitivities for parameter %d' %i)
            start = time.time()
            # get the parameter derivatives.
#            print(K_subs)
            self.K_subs_i = lambdify(changing_vars,K_subs[:,self.params_to_keep[i]])
            self._solve()
            #self.dm_dtheta[:,i] = np.ravel(self.soln[s_mu+self.observables,tstart:].T)
            tmp = self.soln[s_mu:s_v,tstart:]
            self.dm_dtheta[:,i] = np.ravel(tmp[self.observables,:].T)
            tmp = np.reshape(self.soln[s_v:,tstart:],(self.N,self.N,ntimes))
            tmp2 = np.array([tmp[self.observables,:,:]])
            tmp3 = np.array([tmp2[:,self.observables,:]])
  
            # make block diagonal for covariances.   
            for j in range(ntimes):
                self.dv_dtheta[j*self.N_observables:j*self.N_observables+self.N_observables,j*self.N_observables:j*self.N_observables+self.N_observables,i] =tmp3[:,:,j]
            solve_time = time.time()-start
        # downsample
#        self.dv_dtheta = self.dv_dtheta[:,:,self.params_to_keep]
#        self.dm_dtheta = self.dm_dtheta[:,self.params_to_keep]
        
        # get mean/covariances at correct times.
        self.mean = np.zeros((self.N_observables,ntimes))
        self.covariances = np.zeros((self.N_observables,self.N_observables,ntimes))
        for t in range(ntimes):
            self.mean[:,t] = self.soln[self.observables,tstart+t]
            tmp =  np.reshape(self.soln[self.N:self.N+self.N**2,tstart+t],(self.N,self.N))
            tmp2 = np.array([tmp[self.observables,:]])
            tmp3 = np.array([tmp2[:,self.observables]])
            self.covariances[:,:,t] = tmp3
        # convert covariance to blocks.
        if order==2:
            self.block_vars = np.zeros((nblock,nblock))
            self.block_var_inv = np.zeros((nblock,nblock))
            for i in range(ntimes):
                if order==2:
                    self.block_vars[i*self.N_observables:i*self.N_observables+self.N_observables,i*self.N_observables:i*self.N_observables+self.N_observables] = self.covariances[:,:,i]
                    try:
                        self.block_var_inv[i*self.N_observables:i*self.N_observables+self.N_observables,i*self.N_observables:i*self.N_observables+self.N_observables] = np.linalg.inv(self.covariances[:,:,i])
                    except:
                        self.block_var_inv[i*self.N_observables:i*self.N_observables+self.N_observables,i*self.N_observables:i*self.N_observables+self.N_observables] = np.linalg.pinv(self.covariances[:,:,i])
                        print('Unable to take inverse at time index %d, used pseudo-inverse instead.' %i)     
        elif order ==1:      
            try:
               self.block_var_inv = np.linalg.inv(self.block_vars)
            except:
                self.block_var_inv = np.linalg.pinv(self.block_vars)
                print( 'Used pseudo inverse for order = 1')

        # compute the FIM
#        print(self.Nc)
        for i in range(len(self.params_to_keep)):
            for j in range(len(self.params_to_keep)):
                x1 = (np.repeat(self.Nc,self.N_observables)*self.dm_dtheta.T).T
                if order == 1:
                    self.FIM[i,j] = np.dot(np.dot(x1[:,i].T,self.block_var_inv),self.dm_dtheta[:,j])
                elif order == 2:
                     # cell number scaling. 
                     v = np.zeros((nblock,nblock))
                     for k in range(ntimes):
                        v[k*self.N_observables:k*self.N_observables+self.N_observables,k*self.N_observables:k*self.N_observables+self.N_observables] = self.Nc[k]*self.block_var_inv[k*self.N_observables:k*self.N_observables+self.N_observables,k*self.N_observables:k*self.N_observables+self.N_observables]
                     # make a block diagonal from each covariance at each time.
                     #self.FIM[i,j] = np.dot(np.dot(x1[:,i].T,self.block_var_inv),self.dm_dtheta[:,j]) + .5 * np.trace(np.dot(np.dot(v,self.dv_dtheta[:,:,i]),np.dot(self.block_var_inv,self.dv_dtheta[:,:,j])))
                     self.FIM[i,j] = np.dot(np.dot(self.dm_dtheta[:,i].T,v),self.dm_dtheta[:,j]) + .5 * np.trace(np.dot(np.dot(v,self.dv_dtheta[:,:,i]),np.dot(self.block_var_inv,self.dv_dtheta[:,:,j])))
                     self.FIM1[i,j] = np.dot(np.dot(self.dm_dtheta[:,i].T,self.block_var_inv),self.dm_dtheta[:,j])
        if log:
            self.FIM[i,j] = self.FIM[i,j]*self.params[self.dpars][i]*self.params[self.dpars][j]
