import sys
sys.path.append('../')
import generic_solvers
#reload(generic_solvers)
from generic_solvers import GenericFSP,GenericSSA
import numpy as np
import scipy.sparse as sp
import random
import time
import scipy.linalg as linalg
from sympy import Symbol,Matrix,diff,zeros,lambdify

class ToggleFSP(GenericFSP):
    def __init__(self,MX,MY,params,tend,truncation=2500):
        '''
        initialize the Model, supplying some relevant parameters
        or the model. MX is the # of species X, MY is the # of species Y
        '''
        self.tMX = MX+1; self.MX=MX;
        self.tMY = MY+1; self.MY = MY;
        self.totalMSize = self.tMX*self.tMY
        self.params = params
        self.dpars = [0,1,2,3,4,5,9]
        self.N_params = len(params)
        self.tf = tend
        self.ptimes = 3
        self.ti = 0
        self.truncation = truncation
        self.Nc = 100

    def solve(self):
        '''
        solve dp/dt = Ap.
        '''        
        self.make_projection()
        self.N = self.proj_size
        self.xi = np.zeros(self.proj_size)
        self.xi[30*self.tMY+6] = 1.0
        #self.pi = self.xi
        self.pi = np.loadtxt('out/initial_condition.txt') 
        self.A =  self.get_A()
        self._solve()
        self.p = np.zeros((self.totalMSize,len(self.tvec)))
        for i in range(len(self.tvec)):
            self.p[self.proj_space,i] = self.soln[:,i]

    def solve_piecewise(self,uv=0):
        '''
        Solver for different UV levels.
        '''
        self.full_tvec = np.copy(self.tvec)
        self.gettvec()
        A1 = self.getA_piecewise(0)
        A2 = self.getA_piecewise(uv)
        self.pi = np.loadtxt('out/initial_condition.txt')
        p0 = np.loadtxt('out/initial_condition.txt')
        self.tvec = self.tvec_early_off
        self.A = A1
        solns_1 = self._solve()
        # Update initial condition, solve the next step
        self.pi = self.soln[:,-1] 
        self.tvec =  self.tvec_on
        self.A = A2
        start = time.time()
        solns_2 = self._solve()
        print(time.time()-start)
        # Update initial condition, solve the next step
        self.pi = self.soln[:,-1]
        self.tvec =  self.tvec_late_off
        self.A = A1
        start = time.time()
        solns_3 = self._solve()
        print(time.time()-start)
        # concatenate solutions, only keeping the original desired time points
        ptmp = np.hstack((np.array([p0]).T,solns_1,solns_2[:,1:],solns_3[:,1:]))[:,self.times_to_keep.astype(bool)]
        # reset the time vector
        self.tvec = np.copy(self.full_tvec)
        # finally, re-order the solutions. 
        self.p = np.zeros((self.totalMSize,len(self.tvec)))
        for i in range(len(self.tvec)):
            self.p[self.proj_space,i] = ptmp[:,i]

    def get_sym_A(self):
        '''
        get a symbolic version of A. 
        '''
        bx = Symbol('bx'); by = Symbol('by')
        kx = Symbol('kx'); ky = Symbol('ky')
        ayx = Symbol('ayx'); axy = Symbol('axy')
        #nyx = Symbol('nyx'); nxy = Symbol('nxy')
        nyx = self.params[6]; nxy = self.params[7] 
        dx  = Symbol('dx'); dy = Symbol('dy')
        self.sym_pars = [bx,by,kx,ky,ayx,axy,Symbol('nyx'),Symbol('nxy'),dx,dy]
        self.A_sym = zeros(self.totalMSize,self.totalMSize)
        x = np.repeat(np.arange(self.tMX),self.tMY)
        y = np.tile(np.arange(self.tMY),self.tMX)
        
        for i in range(self.totalMSize):    
            for j in range(self.totalMSize):
                if i==j:
                    w1 = bx+(kx/(1+ayx*(y[i])**nyx))
                    w2 = by + (ky/(1+axy*(x[i])**nxy))
                    self.A_sym[i,j] = -w1-w2-dx*x[i]-dy*y[i]
                #if (i<self.totalMSize+1) and i==j-1:
                if i==j-1:
                    self.A_sym[i,j] = dy*y[j]
                if  i==j-self.tMY:   
                    self.A_sym[i,j] = dx*x[j]
                if  i==j+1:
                    w2 = by + (ky/(1+axy*(x[j])**nxy))
                    self.A_sym[i,j] = w2 
                if  i==j+self.tMY:
                    w1 = bx+(kx/(1+ayx*(y[j])**nyx))
                    self.A_sym[i,j] = w1
        tmp = self.A_sym[self.proj_space,:]
        self.A_sym = tmp[:,self.proj_space]

    def get_Ai(self,dx,dy,w1,w2):
        '''
        get the A_i matrices.
        '''
        A_i = np.zeros((self.totalMSize,self.totalMSize))
        x = np.repeat(np.arange(self.tMX),self.tMY)
        y = np.tile(np.arange(self.tMY),self.tMX)
        for i in range(self.totalMSize):
            for j in range(self.totalMSize):
                if i==j:
                    all_params = np.concatenate((self.params,[x[j]],[y[j]]))
                    A_i[i,j] = -w1(*all_params)-w2(*all_params)-dx(*all_params)-dy(*all_params)
                #if (i<self.totalMSize+1) and i==j-1:
                elif i==j-1:
                    all_params = np.concatenate((self.params,[x[j]],[y[j]]))
                    A_i[i,j] = dy(*all_params)
                elif  i==j-self.tMY:   
                    all_params = np.concatenate((self.params,[x[j]],[y[j]]))
                    A_i[i,j] = dx(*all_params)
                elif  i==j+1:
                    all_params = np.concatenate((self.params,[x[j]],[y[j]]))
                    A_i[i,j] = w2(*all_params)
                elif  i==j+self.tMY:
                    all_params = np.concatenate((self.params,[x[j]],[y[j]]))
                    A_i[i,j] = w1(*all_params)
        tmp = A_i[self.proj_space,:]
        A_i = tmp[:,self.proj_space]
        return A_i

    def get_rxn_fnc(self):
        bx = Symbol('bx'); by = Symbol('by')
        kx = Symbol('kx'); ky = Symbol('ky')
        ayx = Symbol('ayx'); axy = Symbol('axy')
        #nyx = Symbol('nyx'); nxy = Symbol('nxy')
        nyx = self.params[6]; nxy = self.params[7]
        dx  = Symbol('dx'); dy = Symbol('dy')
        x = Symbol('x'); y = Symbol('y')
        self.sym_pars = [bx,by,kx,ky,ayx,axy,Symbol('nyx'),Symbol('nxy'),dx,dy]
        mixed_params = []
        for i in range(len(self.params)):
            if i in self.dpars:
                mixed_params.append(self.sym_pars[i])
            else:
                mixed_params.append(self.params[i])
        # deal out
        bx,by,kx,ky,ayx,axy,nyx,nxy,dx,dy = mixed_params
        w1 = bx+(kx/(1+ayx*(y**nyx)))
        w2 = by + (ky/(1+axy*(x**nxy)))
        self.build_param_dict()
        rstrings = self.param_strings[:]
        rstrings.append('x'); rstrings.append('y')
        dxs=[]; dys=[]; w1s=[]; w2s=[]
        for i in self.dpars:
            dxs.append(lambdify(rstrings,diff(dx*x,mixed_params[i])))
            dys.append(lambdify(rstrings,diff(dy*y,mixed_params[i])))
            w1s.append(lambdify(rstrings,diff(w1,mixed_params[i])))
            w2s.append(lambdify(rstrings,diff(w2,mixed_params[i])))
        return dxs,dys,w1s,w2s

    def build_param_dict(self):
        '''
        Build a dictionary for symbolic parameters
        and numerical ones.
        '''
        self.param_dict = {}
        self.param_strings = []
        for i in range(self.N_params):
            self.param_dict[str(self.sym_pars[i])] = self.params[i]
            self.param_strings.append(str(self.sym_pars[i]))

    def get_Q(self):
        '''
        Qs are the derivatives of A with respect
        to each i.
        '''
        dxs,dys,w1s,w2s = self.get_rxn_fnc()
        #self.get_sym_A() 
        self.build_param_dict()
        self.Qs = []
        A = self.get_A().toarray()
        msize,msize = A.shape
        for i in range(len(self.dpars)):
            Q = np.zeros( ( 2*msize,2*msize))
            Q[:msize,:msize] = A
            Q[msize:,:msize] = self.get_Ai(dxs[i],dys[i],w1s[i],w2s[i])
            Q[msize:,msize:] = A
            self.Qs.append(Q)

    def get_Q_piecewise(self):
        '''
        Qs are the derivatives of A with respect 
        to each i. 
        '''
        dxs,dys,w1s,w2s = self.get_rxn_fnc()
        #self.get_sym_A() 
        self.build_param_dict()
        self.Qs = []
        A = self.get_A().toarray()
        msize,msize = A.shape
        for i in range(len(self.dpars)):
            Q = self.get_Ai(dxs[i],dys[i],w1s[i],w2s[i])
            self.Qs.append(Q)

    def get_sensitivity(self,tstart=0,log=False):
        '''
        Solve the sensitivity matrix for the system.
        '''
        self.get_Q()
        self.ss = []
        for i in range(len(self.dpars)):
            self.A = sp.csc_matrix(self.Qs[i])
            start = time.time()
            self._solve()
            print('Sensitivity solution shape: {0}'.format(self.soln.shape))
            tsolve = time.time()-start
            print("Time to solve sensitivity: %f" %tsolve)
            self.p = self.soln[:self.proj_size,tstart:]
            print('PDF shape: {0}'.format(self.p.shape))
            if log:
                self.ss.append(self.params[self.dpars[i]]*self.soln[self.proj_size:,tstart:].ravel())
            else:
                self.ss.append(self.soln[self.proj_size:,tstart:].ravel())
        ntimes = len(self.tvec[tstart:])
        # Make the full S matrix
        self.S = np.vstack(self.ss).T
        # threshold on p to avoid overflow error.
        small_p = self.p<1e-8
        self.p[small_p] = 1e-8
        z = 1.0/self.p
        z[small_p] = 0.0
        self.P_diag = np.diag(z.ravel())
        self.p[small_p] = 0.0

    def get_sensitivity_piecewise(self,tstart=0,log=False,uv=0):
        '''
        Solve the sensitivity matrix for the system.
        '''
        self.get_Q_piecewise()
        self.ss = []
        A1 = self.getA_piecewise(0).toarray()
        A2 = self.getA_piecewise(uv).toarray()
        for i in range(len(self.Qs)):
            # solve the first interval
            self.pi = np.zeros(2*self.proj_size)
            self.pi[:self.proj_size] = np.loadtxt('out/initial_condition.txt')
            self.tvec =  self.tvec_early_off
            Q = np.zeros((2*self.proj_size,2*self.proj_size))
            Q[:self.proj_size,:self.proj_size] = A1
            Q[self.proj_size:,:self.proj_size] = self.Qs[i]
            Q[self.proj_size:,self.proj_size:] = A1
            self.A = sp.csc_matrix(Q)
            solns_1 = self._solve()
            # Update initial condition, solve the next step
            self.pi = self.soln[:,-1]
            Q[:self.proj_size,:self.proj_size] = A2
            Q[self.proj_size:,self.proj_size:] = A2
            self.tvec =  self.tvec_on
            self.A = sp.csc_matrix(Q)
            solns_2 = self._solve()
            # Update initial condition, solve the next step
            self.pi = self.soln[:,-1]
            self.tvec =  self.tvec_late_off
            Q[:self.proj_size,:self.proj_size] = A1
            Q[self.proj_size:,self.proj_size:] = A1
            self.A = sp.csc_matrix(Q)
            solns_3 = self._solve()
            # concatenate solutions, only keeping the original desired time points
            self.soln = np.hstack((solns_1,solns_2[:,1:],solns_3[:,1:]))[:,self.times_to_keep.astype(bool)]
            self.p = self.soln[:self.proj_size,tstart:]
            if log:
                self.ss.append(self.params[self.dpars[i]]*self.soln[self.proj_size:,tstart:].ravel())
            else:
                self.ss.append(self.soln[self.proj_size:,tstart:].ravel())
        # Add in a check for the observable dimension of p(x)
#       if len(self.observable) == 1:
#           for i in range(len(self.Qs)):
#               self.ss_new[i].append(np.sum(self.ss[i].reshape(nt,self.nx,self.ny),axis=self.observable[0]))
#       self.ss = self.ss_new
        self.S = np.vstack(self.ss).T
        self.tvec = self.full_tvec
        ntimes = len(self.tvec[tstart:])
        # threshold on p to avoid overflow error. 
        small_p = self.p<1e-8
        self.p[small_p] = 1e-8
        z = 1.0/self.p
        z[small_p] = 0.0
        self.P_diag = np.diag(z.ravel())
        self.p[small_p] = 0.0

    def propensity_functions(self,x,y):
        '''
        hill functions
        '''
        bx,by,kx,ky,ayx,axy,nyx,nxy,dx,dy = self.params
        w1 = bx+(kx/(1+ayx*(y)**nyx))
        w2 = by + (ky/(1+axy*(x)**nxy))
        return (w1,w2,dx,dy)

    def get_DTD(self,tstart=0):
        '''
        Find E{D^T D} using theory.
        '''
        print('Using {0} cells'.format(self.Nc))
        ntimes = len(self.tvec[tstart:])
        p_ravel = self.p.ravel()
        msize = len(p_ravel)
        self.DTD = np.zeros((msize,msize))
        # Assign the diagonal.
        pr2 = p_ravel**2
        DTD_diag = (self.Nc**2)*pr2+self.Nc*p_ravel-self.Nc*pr2
        self.DTD= (self.Nc**2-self.Nc)*np.dot(np.array([p_ravel]).T,np.array([p_ravel]))
        np.fill_diagonal(self.DTD,DTD_diag)
        return self.DTD

    def get_FIM_piecewise(self,tstart=0,log=False,uv=0):
        '''
        get the FIM for the full FSP model.
        '''
        # hardcoded for now.
        self.make_projection()
        self.full_tvec = np.copy(self.tvec)
        self.gettvec()

        self.N = self.proj_size
        # define the initial condition for the sensitivty.. 
        self.pi =np.loadtxt('out/initial_condition.txt') 
        self.get_sensitivity_piecewise(tstart=tstart,log=log,uv=uv)
        # modify to use new analysis.
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

    def get_FIM(self,tstart=0,log=False):
        '''
        get the FIM for the full FSP model.
        '''
        # hardcoded for now.
        self.make_projection()
        self.N = self.proj_size
        # define the initial condition for the sensitivty.. 
        self.pi = np.zeros(2*self.proj_size)
        #self.xi[30*self.tMY+6] = 1.0
        self.pi[:self.proj_size] = np.loadtxt('out/initial_condition.txt')
        self.get_sensitivity(tstart=tstart,log=log)
        # modify to use new analysis.
#        DTD = self.get_DTD(tstart=tstart)
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

    def model_build(self):
        '''
        Construct the diagonals for the toggle model
        '''
        bx,by,kx,ky,ayx,axy,nyx,nxy,dx,dy = self.params
        k1ForMain = np.tile(np.arange(self.tMY)*dy,self.tMX)
        k1Diag = k1ForMain[1:]
        yw1 = np.repeat(np.arange(self.tMX),self.tMY)
        k_1ForMain = self.propensity_functions(yw1,0)[1]
        k_1Diag = np.copy(k_1ForMain)
        k_1Diag[self.MY::self.tMY]=0.0
        k_1Diag = k_1Diag[:-1]
        kNForMain = np.repeat(np.arange(self.tMX)*dx,self.tMY)
        kNDiag = kNForMain[self.tMY:]
        k_NForMain = self.propensity_functions(0,np.tile(np.arange(self.tMY),self.tMX))[0]
        k_NDiag =k_NForMain[:-self.tMY]
        mainDiag = -(k1ForMain+k_1ForMain+kNForMain+k_NForMain)
        return (mainDiag,k1Diag,kNDiag,k_1Diag,k_NDiag)

    def model_build_piecewise(self,uv):
        '''
        Construct the diagonals for the toggle model
        '''
        bx,by,kx,ky,ayx,axy,nyx,nxy,dx,dy = self.params
        dx = 3.84e-4 + .02*uv**2/(2500.0+uv**3)
        k1ForMain = np.tile(np.arange(self.tMY)*dy,self.tMX)
        k1Diag = k1ForMain[1:]
        yw1 = np.repeat(np.arange(self.tMX),self.tMY)
        k_1ForMain = self.propensity_functions(yw1,0)[1]
        k_1Diag = np.copy(k_1ForMain)
        k_1Diag[self.MY::self.tMY]=0.0
        k_1Diag = k_1Diag[:-1]
        kNForMain = np.repeat(np.arange(self.tMX)*dx,self.tMY)
        kNDiag = kNForMain[self.tMY:]
        k_NForMain = self.propensity_functions(0,np.tile(np.arange(self.tMY),self.tMX))[0]
        k_NDiag =k_NForMain[:-self.tMY]
        mainDiag = -(k1ForMain+k_1ForMain+kNForMain+k_NForMain)
        return (mainDiag,k1Diag,kNDiag,k_1Diag,k_NDiag)

    def make_projection(self):
        '''
        Get the projection space for the Toggle switch using a simple function 
        '''
        a = np.repeat(np.arange(self.tMX),self.tMY)
        b = np.tile(np.arange(self.tMY),self.tMX)
        c = (a-4)*(b-4)
        d = c<self.truncation
        self.proj_space = np.ravel(np.nonzero(d))
        self.proj_size = len(self.proj_space)
        return self.proj_space 

    def get_A(self):
        '''
        Build sparse matrix in CSR format.
        '''
        matrix = sp.diags(self.model_build(),[0,1,self.tMY,-1,-self.tMY],format = 'csr')
        PI = self.make_projection()
        smaller = matrix[PI,:]
        smallest = smaller[:,PI]
        self.A = smallest
        return self.A

    def getA_piecewise(self,uv=0):
        '''
        Build sparse matrix in CSR format.  
        '''
        matrix = sp.diags(self.model_build_piecewise(uv),[0,1,self.tMY,-1,-self.tMY],format = 'csr')
        PI = self.make_projection()
        smaller = matrix[PI,:]
        smallest = smaller[:,PI]
        self.A = smallest        
        return self.A 

    def gettvec(self):
        '''
        update the time vector.
        '''
        self.toff = self.ton+self.delta
        # make full tvec with ton and toff
        full_tvec_new = np.unique(np.sort(np.concatenate((self.full_tvec,[self.ton,self.toff]))))
        self.full_tvec_cat = np.copy(full_tvec_new) 
        # figure out where ton and toff ended up to remove them later
        self.times_to_keep = np.ones(len(full_tvec_new))
        if self.ton not in self.full_tvec:
            self.ton_loc = np.where(full_tvec_new==self.ton)[0][0]
            self.times_to_keep[self.ton_loc]= 0
        else:
            self.ton_loc = np.where(full_tvec_new==self.ton)[0][0]
        if self.toff not in self.full_tvec:
            self.toff_loc = np.where(full_tvec_new==self.toff)[0][0] 
            self.times_to_keep[self.toff_loc] = 0
        else:
            self.toff_loc = np.where(full_tvec_new==self.toff)[0][0] 
        
        # breakdown the vector into piecewise-constant intervals.
        self.tvec_early_off = full_tvec_new[:self.ton_loc+1]
        self.tvec_on = full_tvec_new[self.ton_loc:self.toff_loc+1] 
        self.tvec_late_off = full_tvec_new[self.toff_loc:]

class ToggleSSA(GenericSSA):
    def __init__(self,params,tend):
        '''
        initialize the Model, supplying some relevant parameters
        or the model. MX is the # of species X, MY is the # of species Y
        '''
        self.params = params
        self.N_params = len(params)
        self.tf = tend
        self.ptimes = 3
        self.ti = 0
        self.Nc = 100
        self.xi = np.array([30,5])
        self.type = 'nonlinear'

    def solve(self,n=1):
        '''
        solve the ssa   
        '''
        self.fast_rxn = .00000000000005
        self.get_S()
        self._solve(n)
        self.get_dist()

    def get_P(self,x,t):
        '''
        get the propensities.
        '''
        bx,by,kx,ky,ayx,axy,nyx,nxy,dx,dy = self.params
        w1 = bx+(kx/(1+ayx*(x[1])**nyx))
        w2 = by + (ky/(1+axy*(x[0])**nxy))
        return np.array([ w1,dx*x[0],w2,dy*x[1]])
        
    def get_S(self):
        '''
        Get the stoichiometry matrix 
        '''
        self.S = np.array([ [1,0],
                            [-1,0],
                            [0,1],
                            [0,-1]]).T
            
    def get_dist(self):
        '''
        build distribution (non-normalized and pdf)
        of rna for the model)
        '''
        n_specs, n_times, n_traj = self.data.shape
        max_rna_x = int(np.max(self.data[0,:,:]))
        max_rna_y = int(np.max(self.data[1,:,:]))
        self.pdf = np.zeros((max_rna_x+1,max_rna_y+1,n_times))
        self.fdist = np.zeros((max_rna_x+1,max_rna_y+1,n_times))
        for i in range(n_times):
            for j in range(n_traj):
                x = int(self.data[0,i,j])
                y = int(self.data[1,i,j])
                self.fdist[x,y,i] += 1
            self.pdf[:,:,i] = self.fdist[:,:,i] / np.sum(self.fdist[:,:,i])
