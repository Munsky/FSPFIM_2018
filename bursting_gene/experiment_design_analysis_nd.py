import sys
import matplotlib
matplotlib.use('Agg')
sys.path.append('../')
import bursting_gene as bg
#reload(bg)
import numpy as np
import matplotlib.pyplot as plt
import vis
from scipy.stats import multivariate_normal 

'''
Design the optimal sampling period of bursting gene models using the FIM.
Verification loads data that was generated remotely. 
'''

def nd_design(load=False,scale_time=False):    
    '''
    Load the parameters for each sampling frequency/data set,
    and compute the covariance.
    '''
    fsamp = np.logspace(-3,1,50) 
    f,ax = plt.subplots()
    if not load:
        n_data = 200
        E_opts = []
        D_opts = []
        for i in range(len(fsamp)):
            tmp_pars = [] 
            count = 0
            for j in range(n_data):
                try:
                     tmp_pars.append(10**np.loadtxt('out/fsamp_nd/params_pdf_1.0_fsamp_bursting_gene_exp_{0}_fsamp_{1}.txt'.format(j,i)))
                     #tmp_pars.append(np.log(10**np.loadtxt('out/fsamp_nd/fsamp_old_2/params_pdf_1.0_fsamp_bursting_gene_exp_{0}_fsamp_{1}.txt'.format(j,i))))
                except:
                    count+=1
            tmp_pars = np.array(tmp_pars) 
            tmp_cov = np.cov(np.array(tmp_pars).T)
            if scale_time:
                try:
                    obs_inf = np.linalg.inv(tmp_cov)/400.0
                except:
                    obs_inf = np.zeros((2,2))
            else:
                try:
                    obs_inf = np.linalg.inv(tmp_cov)
                except:
                    obs_inf = np.zeros((2,2))
            vals, vec = np.linalg.eig(obs_inf)
            E_opts.append(np.min(vals))
            D_opts.append(np.prod(vals))
        np.savetxt('out/fsamp_nd/E_opt_nd.txt',E_opts)
        np.savetxt('out/fsamp_nd/D_opt_nd.txt',D_opts)
    elif load:
        E_opts = np.loadtxt('out/fsamp_nd/E_opt_nd.txt')
        D_opts = np.loadtxt('out/fsamp_nd/D_opt_nd.txt')

    return fsamp,E_opts,D_opts

def design_fsamp_fim(kon,fsamp=np.array([]),load=True,save=True,smoments=False,scale_time=False):
    '''
    A code to design experiment design using the fisher information for 
    the bursting gene model. 
    '''
    count = 0
    kr = 100
    g = 1.0
    
    params = [kon,3*kon,kr,g]
    start = 1
    if not fsamp.any():
        fs = np.logspace(-3,1.5,50)
        #fs = np.logspace(-1,3,10)
    else:
        fs = fsamp
    ptimes = 5
    if load:     
        D_opt_fsp = np.loadtxt('out/D_opt_fsamp_fsp'+str(kon)+'_'+str(kr)+'.txt')
        D_opt_moments = np.loadtxt('out/D_opt_fsamp_moments'+str(kon)+'_'+str(kr)+'.txt')
        E_opt_fsp = np.loadtxt('out/E_opt_fsamp_fsp'+str(kon)+'_'+str(kr)+'.txt')
        E_opt_moments = np.loadtxt('out/E_opt_fsamp_moments'+str(kon)+'_'+str(kr)+'.txt')
        if smoments:
            D_opt_smoments = np.loadtxt('out/D_opt_fsamp_smoments'+str(kon)+'_'+str(kr)+'.txt')
            E_opt_smoments = np.loadtxt('out/E_opt_fsamp_smoments'+str(kon)+'_'+str(kr)+'.txt')
    else: 
        D_opt_fsp = []
        D_opt_moments = []
        
        E_opt_fsp = [] 
        E_opt_moments = [] 

        if smoments:
            E_opt_smoments = [] 
            D_opt_smoments = []
                     
        tfs = [] 
        log_fim = False
        for i in range(len(fs)):
            print('iteration: %d' %i)
            # Make FSP model, get FIM
            fsp = bg.BurstingGeneFSP(params)
            fsp.Nc = 1000
            fsp.N = 200 
            fsp.ptimes = ptimes
            fsp.tf = fsp.ptimes * fs[i]
            fsp.get_FIM(tstart=start,rna_only=True,log=log_fim)
             
            # Make moment model, get FIM
            moments = bg.BurstingGeneMoments(params)
            moments.Nc = 1000
            moments.N = 2
            moments.ptimes = ptimes
            moments.tf = moments.ptimes*fs[i] 
            #moments.get_FIM(tstart=start,order=2)
            moments.get_FIM(tstart=start,order=2,log=log_fim)

            if smoments:
                smoments = bg.BurstingGeneMoments(params)
                smoments.Nc = 1000
                smoments.N = 2
                smoments.ptimes = ptimes
                smoments.tf = smoments.ptimes*fs[i] 
                #smoments.get_sample_FIM(tstart=start,rna_only=True)
                smoments.get_sample_FIM(tstart=start,rna_only=True,log=log_fim)

            # Append optimality vectors. 
            if scale_time:
                D_opt_moments.append(np.linalg.det(moments.FIM/400.0))
                val_moments,vec_moments = np.linalg.eig(moments.FIM/400.0)
                E_opt_moments.append(np.min(val_moments))
    
                D_opt_fsp.append(np.linalg.det(fsp.FIM/400.0))
                val_moments,vec_moments = np.linalg.eig(fsp.FIM/400.0)
                E_opt_fsp.append(np.min(val_moments))
    
                if smoments:
                    D_opt_smoments.append(np.linalg.det(smoments.FIM/400.0))
                    val_smoments,vec_smoments = np.linalg.eig(smoments.FIM/400.0)
                    E_opt_smoments.append(np.min(val_smoments))
            else:
                D_opt_moments.append(np.linalg.det(moments.FIM))
                val_moments,vec_moments = np.linalg.eig(moments.FIM)
                E_opt_moments.append(np.min(val_moments))
    
                D_opt_fsp.append(np.linalg.det(fsp.FIM))
                val_moments,vec_moments = np.linalg.eig(fsp.FIM)
                E_opt_fsp.append(np.min(val_moments))
    
                if smoments:
                    D_opt_smoments.append(np.linalg.det(smoments.FIM))
                    val_smoments,vec_smoments = np.linalg.eig(smoments.FIM)
                    E_opt_smoments.append(np.min(val_smoments))
            
            tfs.append(fsp.tf)
    
    # save results
    if save:
        np.savetxt('../out/D_opt_fsamp_fsp'+str(kon)+'_'+str(kr)+'.txt',D_opt_fsp)
        np.savetxt('../out/D_opt_fsamp_moments'+str(kon)+'_'+str(kr)+'.txt',D_opt_moments)
    
        np.savetxt('../out/E_opt_fsamp_fsp'+str(kon)+'_'+str(kr)+'.txt',E_opt_fsp)
        np.savetxt('../out/E_opt_fsamp_moments'+str(kon)+'_'+str(kr)+'.txt',E_opt_moments)
        if smoments:
            np.savetxt('../out/D_opt_fsamp_smoments'+str(kon)+'_'+str(kr)+'.txt',D_opt_smoments)
            np.savetxt('../out/E_opt_fsamp_smoments'+str(kon)+'_'+str(kr)+'.txt',E_opt_smoments)
            return fs, E_opt_fsp, E_opt_moments, E_opt_smoments, D_opt_fsp, D_opt_moments, D_opt_smoments
        else:
            return fs, E_opt_fsp, E_opt_moments, D_opt_fsp, D_opt_moments
    else: 
        if smoments:
            return fs, E_opt_fsp, E_opt_moments, E_opt_smoments, D_opt_fsp, D_opt_moments, D_opt_smoments
        else:
            return fs, E_opt_fsp, E_opt_moments, D_opt_fsp, D_opt_moments

def plot_designs(kon,fsamp_fim,E_opt_fsp,E_opt_moments,fsamp_nd=None,E_opt_nd_fsp=None,E_opt_nd_moments=None,E_opt_smoments=None,rescale=True,scale_time=False):
    '''
    plot the experiment designs
    '''
    if E_opt_smoments is not None:
        smoments = True
    else:
        smoments=False

    if scale_time:
        fsamp_fim=fsamp_fim*20*1.25
        fsamp_nd=fsamp_nd*20*1.25
    imax_moments = np.argmax(E_opt_moments)
    print('Moments Best FSAMP: %f' %fsamp_fim[imax_moments])
    imax_fsp  = np.argmax(E_opt_fsp)
    print('FSP Best FSAMP: %f' %fsamp_fim[imax_fsp])
    imax_smoments = np.argmax(E_opt_smoments)
    print('Sample moments Best FSAMP: %f' %fsamp_fim[imax_smoments])
    xlabs =np.array([.001,.01,.1,1.0,10.0,100.0]) 
    f,ax = plt.subplots()
    if rescale:
        ax.plot(fsamp_fim,E_opt_fsp/np.max(E_opt_fsp),'dodgerblue',linewidth=4,zorder=1)
        ax.plot(fsamp_fim,E_opt_moments/np.max(E_opt_moments),'mediumorchid',linewidth=4,zorder=2)
    else: 
        ax.plot(fsamp_fim,E_opt_fsp,'dodgerblue',linewidth=4,zorder=1)
        ax.plot(fsamp_fim,E_opt_moments,'mediumorchid',linewidth=4,zorder=2)

    if smoments:
        if rescale:
            ax.plot(fsamp_fim,E_opt_smoments/np.max(E_opt_smoments),'limegreen',linewidth=4)
        else:
            ax.plot(fsamp_fim,E_opt_smoments,'limegreen',linewidth=4)
    x = np.array([fsamp_fim[imax_moments],fsamp_fim[imax_fsp],fsamp_fim[imax_smoments]])
    if rescale:
        y = np.array([E_opt_moments[imax_moments]/np.max(E_opt_moments),E_opt_fsp[imax_fsp]/np.max(E_opt_fsp),E_opt_smoments[imax_smoments]/np.max(E_opt_smoments)])
    else:
        y = np.array([E_opt_moments[imax_moments],E_opt_fsp[imax_fsp],E_opt_smoments[imax_smoments]])
    ax.scatter(x,y,color='k',s=65,zorder=3)
    
    if fsamp_nd is not None:
        print('Plotting MCMC onto experiment design figure')
        if rescale:
            ax.scatter(fsamp_nd,E_opt_nd_fsp/np.max(E_opt_fsp),c ='coral',marker="^",zorder=4,s=45)
        else:
            ax.scatter(fsamp_nd,E_opt_nd_fsp,c ='coral',marker="^",zorder=4,s=45)
        #ax.scatter(fsamp_nd,E_opt_nd_fsp,c = 'coral',marker="^",zorder=4,s=55)
        
    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta t$',size=16)
    ax.set_ylabel('E-Optimality',size=16)
    ax.tick_params(labelsize=14)

    plt.tight_layout()
    f.show()

def plot_design_scatter(E_opt_fsp,E_opt_mcmc_fsp,E_opt_moments=None,E_opt_smoments=None,E_mcmc_moments=None):
    '''
    make a scatter plot
    '''
    # solve for m and b
    y = np.array([E_opt_mcmc_fsp]).T
    X = np.vstack((E_opt_fsp,np.ones(len(E_opt_fsp)))).T
    m,b = np.linalg.lstsq(X,y)[0]
    print('****')
    print('Slope: {0}'.format(m))
    print('Y-intercept: {0}'.format(b))
    # make plot
    f,ax = plt.subplots()
    ax.scatter(E_opt_fsp,E_opt_mcmc_fsp,c='dodgerblue',marker='o',s=50,zorder=2)
    if E_opt_moments is not None:
        c2 = (33/255,145/255,251/255)
        ax.scatter(E_opt_moments,E_opt_mcmc_fsp,c='mediumorchid',marker='s',s=50,zorder=3)
    if E_opt_smoments is not None:
        ax.scatter(E_opt_smoments,E_opt_mcmc_fsp,c='limegreen',marker='+',s=50,zorder=3)
    ax.set_xlabel('Predicted E-optimality',size=16)
    ax.set_ylabel('Measured E-optimality',size=16)
    ax.tick_params(labelsize=14)
    x_vals  = ax.get_xlim()
    y_vals = m*x_vals+b
    ax.plot(x_vals,x_vals,'k--',linewidth=2.5,zorder=1)
    ax.set_xlim([-.00002,.0003])
    ax.set_ylim([-.00002,.00035])

    plt.tight_layout()
    f.show()
    return

def main():
    '''
    main code.
    '''
    # get designs from monte-carlo sampling
    fsamp_nd,E_opt_nd,D_opt_nd = nd_design(load=False,scale_time=True)
    # get fim designs
    fsamp_fim, E_opt_fsp, E_opt_moments, E_opt_smoments, D_opt_fsp, D_opt_moments, D_opt_smoments = design_fsamp_fim(1.0,load=False,smoments=True,save=True,scale_time=True)

    # Plotting 
    # Plot E-optimality
    plot_designs(1.0,fsamp_fim,E_opt_fsp,E_opt_moments,fsamp_nd,E_opt_nd,E_opt_smoments = E_opt_smoments,rescale=False,scale_time=True)
    # Get the FIMs only at the same points as the monte carlo sampling
    fsamp_fim, E_opt_fsp, E_opt_moments, E_opt_smoments, D_opt_fsp, D_opt_moments, D_opt_smoments = design_fsamp_fim(1.0,fsamp_nd,load=False,smoments=True,save=False,scale_time=True)
    plot_design_scatter(E_opt_fsp,E_opt_nd,E_opt_moments,E_opt_smoments)

main()
