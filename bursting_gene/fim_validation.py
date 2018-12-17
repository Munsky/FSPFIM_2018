import numpy as np
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import bursting_gene as bg
import vis
import matplotlib
from scipy.stats import multivariate_normal,chi2

cmap_fsp = matplotlib.cm.get_cmap('Oranges')
cmap_lna = matplotlib.cm.get_cmap('Purples')
cmap_sm = matplotlib.cm.get_cmap('Reds')

'''
This is a code to validate the FIM, using various methods. 
'''

def fdist_to_raw(data):
    '''
    convert frequency distribution data to "raw" (unbinned) data
    '''
    n,nt = data.shape
    raw_data = np.zeros((nt,int(np.sum(data,axis=0)[0])))
    for t in range(nt):
        count = 0
        for v in range(n):
            for f in range(int(data[v,t])):
                raw_data[t,count] = v
                count += 1
    sample_means = np.mean(raw_data,axis=1)
    sample_var = np.var(raw_data,axis=1,ddof=1)
    return raw_data, sample_means,sample_var

def test_if_in(M,p,param_guess,ci):
    '''
    test if a particular parameter set is within a given CI. 
    '''
    # rotate
    p_rot = np.linalg.solve(M,p-param_guess[[1,2]])
    # rescale 
    r = chi2.ppf(ci,2)
    return np.sum(p_rot**2)<r

def true_in_mle(p_mle,p_true,atype,ci=.95):
    '''
    Function to check how many times the true parameters are contained 
    in the FIM about the MLE estimate.
    '''
    ptmp = np.copy(p_true)
    ptmp[[1,2]] = p_mle
    p_mle = ptmp
    tf = 10;
    if atype=='fsp':
        model = bg.BurstingGeneFSP(p_mle)
        # specify parameters that can be changed.
        model.free_parameters = [1,2]
        model.N = 200
        model.tf = tf
        model.Nc = 1000.0
        model.get_FIM(tstart=1,rna_only=True)
        CV = np.linalg.inv(model.FIM)
    elif atype=='moments':     
        moments = bg.BurstingGeneMoments(p_mle)
        # specify parameters that can be changed.
        #moments.free_parameters = [1,2]
        moments.ptimes = 10
        moments.tf = tf
        moments.Nc = 1000.0
        moments.N = 2
        #moments.obvs=[1]
        moments.get_FIM(order=2,tstart=1)
        CV = np.linalg.inv(moments.FIM)
    elif atype=='smoments':
        # get the mopments-FIM
        smoments = bg.BurstingGeneMoments(p_mle)
        # specify parameters that can be changed.
        #smoments.free_parameters = [1,2]
        smoments.ptimes = 10
        smoments.tf = tf
        smoments.Nc = 1000.0
        smoments.N = 2
        smoments.obvs=[1]
        smoments.get_sample_FIM(tstart=1,rna_only=True)
        CV = np.linalg.inv(smoments.FIM)

    vals,vecs = np.linalg.eig(CV)
    M = np.dot(vecs.dot(np.diag(np.sqrt(vals))),vecs.T)
    return test_if_in(M,p_true[[1,2]],p_mle,ci)
    
def run_test_in_mle():
    '''
    This function tests how many times the inverse of the FIM evaluated about the 
    MLE estimates contain the true parameters. 
    '''
    # First, generate the "true" fim about the true parameters
    kr = 100.0
    g = 1.0
    kon = .1 
    koff = .3 
    p_true = np.array([kon,koff,kr,g])

    n_params = 200
    all_params = np.zeros((n_params,2))
    all_params_moments = np.zeros((n_params,2))
    all_params_smoments = []
    p_in_smoments = []
    p_in_moments = []
    p_in_fsp= []
    for i in range(n_params):
        all_params[i,:] = 10**np.loadtxt('out/nd_analysis/params_pdf_0.1_{0}.txt'.format(i))
        all_params_moments[i,:] = 10**np.loadtxt('out/nd_analysis/params_moments_0.1_{0}.txt'.format(i))
        try:
            all_params_smoments.append(10**np.loadtxt('out/nd_analysis/params_smoments_0.1_{0}.txt'.format(i+200)))
            if true_in_mle(all_params_smoments[-1],p_true,'smoments',ci=.95):
                p_in_smoments.append(np.array(all_params_smoments[-1]))
        except:
            pass
        # check in this point is in the ellipse for the given analysis type
        if true_in_mle(all_params[i,:],p_true,'fsp',ci=.95):
            p_in_fsp.append(all_params[i,:])
   
        if true_in_mle(all_params_moments[i,:],p_true,'moments',ci=.95):
            p_in_moments.append(all_params_moments[i,:])
    
    print('Number of times true parameters inside FSP FIM_mle {0}'.format(len(p_in_fsp)))
    print('Number of times true parameters inside LNA FIM_mle {0}'.format(len(p_in_moments)))
    print('Number of times true parameters inside SM FIM_mle {0}'.format(len(p_in_smoments)))
    print(len(all_params_smoments))

def run_test_about_true():
    '''
    This function tests how many times the inverse of the FIM evaluated about the 
    true parameters contain the MLE estimated. 
    '''
    use_smoments=True
    # First, generate the "true" fim about the true parameters
    kr = 100.0
    g = 1.0
    kon = .1 
    koff = .3 
    param_guess = np.array([kon,koff,kr,g])
    tf = 10.0
    # Get the covariance from inverse of FIM.
    model = bg.BurstingGeneFSP(param_guess)
    # specify parameters that can be changed.
    model.free_parameters = [1,2]
    model.N = 200
    model.tf = tf
    model.Nc = 1000.0
    model.get_FIM(tstart=1,rna_only=True)
    CV = np.linalg.inv(model.FIM)
    
    # get the mopments-FIM
    moments = bg.BurstingGeneMoments(param_guess)
    moments.ptimes = 10
    moments.tf = tf
    moments.Nc = 1000.0
    moments.N = 2
    moments.get_FIM(order=2,tstart=1)
    CVM = np.linalg.inv(moments.FIM)
    
    if use_smoments:
        # get the mopments-FIM
        smoments = bg.BurstingGeneMoments(param_guess)
        # specify parameters that can be changed.
        smoments.ptimes = 10
        smoments.tf = tf
        smoments.Nc = 1000.0
        smoments.N = 2
        smoments.obvs=[1]
        smoments.get_sample_FIM(tstart=1,rna_only=True)
        CVS = np.linalg.inv(smoments.FIM)
    
    # next load up all of the MLE parameters from the search
    n_params = 200
    all_params = np.zeros((n_params,2))
    all_params_moments = np.zeros((n_params,2))
    if use_smoments:
        #all_params_smoments = np.zeros((n_params,2))
        all_params_smoments = []
    vals,vecs = np.linalg.eig(CV)
    M1 = np.dot(vecs.dot(np.diag(np.sqrt(vals))),vecs.T)
    vals,vecs = np.linalg.eig(CVM)
    M2 = np.dot(vecs.dot(np.diag(np.sqrt(vals))),vecs.T)
    vals,vecs = np.linalg.eig(CVS)
    M3 = np.dot(vecs.dot(np.diag(np.sqrt(vals))),vecs.T)
    p_in = []
    p_out = []
    p_out_moments = []
    p_out_smoments = []
    ci=.95
    for i in range(n_params):
        all_params[i,:] = 10**np.loadtxt('out/nd_analysis/params_pdf_0.1_{0}.txt'.format(i))
        all_params_moments[i,:] = 10**np.loadtxt('out/nd_analysis/params_moments_0.1_{0}.txt'.format(i))
        if use_smoments: 
            try:
                all_params_smoments.append(10**np.loadtxt('out/nd_analysis/params_smoments_0.1_{0}.txt'.format(i+200)))
                if not test_if_in(M3,np.array(all_params_smoments[-1]),param_guess,ci):
                    p_out_smoments.append(np.array(all_params_smoments[-1]))
            except:
                pass
        # check in this point is in the ellipse for the given analysis type
        if not test_if_in(M1,all_params[i,:],param_guess,ci):
            p_out.append(all_params[i,:])
    
        if not test_if_in(M2,all_params_moments[i,:],param_guess,ci):
            p_out_moments.append(all_params_moments[i,:])
    
    
    print('Number of FSP MLE estimates outside of boundary {0}'.format(len(p_out)))
    print('Number of LNA MLE estimates outside of boundary {0}'.format(len(p_out_moments)))
    print('Number of SM MLE estimates outside of boundary {0}'.format(len(p_out_smoments)))
    p_out = np.array(p_out)
    p_out_moments = np.array(p_out_moments)
    p_out_smoments = np.array(p_out_smoments)
    all_params_smoments = np.array(all_params_smoments)
    # plotting
    f,ax = plt.subplots(2,2,figsize=(7.5,6))
    c2 = (33/255,145/255,251/255)
    cis = np.arange(.55,.95,.1)[::-1]
    # plot the single FSP plot
    f2,ax2 = plt.subplots()
    ax2.scatter(all_params[:,0]/20.0,all_params[:,1]/20.0,c='k',alpha=.250)
    ax2 = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,np.cov(all_params.T)/400.0,ax=ax2,color='k',ci=.95,linewidth=2)
    ax2 = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,CV/400.0,ax=ax2,color='dodgerblue',ci=.95,linewidth=2)
    ax2.scatter(param_guess[1]/20.0,100.0/20.0,c='gold')
    ax2.set_xlim([.012, .018])
    f2.show()
    # Plot the FSP results
    ax[0,0].scatter(all_params[:,0]/20.0,all_params[:,1]/20.0,c='k',alpha=.250)
    #ax[0,0].scatter(p_out[:,0]/20.0,p_out[:,1]/20.0,c='r',alpha=.2)
    ax[0,0] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,np.cov(all_params.T)/400.0,ax=ax[0,0],color='k',ci=.95,linewidth=2)
    #for i in range(len(cis)):
    #    ax[0,0] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0]),CV,ax=ax[0,0],color=cmap_fsp(.1+i/(float(len(cis)))),ci=cis[i],linewidth=1)
    ax[0,0] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,CV/400.0,ax=ax[0,0],color='dodgerblue',ci=.95,linewidth=2)
    ax[0,0].scatter(param_guess[1]/20.0,100.0/20.0,c='gold')
    # Plot the LNA results
    ax[0,1].scatter(all_params_moments[:,0]/20.0,all_params_moments[:,1]/20.0,c='k',alpha=0.25)
    #ax[0,1].scatter(p_out_moments[:,0]/20.0,p_out_moments[:,1]/20.0,c='r',alpha=.2)
    ax[0,1] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,np.cov(all_params_moments.T)/400,ax=ax[0,1],color='k',ci=.95,linewidth=2)
    #for i in range(len(cis)):
    #    ax[0,1] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0]),CVM,ax=ax[0,1],color=cmap_lna(.1+i/float(len(cis))),ci=cis[i],linewidth=1)
    ax[0,1] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,CVM/400.0,ax=ax[0,1],color='mediumorchid',ci=.95,linewidth=2)
    ax[0,1].scatter(param_guess[1]/20.0,100.0/20.0,c='gold')
    # Plot the sample moment results
    ax[1,0].scatter(all_params_smoments[:,0]/20.0,all_params_smoments[:,1]/20.0,c='k',alpha=0.25)
    #ax[1,0].scatter(p_out_smoments[:,0]/20.0,p_out_smoments[:,1]/20.0,c='r',alpha=.2)
    ax[1,0] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,np.cov(all_params_smoments.T)/400.0,ax=ax[1,0],color='k',ci=.95,linewidth=2)
    ax[1,0].scatter(param_guess[1]/20.0,100.0/20.0,c='gold')
    #for i in range(len(cis)):
    #    ax[1,0] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0]),CVS,ax=ax[1,0],color=cmap_sm(.1+i/float(len(cis))),ci=cis[i],linewidth=1)
    ax[1,0] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,CVS/400.0,ax=ax[1,0],color='limegreen',ci=.95,linewidth=2)
    ax[0,1].scatter(param_guess[1]/20.0,100.0/20.0,c='gold')
    # Plot all three confidence ellipses
    ax[1,1] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,CVS/400.0,ax=ax[1,1],color='limegreen',ci=.95,linewidth=2)
    ax[1,1] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,CV/400.0,ax=ax[1,1],color='dodgerblue',ci=.95,linewidth=2)
    ax[1,1] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0])/20.0,CVM/400.0,ax=ax[1,1],color='mediumorchid',ci=.95,linewidth=2)
    # make all axis limits the same
    ax[0,1].set_xlim([0.01,0.02])
    ax[0,0].set_xlim(ax[0,1].get_xlim())
    ax[0,0].set_ylim(ax[0,1].get_ylim())
    ax[1,0].set_xlim(ax[0,1].get_xlim())
    ax[1,0].set_ylim(ax[0,1].get_ylim())
    ax[1,1].set_xlim(ax[0,1].get_xlim())
    ax[1,1].set_ylim(ax[0,1].get_ylim())
    for i in range(2):
        for j in range(2):
            ax[i,j].tick_params(labelsize=16)
    #ax[1] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0]),np.cov(all_params.T),ax=ax[1],color='r',ci=.99)
    #ax[1] = vis.plot_conf_ellipse(np.array([param_guess[1],100.0]),CV,ax=ax[1],color='coral',ci=.99)
    f.tight_layout()
    f.show()

