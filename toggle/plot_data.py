import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('~/mpl_styles/light')
import toggle
import vis


def plot_data_2d(data_vector,ax):
    '''
    make a contour plot of the data in 2D 
    '''
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')
    MX = 70
    MY = 100
    ntimes = 5
    array_data = data_vector.reshape(MX+1,MY+1,ntimes)
    for i in range(ntimes-1):
        Zm = np.ma.masked_where(array_data[:,:,i+1] == 0 ,array_data[:,:,i+1] )
        ax[i].contourf(Zm,cmap=cmap)
        ax[i].set_xlim([0,75])
        ax[i].set_ylim([0,50])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    return ax

def plot_solution_2d(array_data, ax):
    '''
    make a contour plot of the data in 2D 
    '''
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')
    ntimes = 5
    for i in range(ntimes-1):
        Zm = np.ma.masked_where(array_data[:,:,i+1] < 2e-4 ,array_data[:,:,i+1] )
        ax[i].contourf(Zm,cmap=cmap)
        ax[i].set_xlim([0,75])
        ax[i].set_ylim([0,50])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    return ax
 
def load_data(rep_id,baseline=True):
    '''
    load the data into a list of vectors for easier use. 
    ''' 
    if baseline: 
        fname = 'data/toggle_ssa_0.0_baseline.txt'
        data = np.loadtxt(fname)
    else:
        data = []
        n_exp = 3
        n_rep = 2
        for i in range(n_exp):
            fname = 'data/toggle_ssa_pdf'+'__experiment_'+str(i)+'_replicate_'+str(rep_id)+'.txt'
            data.append(np.loadtxt(fname))
    return data

def plot_all_2d_data():
    '''
    Make one big plot that plots all of the data. 
    '''
    n_rep = 2
    n_exp = 3
    n_times = 5
    # loop over the replicates.  
    for i in range(n_rep):
        # load data
        data = load_data(i,baseline=False)
        f,ax = plt.subplots(n_exp,n_times-1,figsize=(10,7))
        for j in range(n_exp):
            plot_data_2d(data[j],f,ax[j][:])
        ax[2,0].set_ylabel('LacI')
        ax[2,0].set_xlabel(r'$\lambda$cI')
        plt.tight_layout()
        f.savefig('../../figures/toggle/toggle_joint_rep_{0}.pdf'.format(i))

def plot_baseline_data_2d(ax=None,save=False):
    '''
    load and plot baseline data 
    '''
    fname = 'data/toggle_ssa_0.0_baseline.txt'
    data = np.loadtxt(fname)
    if ax==None:
        f,ax = plt.subplots(1,4,figsize=(10,7/3))

    plot_data_2d(data,ax)
    ax[0].set_ylabel('LacI')
    ax[0].set_xlabel(r'$\lambda$cI')
    #plt.tight_layout()
    if save: 
        f.savefig('../../figures/toggle/toggle_joint_baseline.pdf')
    return ax

def plot_marginal_dist(data,species_id,ax,colormap='viridis'):
    '''
    plot the marginal distributions for a given 
    experiment and species ID of the data. 
    '''
    # reshape the data
    MX = 70
    MY = 100
    ntimes = 5
    array_data = data.reshape(MX+1,MY+1,ntimes)
    marginal_data = np.sum(array_data,axis=species_id)
    nx = marginal_data.shape[0]
    cmap = matplotlib.cm.get_cmap(colormap)
    cmap_inds = np.linspace(0,.9,ntimes-1)
    for i in range(1,ntimes):
        ax.fill_between(np.arange(nx),np.zeros(nx),marginal_data[:,i], facecolor=cmap(cmap_inds[i-1]), alpha=.4)
    
    ax.set_xlim([0,np.max(np.nonzero(marginal_data))])
    return ax

def plot_marginal_dist_3d(data,species_id,ax,colormap='viridis',add_model_solutions=False,params=None):
    '''
    plot the marginal distributions for a given 
    experiment and species ID of the data. 
    '''
    # reshape the data
    MX = 70
    MY = 100
    ntimes = 5
    array_data = data.reshape(MX+1,MY+1,ntimes)
    marginal_data = np.sum(array_data,axis=species_id)/5000
    nx = marginal_data.shape[0]
    cmap = matplotlib.cm.get_cmap(colormap)
    cmap_inds = np.linspace(0,.9,ntimes-1)
    colors = []
    for i in range(ntimes-1):
        colors.append(cmap(cmap_inds[i]))
    xs_tmp = np.arange(nx)
    verts = [] 
    zs = np.arange(ntimes-1)
    zs = np.array([ 1,2,4,8]) 
    # add model solutions
    if add_model_solutions:
        best_fit = get_best_solutions(params)
        marginal_data_fsp = np.sum(best_fit,axis=species_id)
        for i in range(1,ntimes):
            znew = np.repeat(zs[i-1],len(xs_tmp))
            xs =xs_tmp 
            ys = marginal_data_fsp[:,i]
            ax.plot(xs,ys,zs=znew,zdir='y',color=colors[i-1],linewidth=2,alpha=.7) 
    # add data plots
    for i in range(1,ntimes):
        xs = np.concatenate([[xs_tmp[0]],xs_tmp,[xs_tmp[-1]]])
        ys = np.concatenate([[0],marginal_data[:,i],[0]])
        verts.append(list(zip(xs,ys))) 
    poly = PolyCollection(verts,facecolors=colors)
    poly.set_alpha(0.7)
    ax.add_collection3d(poly,zs=zs,zdir='y')
    ax.set_xlim3d(0,np.max(np.nonzero(marginal_data)))
    ax.set_ylim3d(-1,10)
    ax.set_zlim3d(0,np.max(marginal_data[:,1:]))
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(30,-134)
    return ax

def get_best_solutions(params=None):
    ''' 
    get the best solutions for a model. 
    '''
    if params is None:
        # load parameters and likelihoods
        likelihood_chain = np.loadtxt('out/likelihood_chain_pdf_0_baseline.txt')
        parameter_chain = np.loadtxt('out/parameter_chain_pdf_0_baseline.txt')
        params = parameter_chain[np.argmax(likelihood_chain),:]
    tvec = 3600*np.array([0.0,1.0,2.0,4.0,8.0])
    fsp = toggle.ToggleFSP(70,100,params,10)
    fsp.tvec = tvec
    fsp.solve()
    fsp.p = fsp.p.reshape((71,101,5))
    return fsp.p
  
def test_marginal(save=False):
    data = load_data(1,baseline=True)
#    f,ax = plt.subplots(1,2,figsize=(6,3))
#    ax[0] = plot_marginal_dist(data,0,ax[0])
#    ax[1] = plot_marginal_dist(data,1,ax[1])
#    plt.show()

    fig = plt.figure()
    #ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,1,1,projection='3d')
    #plot_marginal_dist_3d(data,0,ax1,colormap='viridis')
    specID = 0
    plot_marginal_dist_3d(data,specID,ax2,colormap='viridis',add_model_solutions=True)
    if save:
        fig.savefig('../../figures/toggle/toggle_3d_species_{0}.pdf'.format(specID))
    plt.show() 

def plot_mcmc_baseline():
    # load parameters and likelihoods
    likelihood_chain = np.loadtxt('out/likelihood_chain_pdf_0_baseline.txt')
    parameter_chain = np.loadtxt('out/parameter_chain_pdf_0_baseline.txt')
    
    free_parameters = np.array([0,1,2,3,4,5,9]) 
    pchain_free = parameter_chain[::50,free_parameters]/np.mean(parameter_chain[::50,free_parameters],axis=0)

    # get the best parameters
    best_pars = parameter_chain[np.argmax(likelihood_chain),free_parameters]/np.mean(parameter_chain[::50,free_parameters],axis=0) 

    # make a scatter plot 
    f,ax = plt.subplots(len(free_parameters),len(free_parameters),figsize=(12,9))
    # do the plotting
    color = (163/255,92/255,158/255)
    for i in range(len(free_parameters)):
        for j in range(i,len(free_parameters)):
            if i==j:
                ax[i,j].hist(pchain_free[:,i],bins=30,color='gray')
                ax[i,j].tick_params(axis='y',labelsize=12)            
                ax[i,j].tick_params(axis='x',labelsize=12,labelcolor='gray')            
                ax[i,j].tick_params(axis='y',labelsize=12,labelcolor='gray')            
                ax[i,j].set_ylim([0,10])
            else:
                ax[i,j].yaxis.tick_right()
                ax[i,j].scatter(pchain_free[:,j],pchain_free[:,i],c=color,alpha=.5)
                # plot the best parameters
                ax[i,j].scatter(best_pars[j],best_pars[i],c='k')

                # other stuff
                ax[i,j].set_ylim([.8,1.2])
                ax[i,j].set_yticks([.9,1,1.1])
             
    # fix up axes
    for i in range(len(free_parameters)):
        for j in range(len(free_parameters)):
            ax[i,j].set_xlim([.8,1.2])
            ax[i,j].set_xticks([.9,1,1.1])
            if i>j:
                ax[i,j].axis('off')
                ax[i,j].set_xticklabels([])
                ax[i,j].set_yticklabels([])
            elif i!=j:
                ax[i,j].set_ylim([.8,1.2])
                if j==len(free_parameters)-1:
                    ax[i,j].set_xticklabels([])
                    ax[i,j].tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off',labelsize=12)            
                    ax[i,j].tick_params(axis='y',labelsize=12,labelcolor=color)            
                else:
                    ax[i,j].set_xticklabels([])
                    ax[i,j].set_yticklabels([])
                    ax[i,j].tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off',labelsize=12)            
                    ax[i,j].tick_params(axis='y', which='both', left='off', right='off',labelsize=8)            
    f.savefig('../../figures/toggle/baseline_scatter.pdf')
    plt.show()

def plot_baseline_fit_joint():
    '''
    plot the joint distribution of LacI and lambdacI.
    with the model fits.
    '''
    ntimes = 4
    pmf = get_best_solutions()
    f,ax = plt.subplots(2,ntimes,figsize=(10,14/3))
    ax[0][:] = plot_solution_2d(pmf,ax[0][:])
    ax[1][:] = plot_baseline_data_2d(ax[1][:])
    plt.tight_layout()
    f.savefig('../../figures/toggle/baseline_joint_fit.pdf')
    
    
#if __name__=='__main__':
#    plot_all_2d_data()            
#    plot_baseline_data_2d()
#    plt.show()
#    test_marginal(save=True)
#    plot_baseline_fit_joint()
#    plot_mcmc_baseline()
