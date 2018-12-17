import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('~/mpl_styles/light')


def plot_data_2d(data_vector,f,ax):
    '''
    make a contour plot of the data in 2D 
    '''
    MX = 70
    MY = 100
    ntimes = 5
    array_data = data_vector.reshape(MX+1,MY+1,ntimes)
    for i in range(ntimes):
        ax[i].contourf(array_data[:,:,i])
    return f,ax
       
def load_data(rep_id):
    '''
    load the data into a list of vectors for easier use. 
    ''' 
    all_data = []
    n_exp = 3
    n_rep = 2
    for i in range(n_exp):
        fname = 'toggle_ssa_pdf'+'__experiment_'+str(i)+'_replicate_'+str(rep_id)+'.txt'
        all_data.append(np.loadtxt(fname))
    return all_data

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
        data = load_data(i)
        f,ax = plt.subplots(n_exp,n_times,figsize=(10,7))
        for j in range(n_exp):
            plot_data_2d(data[j],f,ax[j][:])
        f.savefig('../../figures/toggle/toggle_joint_rep_{0}.pdf'.format(i))

def plot_baseline_data_2d():
    '''
    load and plot baseline data 
    '''
    fname = 'toggle_ssa_0.0_baseline.txt'
    data = np.loadtxt(fname)
    f,ax = plt.subplots(1,5,figsize=(10,7/3))
    plot_data_2d(data,f,ax)
    f.savefig('../../figures/toggle/toggle_joint_baseline.pdf')

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

def plot_marginal_dist_3d(data,species_id,ax,colormap='viridis'):
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
    colors = []
    for i in range(ntimes-1):
        colors.append(cmap(cmap_inds[i]))
    xs_tmp = np.arange(nx)
    verts = [] 
    zs = np.arange(ntimes-1)
    for i in range(1,ntimes):
        xs = np.concatenate([[xs_tmp[0]],xs_tmp,[xs_tmp[-1]]])
        ys = np.concatenate([[0],marginal_data[:,i],[0]])
        verts.append(list(zip(xs,ys))) 
    poly = PolyCollection(verts,facecolors=colors)
    poly.set_alpha(0.7)
    ax.add_collection3d(poly,zs=zs,zdir='y')
    ax.set_xlim3d(0,np.max(np.nonzero(marginal_data)))
    ax.set_ylim3d(-1,4)
    ax.set_zlim3d(0,np.max(marginal_data[:,1:]))
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    return ax

def test_marginal():
    data = load_data(1)[0]
#    f,ax = plt.subplots(1,2,figsize=(6,3))
#    ax[0] = plot_marginal_dist(data,0,ax[0])
#    ax[1] = plot_marginal_dist(data,1,ax[1])
#    plt.show()

    fig = plt.figure()
    #ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,1,1,projection='3d')
    #plot_marginal_dist_3d(data,0,ax1,colormap='viridis')
    plot_marginal_dist_3d(data,1,ax2,colormap='viridis')
    plt.show() 

       
    
    
    

if __name__=='__main__':
#    plot_all_2d_data()            
#    plot_baseline_data_2d()
#    plt.show()
    test_marginal()
