import matplotlib
import matplotlib.pyplot as plt
plt.style.use('~/mpl_styles/light')
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal,chi2

def plot_sensitivities(model):
    pass 

def plot_bar(fim1,fim2,npars = 4,btype='diag',ax=None,pnames=(r'$\theta_1$',r'$\theta_2$')):
    '''
    Make a bar plot .
    '''    
    if not ax:
        f,ax = plt.subplots()
    else:
        pass 

    ind = np.arange(npars)
    width = 0.35
    if btype is 'diag':
        # Get the diags. 
        val1 = np.diag(fim1)
        val2 = np.diag(fim2)
        ax.set_xticks(ind+width)
        ax.set_xticklabels(pnames,size=20)
    else:    
        # Get the eigs.
        val1,vec1 = np.linalg.eig(fim1)
        val2,vec2 = np.linalg.eig(fim2)  
        ax.set_xticks(ind+width)
        ax.set_xticklabels((r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$',r'$\lambda_4$'),size=20)

    
    bar1 = ax.bar(ind,val1,width,color='c')
    bar2 = ax.bar(ind+width,val2,width,color='r')
    
    ax.set_yscale('log')
    
    return ax

def plot_fim_ellipse(ax,FIM,params,color='c',pid=[0,1],pname=[r'$k_{on}$',r'$k_{off}$']):
    '''
    A function to analyse FIM. 
    '''
    pid = np.array(pid)
    # Get the FIM for your young guns
    FIM = FIM[pid,:]
    FIM = FIM[:,pid]    

    # Get the eigs.
    vals,vecs = np.linalg.eig(FIM)
    
    # Get the params of interest. 
    params = np.array(params)
    params = params[pid]

    print('*****PARAMETER IDS*****')
    print(pname)
    print('*****Eigenvalues: ')
    print(vals)
    print('*****Eigenvectors: ' )
    print(vecs)
    print('******* Perpendicular Check: ')
    print(np.dot(vecs[:,0],vecs[:,1]))

    # define unit vector. 
    a = np.array([0.0,1.0])

    # find theta, the angle between x axis and principle direction. 
    theta1 = (360.0/(2*np.pi))*np.arccos( np.dot(a,vecs[:,0] / np.linalg.norm(vecs[:,0])))
    #theta2 = (360/(2*np.pi))*np.arccos( np.dot(a,vecs[:,1] / np.linalg.norm(vecs[:,1])))
    theta2 = (360/(2*np.pi))*np.arccos( np.dot(a,-1*vecs[:,0] / np.linalg.norm(vecs[:,0])))
    
  #  theta = np.max([theta1,theta2])-90
    theta = (360/(2*np.pi))*np.arctan(vecs[1,0]/vecs[0,0])
    print('Theta: %f' %theta)

    # make the ellipse 
    w = 1.0/np.sqrt(vals[0]*5.991)
    h = 1.0/np.sqrt(vals[1]*5.991)
    e = Ellipse(xy = params ,width =  w, height = h,angle=theta ) 
    
#    f,ax = plt.subplots()
    ax.add_artist(e)  
    e.set_clip_box(ax.bbox)
    e.set_alpha(.75)
    e.set_facecolor((color))
    
    # add a dot for the correct parameters. 
#    ax.scatter(params[0],params[1],c='k',s=20)
    
    # test angles and things.
    x1 = params[0]+1.0/np.sqrt(vals[0])*np.array([-1.0,1.0])*vecs[0,0]
    y1 = params[1]+1.0/np.sqrt(vals[0])*np.array([-1.0,1.0])*vecs[1,0]

    x2 = params[0]+1.0/np.sqrt(vals[1])*np.array([-1.0,1.0])*vecs[0,1]
    y2 = params[1]+1.0/np.sqrt(vals[1])*np.array([-1.0,1.0])*vecs[1,1]

    ax.plot(x1,y1,'k--')
    ax.plot(x2,y2,'k--')

    max_w = np.max([w,h])
    ax.set_xlim([params[0]-max_w*1.01,params[0]+max_w*1.01])
    ax.set_ylim([params[1]-max_w*1.01,params[1]+max_w*1.01])
    
    ax.set_xlabel(pname[0],size=20)
    ax.set_ylabel(pname[1],size=20)
    
#    ax.set_xscale('log')
#    ax.set_yscale('log')
    return ax

def plot_distributions(data,plot_type = 'pdf',axarr=None):
    '''
    plot the distributions over time. If type is pdf, data will just have
    each pdf as a function of time. Otherwise, data will be a tuple, 
    (means,covariances) and a normal distribution is assumed.
    '''
    
    if plot_type is 'pdf':
        pass
        
    elif plot_type is 'moments': 
        # make distribution
        nspec,ntimes = data[0].shape
        dist = np.zeros((100,ntimes))
        y = np.linspace(0,3,20)
        x,y  = np.meshgrid(np.linspace(0,100,501),y)

        # create position matrices. 
        pos = np.empty(x.shape + (2,))
        pos[:,:,0] = x; pos[:,:,1] = y; 

        # generate a list of rvs at each time point. 
        rvs = []
        full_dist = np.empty((len(y)*501,ntimes))
        for i in range(ntimes):
            rvs.append(multivariate_normal(data[0][:,i],data[1][:,:,i]))
            full_dist[:,i] = np.ravel(rvs[i].pdf(pos))
              
        # get the marginal distribution. 
        ysize = 20
        for i in range(100):
            dist[i,:] = np.sum(full_dist[ysize*i:ysize*i+ysize,:],axis=0)
        return dist,full_dist  

def plot_means_vars(moments):
    '''
    plot m/v over time. 
    '''
    f,ax = plt.subplots()
    ax.plot(moments.tvec,moments.mean[0,:],'m',linewidth=3)
    ax.fill_between(moments.tvec,moments.mean[0,:]+np.sqrt(moments.covariances[0,0,:]),moments.mean[0,:]-np.sqrt(moments.covariances[0,0,:]),color='k',alpha=.2)
    f.show()
    return ax

def plot_I_results():
    fsp_res = np.loadtxt('fsp_results_kr')
    moments_res = np.loadtxt('moments_results_kr')
    f,ax = plt.subplots(2,1)
    ax[0].plot(fsp_res[0,:],fsp_res[1,:],'c')
    ax[0].plot(fsp_res[0,:],moments_res[1,:],'r')
    ax[1].plot(fsp_res[0,:],moments_res[2,:],'r')
    ax[1].plot(fsp_res[0,:],fsp_res[2,:],'c')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.show()

def plot_dist(moments,fsp):
    # frst, get the normal distribution over the fsp range. 
    x =np.arange(fsp.N)
    count = 0
    var = moments.covariances.ravel()
    print(var)
    mean = moments.mean[1,:]
    f,ax = plt.subplots(2,4)
    for i in range(2):
        for j in range(4):
            ax[i,j].plot(x,fsp.p[:,count],'g',linewidth=2)
            norm = (1.0/(np.sqrt(2*np.pi*var[count])))*np.exp(-.5 * (x-mean[count])**2 / var[count])
            print(np.sum(np.abs(norm-fsp.p[:,count].ravel())))
            ax[i,j].plot(x,norm,'c--',linewidth=2)
            count +=1
    f.show()


def plot_dist_2(moments,fsp):
    # first, get the normal distribution over the fsp range. 
    x =np.arange(fsp.N)
    #var = moments.covariances[1,1,1:]
    var = moments.covariances.ravel()
    print(moments.covariances.shape)
    #mean = moments.mean[1,1:]
    print(moments.mean.shape)
    mean = moments.mean
    f,ax = plt.subplots(1,4,figsize=(14,5))
    for i in range(4):
        ax[i].plot(x,fsp.marginal[:,i+1],'g',linewidth=4)
        norm = (1.0/(np.sqrt(2*np.pi*var[i+1])))*np.exp(-.5 * (x-mean[i+1])**2 / var[i+1])
        ax[i].plot(x,norm,'c--',linewidth=2)
        #ax[i].set_xlim([-1,fsp.N-150])
        #ax[i].set_xlim([-1,fsp.N])
        ax[i].set_xlim([-1,10])
        ax[i].set_ylim([0,ax[i].get_ylim()[1]])
    plt.tight_layout()
    f.savefig('../../figures/test_fsamp_distributions.pdf')

def plot_single_dist(moments,fsp,name=None):
    # frst, get the normal distribution over the fsp range. 
    x = np.arange(fsp.N)
    var = moments.covariances.ravel()
    #mean = moments.mean[0,:]
    mean = moments.mean.ravel()
    count=-1
    f,ax = plt.subplots(figsize=(5,4))
    ax.plot(x,fsp.p[:,count],'dodgerblue',linewidth=4)
    norm = (1.0/(np.sqrt(2*np.pi*var[count])))*np.exp(-.5 * (x-mean[count])**2 / var[count])
    #print(np.sum(np.abs(norm-fsp.p[:,count].ravel())))
    ax.plot(x,norm,'mediumorchid',linewidth=4,linestyle='--')
    ax.set_xlim([0,150])
    ax.set_ylim([0, np.max(norm)+.05*np.max(norm)])
#    if name == 40:
#        ax.set_ylim([0,.05])
#    else:
#        ax.set_ylim([0,0.035])
    
    ax.set_xticks([0,150])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('# RNA')
    ax.set_ylabel('Probability')
    f.tight_layout()
    f.show()
#    if name is not None:
#    else:
#        f.show()

def plot_exp_design():
    d_fsp_1 = np.loadtxt('out/D_opt_fsamp_fsp0.1.txt')
    d_fsp_2 = np.loadtxt('out/D_opt_fsamp_fsp100.0.txt')

    d_moments_1 = np.loadtxt('out/D_opt_fsamp_moments0.1.txt')
    d_moments_2 = np.loadtxt('out/D_opt_fsamp_moments100.0.txt')
    
    fsp = [d_fsp_1,d_fsp_2]
    moments = [d_moments_1,d_moments_2]
    fs = np.linspace(.1,5,50) 
    #xlabs = np.linspace(1,5,5)
    xlabs = np.array([.5,1,3,5])
    kon = [.1,100.0]
    for i in range(2): 
        f,ax = plt.subplots()
        ax.plot(fs,fsp[i]/np.max(fsp[i]),'c',linewidth=8)
        ax.plot(fs,moments[i]/np.max(moments[i]),'m',linewidth=8)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-3,11])
        ax.set_xticks(xlabs)
        ax.set_xticklabels(xlabs)
        ax.set_xlim([.25,5])
        if i == 1:  
            #xlabs = np.linspace(1,3,3) 
            xlabs = np.array([.5,1,3])
            ax.set_xticks(xlabs)
            ax.set_xticklabels(xlabs)
            ax.set_xlim([.25,3.0])
            
        ax.set_xlabel(r'$\Delta$',size=30)
        ax.set_ylabel('D-Optimality',size=30)
        ax.tick_params(labelsize=20)
    
        plt.tight_layout()
        f.savefig('../../figures/optimality_'+str(kon[i])+'.pdf')
        plt.close() 

def plot_likelihoods(lhood,ax=None,save=False):
    if ax is None:
        f,ax = plt.subplots()
    ax.plot(lhood,linewidth=3)
    ax.set_xlabel('Chain Length',size=20)
    ax.set_ylabel(r'$\log \ell (\mathbf{d}|\theta)$',size=28)
    if save:
        f.savefig('../../figures/likelihood_plot.pdf')
    return ax

def scatter_par_chain(parameter_chain,indices,ax=None,color='k'):
    '''
    draw a scatter plot
    '''
    if ax is None:
        f,ax = plt.subplots()
    data = parameter_chain[:,indices]
    ax.scatter(data[::25,0],data[::25,1],color=color,alpha=.1)
    return ax 

def plot_conf_ellipse(mu,cov,ax=None,color='m',crosshairs=False,linestyle='-',linewidth=1,ci=.95):
    '''
    plot ellipse given mu and cov 
    '''
        
    if ax is None: 
        f,ax = plt.subplots()
    if crosshairs:
        ax = plot_crosshairs(mu,cov,ax,color='k')
    vals,vecs = np.linalg.eig(cov)
    theta = (360/(2*np.pi))*np.arctan(vecs[1,0]/vecs[0,0])
    # able to change CI now. 
    scale = chi2.ppf(ci,2)
    w = np.sqrt(vals[0]*scale)*2
    h = np.sqrt(vals[1]*scale)*2
    e = Ellipse(xy = mu.ravel() ,width =  w, height = h,angle=theta,linewidth=linewidth ) 
    
    ax.add_artist(e)  
    e.set_clip_box(ax.bbox)
#    e.set_alpha(.75)
    e.set_edgecolor((color))
    e.set_linestyle((linestyle))
    e.set_facecolor(('none'))
    return ax

def plot_crosshairs(mu,cov,ax=None,color='k'):
    '''
    add crosshairs. 
    '''
    if ax is None:
        f,ax = plt.subplots()
    a = np.linspace(mu[0]-50*np.sqrt(cov[0,0]),mu[0]+50*np.sqrt(cov[0,0]),30)
    b = np.linspace(mu[1]-50*np.sqrt(cov[1,1]),mu[1]+50*np.sqrt(cov[1,1]),30)
    ax.plot(a,np.repeat(mu[1],30),color+'--',linewidth=1)
    ax.plot(np.repeat(mu[0],30),b,color+'--',linewidth=1)
    return ax

def plot_toggle_distributions(fsp,ax=None,linestyles=None):
    '''
    Plot LacI distributions at all times in the 
    tvec. 
    '''
    # get marginal distribution of LacI 
    marginal = np.sum(fsp.p,axis=0)
    nt = fsp.p.shape[2] 
    colors = matplotlib.cm.get_cmap('plasma')
    if linestyles:
        c_args = ['k']*nt
    else:
        c_args = np.linspace(0,1,nt)
        linestyles = ['-']*nt
    for i in range(nt):
        if linestyles:
            c = 'k'
        else:
            c = colors(c_args[i])
        ax.plot(marginal[:,i],linewidth=3,color=c,linestyle=linestyles[i])
        ax.set_xlabel(r'$\xi_2$',size=20)
        ax.set_ylabel(r'$p(\mathbf{x})$',size=25)
        ax.set_xlim([0,80])
        ax.set_ylim([0,.16])
    return ax
        
def plot_bar(crbs,ax,colors,split=2):
    '''
    make a bar plot to compare the uncertainty from preliminary and optimal experiments.
    '''
    us = []
    for i in range(len(crbs)):
        us.append(crbs[i].trace())

    x = np.arange(len(crbs))
    w = .66
    ax.bar(x[:split],us[:split],width=w,color=colors[:split],label='Preliminary Experiment')
    ax.bar(x[split:],us[split:],width=w,color=colors[split:],label='Best Experiment')
    ax.set_yscale('log')
    ax.set_xticks((np.array(x)+w)[::2])
    ax.set_xticklabels([r'$pdf$',r'$moments$'],size=25)
    ax.set_ylabel(r'$tr(FIM^{-1})$',size=25)
   # ax.legend(loc=1)
    return ax

def scale_axarr(axarr):
    '''
    takes an array of axes in grid and scales such that rows and columns
    are on the same scale   
    ''' 
    n_rows,n_cols = axarr.shape()
    max_ys = np.empty(3)
    max_xs = np.empty(3)
    for i in range(n_rows):
        for j in range(n_cols):
            dx = axarr[i,j].get_xlim()[1]-axarr[i,j].get_xlim()[0]
            dy = axarr[i,j].get_ylim()[1]-axarr[i,j].get_ylim()[0]
            max_ys[i] = np.max([dy,max_ys[i]])
            max_xs[j] = np.max([dx,max_xs[j]])
    return 
             


