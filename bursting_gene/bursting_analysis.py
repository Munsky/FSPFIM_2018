import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
import bursting_gene as bg
reload(bg)
import vis
reload(vis)

'''
This is a script to generate Fig. 2 in the manuscript. 
It formulates the FSP-FIM for the bursting gene expression model 
under multiple "switching" rates between the active and inactive 
states. 
'''

# Set parameters, make storage variables. 
kr = 100.0          # RNA production rate 
g = 1.0             # RNA degradation rate
n = 51              # number of alphas. 
fsp_info = []
moments_info = []
smoments_info = []
fsp_info2 = []
moments_info2 = []
smoments_info2 = []
# time settings
tstart = 1
tf = 10 
# switching rates
alphas = np.logspace(-3,4,n)
kon = .1            # baseline on rate
koff = .3           # baseline off rate
# storage for the FIMs. 
moments_fims = np.empty((2,2,n))
fsp_fims = np.empty((2,2,n))
smoments_fims = np.empty((2,2,n))
# pick the indices  of alphas that we want to plot. 
dist_inds = [15,27,40]
for i in range(n):
    print 'Trial %d' %i
    # update parameters
    params = [alphas[i]*kon,alphas[i]*koff,kr,g]
    
    # get the fsp FIM criteria
    fsp = bg.BurstingGeneFSP(params)     
    fsp.Nc = 100 
    fsp.tf = tf
    fsp.N  = 200
    fsp.get_FIM(tstart,rna_only=True)
    fsp_info2.append(fsp.FIM.trace())
    fsp_info.append(np.linalg.det(fsp.FIM))
    fsp_fims[:,:,i] = fsp.FIM
    fsp.get_moments()

    # get the moments FIM criteria
    moments = bg.BurstingGeneMoments(params)    
    moments.Nc = 100
    moments.N = 2
    moments.tf = fsp.tf
    moments.get_FIM(order=2,tstart=tstart)
    moments_info2.append(moments.FIM.trace())
    moments_info.append(np.linalg.det(moments.FIM))
    moments_fims[:,:,i] = moments.FIM

    # get the sample moments FIM criterio
    smoments = bg.BurstingGeneMoments(params)
    smoments.tf = fsp.tf
    smoments.Nc = 100
    smoments.N = 2
    smoments.obvs=[1]
    smoments.get_sample_FIM(tstart=tstart,rna_only=True)
    smoments_info2.append(smoments.FIM.trace())
    smoments_info.append(np.linalg.det(smoments.FIM))
    smoments_fims[:,:,i] = smoments.FIM
    if i in dist_inds:
        print "plotting distributions for alpha = %f " %alphas[i]
        vis.plot_single_dist(moments,fsp,i)


# analysis and plotting. 
x1 = np.tile(alphas[dist_inds],3)
y1 = np.concatenate((np.array(moments_info)[dist_inds],np.array(fsp_info)[dist_inds],np.array(smoments_info)[dist_inds]))
f,ax = plt.subplots(figsize=(5,4))
f2,ax2 = plt.subplots(figsize=(5,4))
ax.plot(alphas,fsp_info,'dodgerblue',linewidth=5,zorder=1)
ax.plot(alphas,moments_info,'mediumorchid',linewidth=5,zorder=2)
ax.plot(alphas,smoments_info,'limegreen',linewidth=5,zorder=2)
ax2.plot(alphas,fsp_info2,'dodgerblue',linewidth=5,zorder=1)
ax2.plot(alphas,moments_info2,'mediumorchid',linewidth=5,zorder=2)
ax2.plot(alphas,smoments_info2,'limegreen',linewidth=5,zorder=2)
ax.scatter(x1,y1,color='k',s=40,zorder=3)
ax.set_xlabel(r'$\alpha$',size=30)
ax.set_ylabel(r'$det\{FIM\}$',size=30)
ax2.set_ylabel(r'$trace\{FIM\}$',size=30)
ax.set_xscale('log')
ax.set_yscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax.set_xlim([alphas[0],alphas[-1]])
ax2.set_xlim([alphas[0],alphas[-1]])
f.tight_layout()
f2.tight_layout()
f.show()
f2.show()

