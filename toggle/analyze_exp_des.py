import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import matplotlib
#cmap = matplotlib.cmap('Oranges')

'''
Generate the contour plots from figure 7 in the MS. 
Generate optimal designs for single, greedy, and dual experiments, 
as in table 2.  All data was generated remotely on the WM Keck Cluster 
at Colorado State University. 
'''
method = 'sampled'
# load and process FIMs. 
uv = np.arange(1,15,dtype=np.float64)
delta = np.arange(1,7,.5,dtype=np.float64)
alpha = np.arange(1,5,dtype=np.float64)
#alpha=1.0
all_E_opts = np.empty((len(alpha),len(delta),len(uv)))
all_D_opts = np.empty((len(alpha),len(delta),len(uv)))
print('Formulating experiment design {0}'.format(method))
keepers = np.arange(100)
for i in range(len(alpha)):
    for j in range(len(delta)):
        for k in range(len(uv)):
            if method=='sampled':
                sFIMs = np.loadtxt('out/all_fims_sampled_fim/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}_sampled.txt'.format(alpha[i],delta[j],uv[k]))
                sFIMs = sFIMs.reshape(7,7,100)
                tmpE = []
                tmpD = [] 
                for m in keepers:
                    vals,vecs = np.linalg.eig(sFIMs[:,:,m])
                    tmpE.append(np.min(vals))
                    tmpD.append(np.prod(vals))
                all_E_opts[i,j,k] = np.mean(tmpE)
                all_D_opts[i,j,k] = np.mean(tmpD)
            elif method == 'true_pars':
                FIM = np.loadtxt('out/all_exp_fims_tp/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}_true_pars.txt'.format(alpha[i],delta[j],uv[k]))
                vals,vecs = np.linalg.eig(FIM)
                all_E_opts[i,j,k] = np.min(vals)
                all_D_opts[i,j,k] = np.prod(vals)
            elif method == 'fmin_params':         
                FIM = np.loadtxt('out/all_exp_fims/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}.txt'.format(alpha[i],delta[j],uv[k]))
                vals,vecs = np.linalg.eig(FIM)
                all_E_opts[i,j,k] = np.min(vals)
                all_D_opts[i,j,k] = np.prod(vals)

            if (i==1) and (j==5) and (k==10):
                print('out/nd_design/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}.txt'.format(alpha[i],delta[j],uv[k]))
                print(np.prod(vals))

# plot a contourplot for a fixed alpha
f,ax = plt.subplots(1,4,figsize=(12,2.5))
f2,ax2 = plt.subplots(1,4,figsize=(12,2.5))
XX,YY = np.meshgrid(delta,uv)
low = np.min(all_E_opts)
hi = np.max(all_E_opts)
lowD = np.min(all_D_opts)
hiD = np.max(all_D_opts)
inds = np.unravel_index(np.argmax(all_E_opts),all_E_opts.shape)
inds_D = np.unravel_index(np.argmax(all_D_opts),all_D_opts.shape)
for i in range(4):
    # make contour plot of E-optimalities
    ax[i].set_title(r'$\alpha={0} hr$'.format(alpha[i]))
    im_ax = ax[i].contourf(uv,delta,all_E_opts[i,:,:],levels=np.linspace(np.floor(low*10)/10,np.ceil(hi*10)/10,20),cmap='Oranges')
    if i==inds[0]:
        ax[i].scatter(uv[inds[2]],delta[inds[1]],c='k')
    if i==2:
        ax[i].scatter(uv[5],delta[6],c='k',marker='^')

    # make contour plot of D-optimalities
    ax2[i].set_title(r'$\alpha={0} hr$'.format(alpha[i]))
    im_ax2 = ax2[i].contourf(uv,delta,all_D_opts[i,:,:],levels=np.linspace(lowD,hiD,20),cmap='Oranges')
    if i==inds_D[0]:
        ax2[i].scatter(uv[inds_D[2]],delta[inds_D[1]],c='k')
    if i==2:
        ax2[i].scatter(uv[5],delta[6],c='k',marker='^')
        print('off by 1 info: {0}'.format(all_D_opts[i,6,5]))

    ax[i].set_xlim([1,14])
# adjust plots, add colorbars, add figure titles.
f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.82, 0.1, 0.02, 0.8])
cbar = f.colorbar(im_ax, cax=cbar_ax,ticks=np.round(np.linspace(np.floor(low*10)/10,np.ceil(hi*10)/10,4),2)) 
#cbar = f.colorbar(im_ax, cax=cbar_ax) 
#cbar.ax.set_yticklabels(ticks=np.linspace(np.floor(low*10)/10,np.ceil(hi*10)/10,4))
f.suptitle('E-optimality')

f2.subplots_adjust(right=0.8)
cbar_ax2 = f2.add_axes([0.82, 0.1, 0.02, 0.8])
cbar2 = f2.colorbar(im_ax2, cax=cbar_ax2,ticks=np.linspace(10**np.ceil(np.log10(lowD)),10**np.floor(np.log10(hiD)),3)) 
cbar2.ax.set_yticklabels(np.linspace(np.ceil(np.log10(lowD)),np.floor(np.log10(hiD)),3))
f2.suptitle('D-optimality')

ax[0].set_ylabel(r'Sampling period')
ax[0].set_xlabel(r'UV')
ax2[0].set_ylabel(r'Sampling period')
ax2[0].set_xlabel(r'UV')
f.show()
f2.show()
print('Single experiment optimal design (D optimality): \n alpha: {0} \n delta: {1} \n uv: {2}'.format(alpha[inds_D[0]],delta[inds_D[1]],uv[inds_D[2]]))
print('Information: {0}'.format(hiD))
print('Single experiment optimal design (E optimality): \n alpha: {0} \n delta: {1} \n uv: {2}'.format(alpha[inds[0]],delta[inds[1]],uv[inds[2]]))
print('Information: {0}'.format(hi))
f3 = plt.figure()
XX,YY = np.meshgrid(uv,delta)
ax3 = f3.gca(projection='3d')
surf = ax3.plot_surface(XX, YY, all_E_opts[1,:,:], cmap='Oranges',
                       linewidth=0, antialiased=False)
ax3.view_init(elev=16, azim=-126)
ax3.grid(False)
f3.colorbar(surf)
f3.show()


# Next, we aim to find the best experiment when the first experiment is fixed.  
all_E_opts_2 = np.empty((len(alpha),len(delta),len(uv)))
all_D_opts_2 = np.empty((len(alpha),len(delta),len(uv)))
sFIMs_1 = np.loadtxt('out/all_fims_sampled_fim/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}_sampled.txt'.format(alpha[inds[0]],delta[inds[1]],uv[inds[2]]))
sFIMs_1 = sFIMs_1.reshape(7,7,100)
for i in range(len(alpha)):
    for j in range(len(delta)):
        for k in range(len(uv)):
            if method=='sampled':
                sFIMs = np.loadtxt('out/all_fims_sampled_fim/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}_sampled.txt'.format(alpha[i],delta[j],uv[k])) 
                sFIMs = sFIMs.reshape(7,7,100)
                sFIMs = sFIMs+sFIMs_1
                tmpE = []
                tmpD = [] 
                for m in keepers:
                    vals,vecs = np.linalg.eig(sFIMs[:,:,m])
                    tmpE.append(np.min(vals))
                    tmpD.append(np.prod(vals))
                all_E_opts_2[i,j,k] = np.mean(tmpE)
                all_D_opts_2[i,j,k] = np.mean(tmpD)

hi = np.max(all_E_opts_2)
hiD = np.max(all_D_opts_2)
inds = np.unravel_index(np.argmax(all_E_opts_2),all_E_opts_2.shape)
inds_D = np.unravel_index(np.argmax(all_D_opts_2),all_D_opts_2.shape)
print('Greedy experiment optimal design (D optimality): \n alpha: {0} \n delta: {1} \n uv: {2}'.format(alpha[inds_D[0]],delta[inds_D[1]],uv[inds_D[2]]))
print('Information: {0}'.format(hiD))
print('Greedy experiment optimal design (E optimality): \n alpha: {0} \n delta: {1} \n uv: {2}'.format(alpha[inds[0]],delta[inds[1]],uv[inds[2]]))
print('Information: {0}'.format(hi))

# Dual experiment (expensive, ran on WM Keck Cluster).
#all_D_opts_3 = np.empty((len(alpha),len(delta),len(uv),len(alpha),len(delta),len(uv)))
#all_E_opts_3 = np.empty((len(alpha),len(delta),len(uv),len(alpha),len(delta),len(uv)))
#for i in range(len(alpha)):
#    print(i)
#    for j in range(len(delta)):
#        for k in range(len(uv)):
#            for ii in range(len(alpha)):
#                for jj in range(len(delta)):
#                    for kk in range(len(uv)):
#                        sFIMs1 = np.loadtxt('out/all_fims_sampled_fim/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}_sampled.txt'.format(alpha[i],delta[j],uv[k])) 
#                        sFIMs2 = np.loadtxt('out/all_fims_sampled_fim/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}_sampled.txt'.format(alpha[ii],delta[jj],uv[kk])) 
#                        sFIMs = (sFIMs1+sFIMs2).reshape(7,7,100)
#                        tmpE = []
#                        tmpD = [] 
#                        for m in keepers:
#                            vals,vecs = np.linalg.eig(sFIMs[:,:,m])
#                            tmpE.append(np.min(vals))
#                            tmpD.append(np.prod(vals))
#                        all_E_opts_3[i,j,k,ii,jj,kk] = np.mean(tmpE)
#                        all_D_opts_3[i,j,k,ii,jj,kk] = np.mean(tmpD)
#hi = np.max(all_E_opts_3)
#hiD = np.max(all_D_opts_3)
#inds = np.unravel_index(np.argmax(all_E_opts_3),all_E_opts_3.shape)
#inds_D = np.unravel_index(np.argmax(all_D_opts_3),all_D_opts_3.shape)
#print('Dual experiment optimal design (D optimality): \n alpha: ({0},{1}) \n delta: ({2},{3}) \n uv: ({4},{5})'.format(alpha[inds_D[0]],alpha[inds_D[3]],delta[inds_D[1]],delta[inds_D[4]],uv[inds_D[2]],uv[inds_D[5]]))
#print('Information: {0}'.format(hiD))
#print('Dual experiment optimal design (E optimality): \n alpha: ({0},{1}) \n delta: ({2},{3}) \n uv: ({4},{5})'.format(alpha[inds[0]],alpha[inds[3]],delta[inds[1]],delta[inds[4]],uv[inds[2]],uv[inds[5]]))
#print('Information: {0}'.format(hi))
##hiD = np.max(all_D_opts_2_exp)
