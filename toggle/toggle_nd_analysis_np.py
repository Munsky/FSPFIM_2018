import numpy as np
import matplotlib.pyplot as plt
import toggle
import vis
from scipy.stats import multivariate_normal
import pickle

'''
Verify the toggle-FIM for multiple parameters. 
'''

def get_toggle_FIM(parameters,log=False,fim_type='mean',uv=0):
    '''
    '''
    print('Computing FSP-FIM')
    fsp = toggle.ToggleFSP(70,100,parameters,10)
    fsp.tvec = 3600*np.array([0.0,1.0,4.0,8.0]) 
    fsp.Nc = 1000
    fsp.dpars = np.array([0,1,2,3,4,5,9])
    if log:
        fsp.get_FIM(tstart=1,log=True)
        print('Sensitivity matrix shape {0}'.format(fsp.S.shape))
        CRLB_baseline = np.linalg.inv(fsp.FIM)
        np.savetxt('out/new/crlb_toggle_'+fim_type+'_baseline_log_np_uv_{0}.txt'.format(uv),CRLB_baseline)
    else:
        fsp.get_FIM(tstart=1,log=False)
        CRLB_baseline = np.linalg.inv(fsp.FIM)
        np.savetxt('out/new/crlb_toggle_'+fim_type+'_baseline_exp_np.txt',CRLB_baseline)
    return CRLB_baseline,fsp

def make_np_scatter(free_parameters,all_MLE,CRLB_pdf,true_pars,uv=0):
    '''
    Make a big scatter plot of all the parameters. 
    '''
    #parameter_names = [r'$b_x$',r'$b_y$',r'$k_x$',r'$k_y$',r'$a_{yx}$',r'$a_{xy}$',r'$\gamma_x$',r'$\gamma_y$']
    parameter_names = [r'$\tilde{b}_x$',r'$\tilde{b}_y$',r'$\tilde{k}_x$',r'$\tilde{k}_y$',r'$\tilde{\alpha}_{yx}$',r'$\tilde{\alpha}_{xy}$',r'$\tilde{\gamma}_x$',r'$\gamma_y$']
    limits = [[-31,3],[-6.5,-5.75],[-4.25,-4.0],[-4.6,-3.6],[-5.4,-4.75],[-6.75,-5.4],[-8.2,-7.5]]
    f,ax = plt.subplots(len(free_parameters),len(free_parameters),figsize=(10,7))
    color = (163/255,92/255,158/255)
    for i in range(len(free_parameters)):
        for j in range(i+1):
#            ax[i,j].clear()
            #cov_ij = cov_pdf[:,[j,i]][[j,i],:]
            crlb_ij = CRLB_pdf[:,[j,i]][[j,i],:]
            if i==j:
                ax[i,j].hist(np.log(all_MLE[:,i]),bins=30,color='gray')
                ax[i,j].set_xlim(limits[i])
                #ax[i,j].set_ylim([0,10])
            else:
                ax[i,j].scatter(np.log(all_MLE[::2,j]),np.log(all_MLE[::2,i]),c=color,alpha=.5)
                # plot the best parameters
                ax[i,j].scatter(np.log(true_pars[j]),np.log(true_pars[i]),c='gold')
                ax[i,j].scatter(np.log(all_MLE[1,j]),np.log(all_MLE[1,i]),c='red')
                # plot the 95% confidence interval for CRLB. 
                ax[i,j] = vis.plot_conf_ellipse(np.log(np.array([true_pars[j],true_pars[i]])),crlb_ij,ax=ax[i,j],color='k',crosshairs=False)
                ax[i,j].set_xlim(limits[j])
                ax[i,j].set_ylim(limits[i])
    # fix up axes
    for i in range(len(free_parameters)):
        for j in range(len(free_parameters)):
#            ax[i,j].set_xlim([.8,1.2])
#            ax[i,j].set_xticks([.9,1,1.1])
            if i==j:
                # make y axis labels be on the right for all plots.
                ax[i,j].tick_params(axis='y',labelsize=8,labelcolor='gray')            
                if i>0:
                    ax[i,j].yaxis.tick_right()
            if i==len(free_parameters)-1:
                # add xlabels and ticks to bottom row.
                ax[i,j].set_xlabel(parameter_names[j])
                ax[i,j].tick_params(axis='x',labelsize=8,labelcolor='gray')            
            if j==0:
                # add yticks and ylabels 
                ax[i,j].set_ylabel(parameter_names[i])
                if i>0:
                    ax[i,j].tick_params(axis='y',labelsize=8,labelcolor='gray')
            if i < (len(free_parameters)-1):
                    ax[i,j].set_xticks([])
                    ax[i,j].set_xticklabels([])
            if (j > 0 and j<i):
                    ax[i,j].set_yticks([])
                    ax[i,j].set_yticklabels([])
            if i<j:
                # turn off top right plots
                ax[i,j].axis('off')
                ax[i,j].set_xticklabels([])
                ax[i,j].set_yticklabels([])

    f.suptitle(r'$UV = {0}$ $J/m^2$ '.format(uv))
    return f,ax

def main():
    '''
    Generate the verification for all of the parameters . 
    '''
    # load data and store the best parameters.
    f,ax = plt.subplots()
    free_params = np.array([0,1,2,3,4,5,9])
    f4,ax4 = plt.subplots()
    # get the toggle FIM around the true parameters
    true_parameters = np.array([6.8e-5,2.2e-3,1.6e-2,1.7e-2,6.1e-3,2.6e-3,2.1,3.0,3.84e-4,3.8e-4]) 
    try: 
        CRLB =np.loadtxt('out/crlb_toggle_true_baseline_log_np.txt')
    except:
        CRLB,model = get_toggle_FIM(true_parameters,log=True,fim_type='true')
    all_pml = []
    all_ptrue = []
    all_new_ML = []
    for i in range(100):
        print('Loading and processing data set {0}'.format(i))
        # load parameters and store
        try: 
            all_new_ML.append(np.loadtxt('out/parameters_fmin/seven_parameter_analysis/parameters_{0}.txt'.format(i))[free_params])
        except:
            print('Unable to load MLE parameters from data set {0}'.format(i))

    all_new_ML = np.array(all_new_ML)

    # get the FIM around the mean of the ellipse just for fun
    mpars = np.copy(true_parameters)
    mpars[free_params] = np.mean(all_new_ML,axis=0)
    f4,ax4 = make_np_scatter(free_params,all_new_ML,CRLB,true_parameters[free_params])
    f4.show()

if __name__=='__main__':
    main()
