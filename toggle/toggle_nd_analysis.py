import numpy as np
import matplotlib.pyplot as plt
import toggle
import vis
from scipy.stats import multivariate_normal

'''
Validate the toggle-FIM for two free parameters, Fig. 6 
in the manuscript. 
'''

def get_toggle_FIM(parameters,log=False,fim_type='mean'):
    '''
    compute the FSP-FIM for the toggle model. 
    '''
    print('Computing FSP-FIM')
    fsp = toggle.ToggleFSP(70,100,parameters,10)
    fsp.tvec = 3600*np.array([0.0,2.0]) 
    fsp.Nc = 1000
    UV_param = 3.84e-4 
    fsp.params[8] = UV_param
    fsp.dpars = [1,5]
    #fsp.dpars = np.array([0,1,2,3,4,5,9])
    if log:
        fsp.get_FIM(tstart=1,log=True)
        print('Sensitivity matrix shape {0}'.format(fsp.S.shape))
        CRLB_baseline = np.linalg.inv(fsp.FIM)
        np.savetxt('out/new/crlb_toggle_'+fim_type+'_baseline_log.txt',CRLB_baseline)
    else:
        fsp.get_FIM(tstart=1,log=False)
        CRLB_baseline = np.linalg.inv(fsp.FIM)
        np.savetxt('out/new/crlb_toggle_'+fim_type+'_baseline_exp.txt',CRLB_baseline)
    return CRLB_baseline,fsp

if __name__=='__main__':
    # load data and store the best parameters.
    all_covs = []
    f,ax = plt.subplots()
    free_params = np.array([1,5])
    start = 50
    f2,ax2 = plt.subplots()
    # get the toggle FIM around the true parameters
    true_parameters = np.array([6.8e-5,2.2e-3,1.6e-2,1.7e-2,6.1e-3,2.6e-3,2.1,3.0,3.84e-4,3.8e-4]) 
    try: 
        CRLB =np.loadtxt('out/crlb_toggle_true_baseline_log.txt')
    except:
        CRLB,model = get_toggle_FIM(true_parameters,log=True,fim_type='true')
    all_pml = []
    all_ptrue = []
    all_new_ML = []
    for i in range(100):
        # load parameters and store
        all_new_ML.append(np.loadtxt('out/parameters_fmin/two_parameter_analysis/parameters_{0}.txt'.format(i))[free_params])

    all_new_ML = np.array(all_new_ML)
    logML = np.log(all_new_ML[:,[0,1]])
    ax2.scatter(np.log(all_new_ML[:,0]),np.log(all_new_ML[:,1]),zorder=1,c='k',alpha=.5)
    ax2.scatter(np.log(true_parameters[free_params[0]]),np.log(true_parameters[free_params[1]]),c='gold',zorder=1)
    ax2 = vis.plot_conf_ellipse(np.log(true_parameters[free_params]),CRLB,ax=ax2,color='k')
    ax2.set_xlabel(r'$\log{ b_y}$')
    ax2.set_ylabel(r'$\log{ a_{xy}}$')
    ax2.set_xlim([-6.25,-6.0])
    ax2.set_ylim([-6.6,-4.7])
    f2.tight_layout()
    f2.show()
#    f2.savefig('../../figures/toggle/toggle_MLE_fig.pdf')


