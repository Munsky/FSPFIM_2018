import numpy as np 
import matplotlib.pyplot as plt
import toggle

'''
Analyze optimal experiments for the toggle model.  
'''
def load_FIMs(alpha,delta,uv):
    '''
    load the FIMs and get the E-optimality and D-optimality
    '''
    E_opts = []
    D_opts = []
    sFIMs = np.loadtxt('out/all_fims_sampled_fim/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}_sampled.txt'.format(alpha,delta,uv))
    sFIMs = sFIMs.reshape(7,7,100)
    for i in range(100):
        FIM = sFIMs[:,:,i]
        vals,vecs = np.linalg.eig(FIM)
        E_opts.append(np.min(vals))
        D_opts.append(np.prod(vals))
    return np.array(E_opts),np.array(D_opts)

def plot_bars(alpha,delta,uv):
    '''
    make bar plots of the eigenvalues. 
    '''
    mu_evals = []
    std_evals = []
    mu_vars = []
    std_vars = []
    for n in range(len(alpha)):
        sFIMs = np.loadtxt('out/all_fims_sampled_fim/exp_design_FIM_alpha_{0}_delta_{1}_uv_{2}_sampled.txt'.format(alpha[n],delta[n],uv[n]))
        sFIMs = sFIMs.reshape(7,7,100)
        all_vals = np.zeros((100,7))
        all_vars = np.zeros((100,7))
        for i in range(100):
#            vals,vec = np.linalg.eig(sFIMs[:,:,i])
#            all_vals[i,:] = 1/np.sqrt(np.sort(vals))
            sigma = np.linalg.inv(sFIMs[:,:,i])
            vals,vec = np.linalg.eig(sigma)
            all_vals[i,:] = np.sort(vals)[::-1]
            all_vars[i,:] = np.sqrt(np.diag(sigma))

        mu_evals.append(np.mean(all_vals,axis=0))
        std_evals.append(np.sqrt(np.var(all_vals,axis=0)))
        mu_vars.append(np.mean(all_vars,axis=0))
        std_vars.append(np.sqrt(np.var(all_vars,axis=0)))
    print(std_vars)
    # plotting
    n_exp = len(mu_evals)
    f1,ax1 = plt.subplots()
    f2,ax2 = plt.subplots()
    locs = np.arange(7)
    width=1.0/(n_exp+1)
    #eig_labels = [r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$',r'$\lambda_4$',r'$\lambda_5$',r'$\lambda_6$',r'$\lambda_7$']
    eig_labels = [r'$v_1$',r'$v_2$',r'$v_3$',r'$v_4$',r'$v_5$',r'$v_6$',r'$v_7$']
    #parameter_names = [r'$\tilde{b}_x$',r'$\tilde{b}_y$',r'$\tilde{k}_x$',r'$\tilde{k}_y$',r'$\tilde{\alpha}_{yx}$',r'$\tilde{\alpha}_{xy}$',r'$\tilde{\gamma}_x$']
    parameter_names = [r'${b}_x$',r'${b}_y$',r'${k}_x$',r'${k}_y$',r'${\alpha}_{yx}$',r'${\alpha}_{xy}$',r'${\gamma}_x$']
    all_ticks = []
    colors = ['k','gray']
    xticks = width*(n_exp/2.0)+locs-width/2.0
    for k in range(n_exp):
        all_ticks.append(locs+k*width)
#        ax1.bar(locs+k*width,mu_evals[k],width = width,yerr =std_evals[k])
#        ax2.bar(locs+k*width,mu_vars[k],width = width,yerr =std_vars[k])
        ax1.bar(locs+k*width,mu_evals[k],width = width,color=colors[k])
        ax2.bar(locs+k*width,mu_vars[k],width = width,color=colors[k])
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax1.set_xticklabels(eig_labels)
    ax2.set_xticklabels(parameter_names)
    ax1.set_yscale('log')
#    ax2.set_yscale('log')
    #ax1.set_ylabel(r'$\frac{1}{\sqrt{\lambda_i}}$')
    ax1.set_ylabel(r'Eigenvalues of $FIM^{-1}$')
    ax2.set_ylabel(r'Standard deviation')
    ax1.legend(['Experiment A','Experiment B'])
    ax2.legend(['Experiment A','Experiment B'])
    f1.tight_layout()
    f2.tight_layout()
    f1.savefig('../../figures/eigendecomp_two_exp.eps')
    f2.savefig('../../figures/parameter_sd_two_exp.eps')
    return

def main(get_baseline=True):
    '''
    Generate supplementary figures for toggle analysis, make the scatter 
    plot to compare optimal experiment and nearby experiments. 
    '''
    E_opts_2,D_opts_2 = load_FIMs(FIM,2.0,3.0,9.0)
    E_opts_1,D_opts_1 = load_FIMs(FIM,3.0,4.0,6.0)
    print('Experiment 1 E-opt average: {0}'.format(np.mean(E_opts_1)))
    print('Experiment 2 E-opt average: {0}'.format(np.mean(E_opts_2)))
    print('Experiment 1 D-opt average: {0}'.format(np.mean(D_opts_1)))
    print('Experiment 2 D-opt average: {0}'.format(np.mean(D_opts_2)))
    print('E-opt had {0} out of {1} with same ranking.'.format(np.sum(E_opts_1<E_opts_2),len(E_opts_1)))
    print('D-opt had {0} out of {1} with same ranking.'.format(np.sum(D_opts_1<D_opts_2),len(D_opts_1)))
    f,ax = plt.subplots(1,2)
    plot_histogram([E_opts_1,E_opts_2],ax[0])
    plot_histogram([D_opts_1,D_opts_2],ax[1])
    f2,ax2 = plt.subplots(figsize=(8,8))
    ax2.hist(np.array(D_opts_2)-np.array(D_opts_1),color='mediumorchid',bins=30,alpha=.9)
    ax2.set_xlabel('(D-opt A) - (D-opt B)',size=12)
    ax2.set_ylabel('Frequency',size=12)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    f3,ax3 = plt.subplots(figsize=(6,4))
    ax3.scatter(E_opts_1,E_opts_2,c='orange',alpha=.5)
    ax3.plot([np.min(E_opts_1)*.1,np.max(E_opts_1)*5],[np.min(E_opts_1)*.1,np.max(E_opts_1)*5],'k--')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
#    ax3.set_xlim([np.min(D_opts_1)*.9,np.max(D_opts_1)*1.1])
#    ax3.set_ylim([np.min(D_opts_2)*.4,1.5*np.max(D_opts_2)])
    ax3.set_xlim([np.min(E_opts_1)*.9,np.max(E_opts_1)*1.5])
    ax3.set_ylim([np.min(E_opts_2)*.4,1.5*np.max(E_opts_2)])
#    ax3.set_xlabel('D-opt B',size=12)
#    ax3.set_ylabel('D-opt A',size=12)
    ax3.set_xlabel('E-opt B',size=12)
    ax3.set_ylabel('E-opt A',size=12)
#    f.show()
#    f2.tight_layout()
#    f2.show()
    f3.tight_layout()
#    f3.savefig('../../figures/toggle/sample_exp_des_e_opt_new.eps')
    f3.show()

if __name__=='__main__':
    #main(False)
    plot_bars([2.0,3.0],[3.0,4.0],[9.0,6.0])
    plt.show()
