import sys
import importlib
sys.path.append('../')
import birth_death as bd
import numpy as np
import matplotlib.pyplot as plt
import vis 
import time 
from scipy.stats import poisson

'''
This script makes Figure 1 in the manuscript.
'''
# specify a number of production rates. 
krs = np.logspace(-1,2,50)
mom_fim_00 = []
mom_fim_11 = []
norm_an_00 = []
norm_an_11 = []

fsp_fim_00 = []
fsp_fim_11 = []
pois_an_00 = []
pois_an_11 = []
means = []
tstart=1
for kr in krs:
    # fixed degradation rate
    g = 1.0 
    # define parameters
    params = [kr,g]
    Nc = 500
    # Make FSP model, get FIM
    fsp = bd.BirthDeathFSP(params)
    fsp.Nc = Nc
    fsp.N =150 
    fsp.tf = 10 
    fsp.ptimes = 5 
    fsp.get_FIM(tstart=tstart)
    
    # Make moment model, get FIM
    moments = bd.BirthDeathMoments(params) 
    moments.Nc=Nc 
    moments.tf = fsp.tf 
    moments.ptimes =fsp.ptimes 
    moments.observables = [0]
    moments.get_FIM(order=2,tstart=tstart) 
    
    # Get analytical info for poisson and normal dist. 
    lamb = []
    norm = []
    poisskr = []
    poissg = []
    
    normkr = []
    normg = []
    
    # Get the analytical FIM from theory for both 
    # the poisson and gaussian approximation. 
    for t in fsp.tvec[tstart:]:
        lamb.append( kr/float(g) * ( 1- np.exp(-g*t)) )
        nkern = .25*(2.0/(lamb[-1]**2) + 4.0/lamb[-1])
    
        normkr.append(Nc*nkern * (1.0/(g**2))*(1.0-np.exp(-g*t))**2)
        normg.append(Nc*nkern * ((kr/g)*t*np.exp(-g*t) - (kr/g**2) * (1 - np.exp(-g*t)))**2)
    
        poisskr.append( Nc*(1.0/lamb[-1]) * (1.0/(g**2))*(1.0-np.exp(-g*t))**2)
        poissg.append(Nc*1.0/lamb[-1] * ((kr/g)*t*np.exp(-g*t) - (kr/g**2) * (1 - np.exp(-g*t)))**2)

    mom_fim_00.append(moments.FIM[0,0])
    mom_fim_11.append(moments.FIM[1,1])

    norm_an_00.append(np.sum(normkr))
    norm_an_11.append(np.sum(normg)) 

    fsp_fim_00.append(fsp.FIM[0,0])
    fsp_fim_11.append(fsp.FIM[1,1])

    pois_an_00.append(np.sum(poisskr))
    pois_an_11.append(np.sum(poissg))
    
    means.append(lamb[-1])
    
f,ax = plt.subplots(1,2,figsize=(12,4))
ax[0].plot(means,norm_an_00,'r',linewidth=4,label='LNA (analytical)')
ax[0].plot(means,mom_fim_00,'y--',linewidth=4,label='LNA (numerical)')
ax[0].plot(means,pois_an_00,'orange',linewidth=4,label='Poisson (analytical)')
ax[0].plot(means,fsp_fim_00,'c--',linewidth=4,label='FSP (numerical)')
ax[0].set_xlim([krs[0],krs[-1]])
ax[0].set_xlabel(r'$\lambda$',size=20)
ax[0].set_ylabel(r'$I_{k_r}(\theta)$',size=20)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].legend()

ax[1].plot(means,norm_an_11,'r',linewidth=4,label='LNA (analytical)')
ax[1].plot(means,mom_fim_11,'y--',linewidth=4,label='LNA (numerical)')
ax[1].plot(means,pois_an_11,'orange',linewidth=4,label='Poisson (analytical)')
ax[1].plot(means,fsp_fim_11,'c--',linewidth=4,label='FSP (numerical)')
ax[1].set_xlim([krs[0],krs[-1]])
ax[1].set_xlabel(r'$\lambda$',size=20)
ax[1].set_ylabel(r'$I_{\gamma}(\theta)$',size=20)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
plt.tight_layout()
plt.show()
