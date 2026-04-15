import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

#--matplotlib
import matplotlib
matplotlib.use('Agg')
import pylab as py
from matplotlib.ticker import MultipleLocator

#--from scipy stack 
from scipy.integrate import fixed_quad
from scipy import interpolate
import scipy.stats as stats

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#-- from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier
from analysis.corelib import summary

def get_zscore(wdir,process='sia kaon'):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    #labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    #cluster,colors,nc,order = get_clusters(wdir,istep,kc) 
    #chi2     = labels['chi2dof']
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))

    if process=='sia kaon':
        reaction = 'sia'
        had = 'kaon'

    chi2 = 0.0
    npts = 0.0
    for idx in data['reactions'][reaction]:
        if data['reactions'][reaction][idx]['hadron'][0] != had: continue
        res =np.array(data['reactions'][reaction][idx]['residuals-rep'])
        rres=np.array(data['reactions'][reaction][idx]['rres-rep'])
        chi2+=np.array([np.sum(r**2) for r in res])
        #chi2+=np.array([np.sum(r**2)/len(r) for r in rres])
        npts+=len(data['reactions'][reaction][idx]['value'])

    reps = len(chi2)

    chi2 = np.sort(chi2)

    pdf   = stats.chi2.pdf(chi2,npts)
    cdf   = stats.chi2.cdf(chi2,npts)
    sigma = stats.norm.ppf(cdf)

    nrows, ncols = 1,3
    fig, ax = py.subplots(nrows,ncols,figsize=(ncols*5,nrows*4))
    
    #x = np.linspace(0,reps,reps)
    x = np.linspace(0,2*npts)
    pdf   = stats.chi2.pdf(x,npts)
    cdf   = stats.chi2.cdf(x,npts)
    sigma = stats.norm.ppf(cdf)

    ax[0].plot(x,pdf)
    ax[1].plot(x,cdf)
    ax[2].plot(x,sigma)

    chi2red = 0.49*npts

    ax[0].axvline(chi2red)
    ax[1].axvline(chi2red)
    ax[2].axvline(chi2red)

    idx = (np.abs(x-chi2red)).argmin()
    result = sigma[idx]
    print(result)


    filename = '%s/gallery/zscore'%(wdir)
    py.savefig(filename)







