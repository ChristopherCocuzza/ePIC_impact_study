#!/usr/bin/env python
from tools.config import load_config,conf
from fitlib.resman import RESMAN
import numpy as np

#--matplotlib
import matplotlib
import pylab  as py
from matplotlib.ticker import MultipleLocator

#--from tools
from tools.tools import load,lprint,save

#--from corelib
from analysis.corelib import core,classifier

import kmeanconf as kc

#--generate offshell corrections
def gen_df0(wdir,Q2=None):

    replicas = core.get_replicas(wdir)

    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']


    istep = core.get_istep()
    core.mod_conf(istep,replicas[0])
    if 'off pdf' not in conf['steps'][istep]['active distributions']: 
        if 'off pdf' not in conf['steps'][istep]['passive distributions']: 
            return

    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman

    replicas = core.get_replicas(wdir)
    names    = core.get_replicas_names(wdir)

    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']
   
    resman.setup_idis() 
    
    if 'offpdf' in conf: off = conf['offpdf']
    else:
        print('Offshell corrections not present.')
        return

    ##############################################
    #--generate offshell
    ##############################################
    X1   = 10**np.linspace(-4,-1,100)
    X2   = np.linspace(0.1,0.98,100)
    X    = np.append(X1,X2)
    Q2   = np.ones(X.size)*Q2
    cnt = 0

    OFF = {}
    OFF['dfDp'] = []
    OFF['dfDm'] = []
    OFF['X'] = X
    for par in replicas:
         
        lprint('Generating offshell %s/%s'%(cnt+1,len(replicas)))
        parman.set_new_params(par)
        #--full structure functions
        F2p     = conf['idis'].get_FX('F2',X,Q2,'p',idx=None)
        F2n     = conf['idis'].get_FX('F2',X,Q2,'n',idx=None)
        #--offshell piece only
        F2p_off = conf['idis'].get_FX('F2',X,Q2,'p',kind='offshell',nucleus='d',idx=None)
        F2n_off = conf['idis'].get_FX('F2',X,Q2,'n',kind='offshell',nucleus='d',idx=None)
        dfDp  = ( F2p_off+F2n_off)/(F2p+F2n)
        dfDm  = (-F2p_off+F2n_off)/(F2p+F2n)
        OFF['dfDp'].append(dfDp)
        OFF['dfDm'].append(dfDm)
        cnt +=1
     
    filename ='%s/data/off-Q2=%3.5f.dat'%(wdir,Q2[0])
    save(OFF,filename) 
    print()
    print('Saving offshell data to %s'%filename)

def plot_df0(wdir,Q2=None,mode=0):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)

    j = 0
    hand = {}
    thy  = {}
    replicas = core.get_replicas(wdir)

    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']

    istep = core.get_istep()
    core.mod_conf(istep,replicas[0])
    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman

    replicas = core.get_replicas(wdir)
    names    = core.get_replicas_names(wdir)

    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']
    
    if 'off' in conf: off = conf['off']
    elif 'offpdf' in conf: off = conf['offpdf']
    else:
        print('Offshell corrections not present.')
        return

    if 'off' not in conf['steps'][istep]['active distributions']: 
        if 'off' not in conf['steps'][istep]['passive distributions']: 
            if 'off pdf' not in conf['steps'][istep]['active distributions']: 
                if 'off pdf' not in conf['steps'][istep]['passive distributions']: 
                    return

    #--try to load data.  Generate it if it does not exist
    filename ='%s/data/off-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data = load(filename)
    except:
        gen_off(wdir,Q2)
        data = load(filename)
    ##############################################
    #--plot offshell
    ##############################################
    X   = data['X']

    if 'dfD' in data:  dfD = data['dfD']
    if 'dfDp' in data: dfD = data['dfDp']
    if mode==0:
        for i in range(len(dfD)):        
            thy[j]    ,= ax11.plot(X,dfD[i],color='firebrick',alpha=0.2)
    if mode == 1:
       meanD = np.mean(np.array(dfD),axis=0)
       stdD  = np.std (np.array(dfD),axis=0)
       thy[j]    = ax11.fill_between(X,meanD-stdD,meanD+stdD,color='red'      ,alpha=0.7,zorder=1)

 
    ##############################################
    X1   = 10**np.linspace(-4,-1,100)
    X2   = np.linspace(0.1,0.98,100)
    X    = np.append(X1,X2)
    #--CJ15 
    C =-3.6735
    x0= 5.7717e-2
    x1=0.36419
    dfcj=C*(X-x0)*(X-x1)*(1+x0-X)
    hand['CJ'] ,= ax11.plot(X,dfcj,'b--')
    #--KP 
    C = 8.10
    x0= 0.448
    x1= 0.05
    dfcj=C*(X-x0)*(X-x1)*(1+x0-X)
    hand['KP'] ,= ax11.plot(X,dfcj,'g--')

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    ax11.text(0.30,0.85,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=30,transform=ax11.transAxes)

    ax11.text(0.05,0.85,r'\boldmath$\delta f^D$',transform=ax11.transAxes,size=40)
 
    for ax in [ax11]:
        ax.set_ylim(-2.2,1.2)
        ax.set_xlim(0,0.90)
        ax.axhline(0,alpha=0.5,color='k',ls='--')
        ax.set_xlabel(r'\boldmath$x$'         ,size=30)
        ax.xaxis.set_label_coords(0.98,0.00)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])


    handles,labels=[],[]

    handles.append(hand['CJ'])
    handles.append(hand['KP'])
    labels.append(r'\textbf{\textrm{CJ15}}')
    labels.append(r'\textbf{\textrm{KP}}')

    ax11.legend(handles,labels,frameon=False,loc='lower left',fontsize=28, handletextpad = 0.5, handlelength = 1.5, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(wspace=0)

    filename = '%s/gallery/off-Q2=%3.5f'%(wdir,Q2)
    if mode == 1: filename += '-bands'
    filename += '.png'
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()



def gen_df02(wdir,Q2=None):

    replicas = core.get_replicas(wdir)

    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']


    istep = core.get_istep()
    core.mod_conf(istep,replicas[0])
    if 'off pdf0' not in conf['steps'][istep]['active distributions']: 
        if 'off pdf0' not in conf['steps'][istep]['passive distributions']: 
            return

    if 'off pdf1' not in conf['steps'][istep]['active distributions']: 
        if 'off pdf1' not in conf['steps'][istep]['passive distributions']: 
            return

    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman

    replicas = core.get_replicas(wdir)
    names    = core.get_replicas_names(wdir)

    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']
   
    resman.setup_idis() 
    
    if 'offpdf' in conf: off = conf['offpdf']
    else:
        print('Offshell corrections not present.')
        return

    ##############################################
    #--generate offshell
    ##############################################
    X1   = 10**np.linspace(-4,-1,100)
    X2   = np.linspace(0.1,0.98,100)
    X    = np.append(X1,X2)
    Q2   = np.ones(X.size)*Q2
    cnt = 0

    OFF = {}
    OFF['dfDp'] = []
    OFF['dfDm'] = []
    OFF['X'] = X
    for par in replicas:
         
        lprint('Generating offshell %s/%s'%(cnt+1,len(replicas)))
        parman.set_new_params(par)
        #--full structure functions
        F2p     = conf['idis'].get_FX('F2',X,Q2,'p',idx=None)
        F2n     = conf['idis'].get_FX('F2',X,Q2,'n',idx=None)
        #--offshell piece only
        F2p_off = conf['idis'].get_FX('F2',X,Q2,'p',kind='offshell',nucleus='d',idx=None)
        F2n_off = conf['idis'].get_FX('F2',X,Q2,'n',kind='offshell',nucleus='d',idx=None)
        dfDp  = ( F2p_off+F2n_off)/(F2p+F2n)
        dfDm  = (-F2p_off+F2n_off)/(F2p+F2n)
        OFF['dfDp'].append(dfDp)
        OFF['dfDm'].append(dfDm)
        #print(dfDp)
        cnt +=1
     
    filename ='%s/data/off2-Q2=%3.5f.dat'%(wdir,Q2[0])
    save(OFF,filename) 
    print()
    print('Saving offshell data to %s'%filename)



















