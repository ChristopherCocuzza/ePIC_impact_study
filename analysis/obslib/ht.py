#!/usr/bin/env python
import os,sys
from tools.config import load_config,conf
from fitlib.resman import RESMAN
import numpy as np

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab  as py
from matplotlib.ticker import MultipleLocator
from matplotlib import cm

#--from tools
from tools.tools     import load,save,checkdir,lprint

#--from corelib
from analysis.corelib import core,classifier

from analysis.obslib import stf

import kmeanconf as kc

cmap = cm.get_cmap('plasma')

TAR = ['p','n']

#--plot as function of W2
def gen_ht_W2fixed(wdir,W2=3.5):
  
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating HT from %s for at W2 = %3.5f'%(wdir,W2))

    if 'ht4' not in conf['steps'][istep]['active distributions']:
        if 'ht4' not in conf['steps'][istep]['passive distributions']:
                print('ht is not an active or passive distribution')
                return 

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    M2 = conf['aux'].M2
    parman=resman.parman
    resman.setup_idis()

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    #--set up plotting kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))
    Q2 = X/(1-X)*(W2 - M2) 

    ht4  = conf['ht4']
    idis = conf['idis']

    STF = ['F2','FL','F3']
    HT = {}
    C  = {}
    FX = {}
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for tar in TAR:
            if tar not in HT: HT[tar] = {}
            if tar not in C:  C [tar] = {}
            if tar not in FX: FX[tar] = {}
            for stf in STF:
                if stf not in HT[tar]: HT[tar][stf] = []
                if stf not in C[tar]:  C [tar][stf] = []
                if stf not in FX[tar]: FX[tar][stf] = []

                ht = ht4.get_ht(X,Q2,tar,stf)

                fx = conf['idis'].get_FX(stf,X,Q2,tar,idx=None)
                
                c = ht/fx

                HT[tar][stf].append(ht)
                C [tar][stf].append(c)
                FX[tar][stf].append(fx)

    for tar in TAR:
        for stf in STF:
            HT[tar][stf] = np.array(HT[tar][stf])
            C [tar][stf] = np.array(C [tar][stf])
            FX[tar][stf] = np.array(FX[tar][stf])

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/ht-W2=%3.5f.dat'%(wdir,W2)

    save({'X':X,'Q2':Q2,'HT':HT,'C':C,'FX':FX},filename)
    print ('Saving data to %s'%filename)

def plot_ht_W2fixed(wdir,W2=3.5,stf='F2',mode=1):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1) 
    ax12 = py.subplot(nrows,ncols,2) 


    hand = {}
    replicas = core.get_replicas(wdir)

    scale = classifier.get_scale(wdir)

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    core.mod_conf(istep,replicas[0])

    if 'ht4' not in conf['steps'][istep]['active distributions']:
        if 'ht4' not in conf['steps'][istep]['passive distributions']:
                print('ht is not an active or passive distribution')
                return 

    replicas = core.get_replicas(wdir)

    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas = jar['replicas']

    #--generate structure functions
    HT = {}
    filename ='%s/data/ht-W2=%3.5f.dat'%(wdir,W2)
    try:
        HT = load(filename)
    except:
        gen_ht_W2fixed(wdir,W2=W2)
        HT = load(filename)


    X       = HT['X']
    Q2      = HT['Q2']

    #ht_type = 'mult'
    #if 'ht type' in conf: ht_type = conf['ht type']

    ##############################################
    #--plot ht
    ##############################################
    ratp = (np.abs(HT['HT']['p'][stf]))/HT['FX']['p'][stf]
    ratn = (np.abs(HT['HT']['n'][stf]))/HT['FX']['n'][stf]

    if mode==0:
        for i in range(len(HT['HT']['p'][stf])):
            hand['p']     ,= ax11.plot(X,HT['HT']['p'][stf][i]       ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax11.plot(X,HT['HT']['n'][stf][i]       ,color='green' ,alpha=0.3)
            hand['p']     ,= ax12.plot(X,ratp[i]       ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax12.plot(X,ratn[i]       ,color='green' ,alpha=0.3)

    if mode == 1:
        HTmeanp = np.mean(HT['HT']['p'][stf],axis=0)
        HTmeann = np.mean(HT['HT']['n'][stf],axis=0)
        HTstdp  = np.std (HT['HT']['p'][stf],axis=0)
        HTstdn  = np.std (HT['HT']['n'][stf],axis=0)
        hand['p'] = ax11.fill_between(X,HTmeanp-HTstdp,HTmeanp+HTstdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n'] = ax11.fill_between(X,HTmeann-HTstdn,HTmeann+HTstdn,color='green',alpha=0.9, zorder=1)

        ratpmean = np.mean(ratp,axis=0)
        ratnmean = np.mean(ratn,axis=0)
        ratpstd  = np.std (ratp,axis=0)
        ratnstd  = np.std (ratn,axis=0)

        hand['p'] = ax12.fill_between(X,ratpmean-ratpstd,ratpmean+ratpstd,color='red'  ,alpha=0.9, zorder=2)
        hand['n'] = ax12.fill_between(X,ratnmean-ratnstd,ratnmean+ratnstd,color='green',alpha=0.9, zorder=1)

    print 
    ##############################################
    #h0 =-3.2874
    #h1 = 1.9274
    #h2 =-2.0701
    #ht = h0*X**h1*(1+h2*X)
    #hand['CJ15'] ,= ax11.plot(X,ht/Q2,'k--')
  
    if stf=='F2': 
        ax11.set_ylim(0,0.018)
        ax11.set_yticks([0.005,0.010,0.015])
        ax11.yaxis.set_minor_locator(MultipleLocator(0.001))

        ax12.set_ylim(0,0.38)
        ax12.set_yticks([0.1,0.2,0.3])
        ax12.yaxis.set_minor_locator(MultipleLocator(0.05))

    if stf=='FL': 
        ax11.set_ylim(-0.018,0.018)
        #ax11.set_yticks([0.005,0.010,0.015])
        #ax11.yaxis.set_minor_locator(MultipleLocator(0.001))

        ax12.set_ylim(-5.00,5.00)
        #ax12.set_yticks([0.1,0.2,0.3])
        #ax12.yaxis.set_minor_locator(MultipleLocator(0.05))

    k = stf[-1]
    ax11.text(0.03,0.82,r'\boldmath$F^{N,{\rm HT}}_%s$'%(k)                              ,transform=ax11.transAxes,size=40)
    ax12.text(0.03,0.82,r'\boldmath$F_%s^{N,{\rm HT}}/F_%s^{N,{\rm tot}}$'%(k,k),         transform=ax12.transAxes,size=40)

    M2 = 0.938**2
    xmin,xmax = 0.43,0.90
    Q2min = xmin/(1-xmin)*(W2 - M2) 
    Q2max = xmax/(1-xmax)*(W2 - M2) 
    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(xmin,xmax)
        ax.set_xlabel(r'\boldmath$x$',size=40)
        ax.xaxis.set_label_coords(0.96,0.01)
        #ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.5,0.6,0.7,0.8])

    ax11.text(0.03,0.15,r'$W^2 = %3.1f~{\rm GeV}^2$'%W2,                          transform=ax11.transAxes,size=25)
    ax11.text(0.03,0.05,r'$%3.2f < Q^2 < %3.0f~{\rm GeV}^2$'%(Q2min,Q2max),       transform=ax11.transAxes,size=25)
    #if conf['tmc']=='GP':  ax11.text(0.05,0.05,r'\textbf{\textrm{(GP)}}' ,size=25,transform=ax11.transAxes)
    #if conf['tmc']=='AOT': ax11.text(0.05,0.05,r'\textbf{\textrm{(AOT)}}',size=25,transform=ax11.transAxes)





    handles,labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    #handles.append(hand['CJ15'])
    labels.append(r'\boldmath$p$')
    labels.append(r'\boldmath$n$')
    #labels.append(r'\textbf{\textrm{CJ15}}')
    ax11.legend(handles,labels,frameon=False,loc='upper right',fontsize=30, handletextpad = 0.5, handlelength = 1.5, ncol=1, columnspacing = 0.5)

    py.tight_layout()
    filename = '%s/gallery/ht-%s-W2=%3.5f'%(wdir,stf,W2)
    if mode == 1: filename += '-bands'
    filename += '.png'
    print()
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()


#--plot as function of Q2
#--need to update
def old_gen_ht_Q2fixed(wdir,Q2=None):
  
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating STF from %s for %s at Q2 = %3.5f'%(wdir,tar,Q2))

    if 'ht4' not in conf['steps'][istep]['active distributions']:
        if 'ht4' not in conf['steps'][istep]['passive distributions']:
                print('ht is not an active or passive distribution')
                return 

    M2 = conf['aux'].M2

    #--setup kinematics for higher twist to be calculated at
    #Xgrid = np.geomspace(1e-5,1e-1,20)
    #Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    #Q2grid = Xgrid/(1-Xgrid)*(W2cut - M2) 
    #conf['idis grid'] = {}
    #conf['idis grid']['X']  = Xgrid 
    #conf['idis grid']['Q2'] = Q2grid 
    conf['idis grid'] = 'predict'
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
    
    idis.data[tar] = {}
    idis.data[tar]['F2'] = np.zeros(idis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics for structure functions to be interpolated to
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))
    Q2 = X/(1-X)*(W2cut - M2) 

    ht4 = conf['ht4']
    #--compute X*STF for all replicas        
    XF=[]
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        idis._update()

        xf = X*idis.get_stf(X,Q2,stf='F2',tar=tar)*ht4.get_ht(X,Q2,tar,'F2')
        XF.append(xf)

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/ht-%s-Q2=%3.5f.dat'%(wdir,tar,W2cut)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def gen_ht_Q2fixed(wdir,Q2=1):
  
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating HT from %s for at Q2 = %3.5f'%(wdir,Q2))

    if 'ht4' not in conf['steps'][istep]['active distributions']:
        if 'ht4' not in conf['steps'][istep]['passive distributions']:
                print('ht is not an active or passive distribution')
                return 

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    M2 = conf['aux'].M2
    parman=resman.parman
    resman.setup_idis()

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    #--set up plotting kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))
    Q2 = Q2 * np.ones(len(X)) 

    ht4  = conf['ht4']
    idis = conf['idis']

    STF = ['F2','FL','F3']
    HT = {}
    C  = {}
    FX = {}
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for tar in TAR:
            if tar not in HT: HT[tar] = {}
            if tar not in C:  C [tar] = {}
            if tar not in FX: FX[tar] = {}
            for stf in STF:
                if stf not in HT[tar]: HT[tar][stf] = []
                if stf not in C[tar]:  C [tar][stf] = []
                if stf not in FX[tar]: FX[tar][stf] = []

                ht = ht4.get_ht(X,Q2,tar,stf)

                #--turn off ht in order to get LT structure function
                conf['idis'].ht_flag = False
                fx = conf['idis'].get_FX(stf,X,Q2,tar,idx=None)
                conf['idis'].ht_flag = True
                
                c = ht/fx

                HT[tar][stf].append(ht)
                C [tar][stf].append(c)
                FX[tar][stf].append(fx)

    for tar in TAR:
        for stf in STF:
            HT[tar][stf] = np.array(HT[tar][stf])
            C [tar][stf] = np.array(C [tar][stf])
            FX[tar][stf] = np.array(FX[tar][stf])

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/ht-Q2=%3.5f.dat'%(wdir,Q2[0])

    save({'X':X,'Q2':Q2,'HT':HT,'C':C,'FX':FX},filename)
    print ('Saving data to %s'%filename)

def plot_ht_Q2fixed(wdir,Q2=None,_stf='F2',mode=0):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1) 
    ax12 = py.subplot(nrows,ncols,2) 


    hand = {}
    replicas = core.get_replicas(wdir)

    scale = classifier.get_scale(wdir)

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if Q2==None: Q2 = conf['Q20']
    core.mod_conf(istep,replicas[0])
    #conf['idis grid'] = 'prediction'
    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman

    replicas = core.get_replicas(wdir)

    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']

    #--generate structure functions
    if Q2 == None: Q2=conf['Q20']
    TAR = ['p','n']
    filename ='%s/data/stf-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        STF = load(filename)
    except:
        stf.gen_stf(wdir,Q2)
        STF = load(filename)

    X   = STF['X']

    STF['p'] = STF['XF']['p'][_stf]
    STF['n'] = STF['XF']['p'][_stf]


    if 'ht4' in conf: ht4 = conf['ht4']
    else:
        print('Higher twist corrections not present.')
        return       

    ht_type = 'mult'
    if 'ht type' in conf: ht_type = conf['ht type']

    ##############################################
    #--plot ht
    ##############################################
    cnt = 0
    FXp, FXn, FXpHT, FXnHT = [], [], [], []
    plus, minus = [], []
    for par in replicas:
        parman.set_new_params(par)
        if ht_type=='mult':
            FXpHT.append(ht4.get_ht(X,Q2,'p',_stf))
            FXnHT.append(ht4.get_ht(X,Q2,'n',_stf))
            FXp.append(STF['p'][cnt]*ht4.get_ht(X,Q2,'p',_stf)/X)
            FXn.append(STF['n'][cnt]*ht4.get_ht(X,Q2,'n',_stf)/X)
            plus.append (STF['n'][cnt]*ht4.get_ht(X,Q2,'n',_stf)/X + STF['p'][cnt]*ht4.get_ht(X,Q2,'p',_stf)/X)
            minus.append(STF['n'][cnt]*ht4.get_ht(X,Q2,'n',_stf)/X - STF['p'][cnt]*ht4.get_ht(X,Q2,'p',_stf)/X)
        else: pass
        if mode==0:
            hand['p']     ,= ax11.plot(X,FXpHT[cnt] ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax11.plot(X,FXnHT[cnt] ,color='green' ,alpha=0.3)
            hand['p']     ,= ax12.plot(X,FXp[cnt]   ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax12.plot(X,FXn[cnt]   ,color='green' ,alpha=0.3)
            hand['plus']  ,= ax12.plot(X,plus[cnt]  ,color='blue'  ,alpha=0.3)
            hand['minus'] ,= ax12.plot(X,minus[cnt] ,color='purple',alpha=0.3)
        cnt+=1

    if mode == 1:
        meanp = np.mean(np.array(FXpHT),axis=0)
        stdp  = np.std (np.array(FXpHT),axis=0)
        meann = np.mean(np.array(FXnHT),axis=0)
        stdn  = np.std (np.array(FXnHT),axis=0)
        hand['p'] = ax11.fill_between(X,meanp-stdp,meanp+stdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n'] = ax11.fill_between(X,meann-stdn,meann+stdn,color='green',alpha=0.9, zorder=1)
        meanp = np.mean(np.array(FXp),axis=0)
        stdp  = np.std (np.array(FXp),axis=0)
        meann = np.mean(np.array(FXn),axis=0)
        stdn  = np.std (np.array(FXn),axis=0)
        meanplus  = np.mean(np.array(plus),axis=0)
        stdplus   = np.std (np.array(plus),axis=0)
        meanminus = np.mean(np.array(minus),axis=0)
        stdminus  = np.std (np.array(minus),axis=0)
        hand['p']     = ax12.fill_between(X,meanp-stdp,meanp+stdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n']     = ax12.fill_between(X,meann-stdn,meann+stdn,color='green',alpha=0.9, zorder=1)
        hand['plus']  = ax12.fill_between(X,meanplus-stdplus  ,meanplus+stdplus  ,color='blue'  ,alpha=0.9, zorder=1)
        hand['minus'] = ax12.fill_between(X,meanminus-stdminus,meanminus+stdminus,color='purple',alpha=0.9, zorder=1)

    print 
    ##############################################
    if _stf=='F2':
        h0 =-3.2874
        h1 = 1.9274
        h2 =-2.0701
        ht = h0*X**h1*(1+h2*X)/Q2
        hand['CJ15'] ,= ax11.plot(X,ht,'k--')
   
    ax11.set_ylim(-0.09,0.25)
    ax12.set_ylim(-0.03,0.07)

    ax11.text(0.05,0.32,r'\boldmath$\frac{C_{\rm HT}^N}{Q^2}$',transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.05,r'\boldmath$F_2^N \frac{C^N_{\rm HT}}{Q^2}$',transform=ax12.transAxes,size=40)

    if Q2==1.27**2: ax12.text(0.50,0.10,r'$Q^2 = m_c^2$',transform=ax12.transAxes,size=25)
    else:           ax12.text(0.50,0.10,r'$Q^2 = %3.2f~{\rm GeV}^2$'%Q2,transform=ax12.transAxes,size=25)
    
    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0,0.5)
        ax.set_xlabel(r'\boldmath$x$',size=30)
        ax.xaxis.set_label_coords(0.95,0.00)
        ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0,0.1,0.2,0.3,0.4])

    if conf['tmc']=='GP':  ax11.text(0.05,0.05,r'\textbf{\textrm{GP}}' ,size=25,transform=ax11.transAxes)
    if conf['tmc']=='AOT': ax11.text(0.05,0.05,r'\textbf{\textrm{AOT}}',size=25,transform=ax11.transAxes)

    minorLocator = MultipleLocator(0.025)
    majorLocator = MultipleLocator(0.1)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.02)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)

    handles,labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    if 'CJ15' in hand: handles.append(hand['CJ15'])
    labels.append(r'\boldmath$p$')
    labels.append(r'\boldmath$n$')
    if 'CJ15' in hand: labels.append(r'\textbf{\textrm{CJ15}}')
    ax11.legend(handles,labels,frameon=False,loc=2,fontsize=25, handletextpad = 0.5, handlelength = 1.5, ncol=1, columnspacing = 0.5)

    handles,labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    handles.append(hand['plus'])
    handles.append(hand['minus'])
    labels.append(r'\boldmath$p$')
    labels.append(r'\boldmath$n$')
    labels.append(r'\boldmath$n+p$')
    labels.append(r'\boldmath$n-p$')
    ax12.legend(handles,labels,frameon=False,loc=2,fontsize=25, handletextpad = 0.5, handlelength = 1.5, ncol=1, columnspacing = 0.5)


    #if mode==0:
    #    sm   = py.cm.ScalarMappable(cmap=cmap)
    #    sm.set_array([])
    #    cax = fig.add_axes([0.45,0.88,0.30,0.04])
    #    cax.tick_params(axis='both',which='both',labelsize=15,direction='in')
    #    cax.xaxis.set_label_coords(0.65,-1.8)
    #    cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    #    cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=20)

    py.tight_layout()
    filename = '%s/gallery/ht-%s-Q2=%3.5f'%(wdir,_stf,Q2)
    if mode == 1: filename += '-bands'
    filename += '.png'
    print()
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()


#--generate HT at nuclear level
def gen_htA_Q2fixed(wdir,Q2=10):
  
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating HTA from %s at Q2 = %3.5f'%(wdir,Q2))

    if 'ht4' not in conf['steps'][istep]['active distributions']:
        if 'ht4' not in conf['steps'][istep]['passive distributions']:
                print('ht is not an active or passive distribution')
                return 

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    M2 = conf['aux'].M2
    parman=resman.parman
    resman.setup_idis()

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    #--set up plotting kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))
    #Q2 = X/(1-X)*(W2 - M2) 
    Q2 = Q2*np.ones(len(X))

    ht4  = conf['ht4']
    idis = conf['idis']

    STF = ['F2','FL','F3']
    NUCTAR = ['d','h','t']
    HT = {}
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for tar in NUCTAR:
            if tar not in HT: HT[tar] = {}
            for stf in STF:
                if stf not in HT[tar]: HT[tar][stf] = []

                ht = conf['idis'].get_HTA(stf,X,Q2,tar,idx=None)
                
                HT[tar][stf].append(ht)

    for tar in NUCTAR:
        for stf in STF:
            HT[tar][stf] = np.array(HT[tar][stf])

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/htA-Q2=%3.5f.dat'%(wdir,Q2[0])

    save({'X':X,'Q2':Q2,'HT':HT},filename)
    print ('Saving data to %s'%filename)
    





