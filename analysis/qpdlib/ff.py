#!/usr/bin/env python
import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

## matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as py

## from scipy stack
from scipy.integrate import quad

## from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

from qcdlib.qpdcalc import QPDCALC

## from fitlib
from fitlib.resman import RESMAN

## from local
from analysis.corelib import core
from analysis.corelib import classifier

import lhapdf

flav_dict = {}
flav_dict['g']  = 0
flav_dict['u']  = 2
flav_dict['d']  = 1
flav_dict['s']  = 3
flav_dict['c']  = 4
flav_dict['b']  = 5
flav_dict['ub'] = -2
flav_dict['db'] = -1
flav_dict['sb'] = -3
flav_dict['cb'] = -4
flav_dict['bb'] = -5

def gen_xf(wdir, had, flavors = ['g', 'u', 'ub', 'd', 'db', 's', 'sb','c','b'], Q2 = None):

    fflabel='ff%s'%had

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    if Q2==None: Q2 = conf['Q20']
    print('\ngenerating ff-%s from %s at %3.5f' % (had,wdir,Q2))

    _replicas = core.get_replicas(wdir)
    core.mod_conf(istep,_replicas[0]) ## set conf as specified in istep

    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('ff-%s not in active distribution' % had)
            return

    if fflabel in conf['steps'][istep]['active distributions']:
        passive = False
    else:
        passive = True

    resman = RESMAN(nworkers = 1, parallel = False, datasets = False)
    parman = resman.parman
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    ff = conf[fflabel]

    ## setup kinematics
    X = np.linspace(0.01, 0.99, 100)

    ## compute XF for all replicas
    XF = {}
    cnt = 0
    for par in replicas:
        if passive: core.mod_conf(istep,_replicas[cnt])   
        lprint('%d/%d' % (cnt+1, len(replicas)))

        parman.set_new_params(par, initial = True)


        for flavor in flavors:
            if flavor not in XF: XF[flavor] = []
            if   flavor=='c' or flavor=='cb' or flavor=='c+cb': 
                if _Q2 < conf['aux'].mc2: _Q2=conf['aux'].mc2+1
                else:                     _Q2=Q2
            elif flavor=='b' or flavor=='bb' or flavor=='b+bb':
                if _Q2 < conf['aux'].mb2: _Q2=conf['aux'].mb2+1
                else:                     _Q2=Q2
            else:                         _Q2=Q2
            if  flavor=='u+ub':
                func=lambda x: ff.get_xF(x,_Q2,'u')+ff.get_xF(x,_Q2,'ub')
            elif flavor=='d+db':
                func=lambda x: ff.get_xF(x,_Q2,'d')+ff.get_xF(x,_Q2,'db')
            elif flavor=='s+sb':
                func=lambda x: ff.get_xF(x,_Q2,'s')+ff.get_xF(x,_Q2,'sb')
            elif flavor=='c+cb':
                func=lambda x: ff.get_xF(x,_Q2,'c')+ff.get_xF(x,_Q2,'cb')
            elif flavor=='b+bb':
                func=lambda x: ff.get_xF(x,_Q2,'b')+ff.get_xF(x,_Q2,'bb')
            else:
                func=lambda x: ff.get_xF(x,_Q2,flavor)

            XF[flavor].append(np.array([func(x) for x in X]))
        cnt += 1

    #test = np.std(XF['u'],axis=0)
    #print(test)    
    print
    checkdir('%s/data' % wdir)
    filename = '%s/data/ff%s-Q2=%3.5f.dat' % (wdir, had, Q2)
    save({'X': X, 'Q2': Q2, 'XF': XF},filename) 
    print('Saving data to %s'%filename)

    print()

def plot_xf_pion(wdir,Q2=100,mode=1,logx=False,logy=False):
   
    
    nrows,ncols=3,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    fflabel = 'ffpion'
    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('%s not in active distribution' % fflabel)
            return

    filename = '%s/data/%s-Q2=%3.5f.dat' % (wdir, fflabel,Q2)
    #--load data if it exists
    try:
        data = load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,fflabel[2:],Q2=Q2)
        data = load(filename)
    
    hand = {}

    flavs = ['u','d','s','ub','db','sb','c','b','g']

    JAM20 = lhapdf.mkPDFs('JAM20-SIDIS_FF_pion_nlo')
    MAP10 = lhapdf.mkPDFs('MAPFF10NNLOPIp')

    for flav in flavs:

        X=data['X']
        if   flav=='u':  ax = axs[1]
        elif flav=='d':  ax = axs[2]
        elif flav=='s':  ax = axs[3]
        elif flav=='ub': ax = axs[4]
        elif flav=='db': ax = axs[5]
        elif flav=='sb': ax = axs[6]
        elif flav=='c':  ax = axs[7]
        elif flav=='b':  ax = axs[8]
        elif flav=='g':  ax = axs[9]

        #--plot average and standard deviation
        if mode==0:
            for i in range(len(data['XF'][flav])):
                thy = ax.plot(X,data['XF'][flav][i],color='red',alpha=0.2,zorder=1.2)
        if mode==1:
            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)
            thy = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=0.9,zorder=1.2)
        if mode==2:
            ci_min, ci_max = core.get_ci(data['XF'][flav])
            thy = ax.fill_between(X,ci_min,ci_max,color='red',alpha=0.9,zorder=1.2)

        #--plot JAM20
        ff = np.array([JAM20[i].xfxQ2(flav_dict[flav],X,Q2*np.ones(len(X))) for i in range(len(JAM20))])
        mean = np.mean(ff,axis=0)
        std  = np.std (ff,axis=0)

        color = 'blue'
        alpha = 1.0
        hand['JAM20'] = ax.fill_between(X[X>0.2],(mean-std)[X>0.2],(mean+std)[X>0.2],fc=color,alpha=alpha,zorder=1.3,lw=1.5)

        #--plot MAP10
        ff = np.array([MAP10[i].xfxQ2(flav_dict[flav],X,Q2*np.ones(len(X))) for i in range(len(MAP10))])
        mean = np.mean(ff,axis=0)
        std  = np.std (ff,axis=0)

        color = 'green'
        alpha = 0.8
        hand['MAP10'] = ax.fill_between(X,(mean-std),(mean+std),fc=color,alpha=alpha,zorder=1.1,lw=1.5)




   
    for i in range(N):
        if logx:
            axs[i+1].semilogx()
            axs[i+1].set_xlim(2e-2,0.9)
        else:
            axs[i+1].set_xlim(2e-2,0.9)
            axs[i+1].set_xticks([0.2,0.4,0.6,0.8])
            minorLocator = MultipleLocator(0.05)
            axs[i+1].xaxis.set_minor_locator(minorLocator)

        axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
        axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)

        axs[i+1].axhline(0,0,1,color='black')

    axs [1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axs [5].tick_params(labelbottom=False)
    axs [6].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axs [3].tick_params(labelleft=False)
    axs [5].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axs [8].tick_params(labelleft=False)
    axs [9].tick_params(labelleft=False)


    for i in [1,2,3,4,5,6,7,8,9]:
        if logy:
            axs[i].semilogy()
            axs[i].set_ylim(2e-5,5)
        else:
            axs[i].set_ylim(-0.1,1.60)
            axs[i].set_yticks([0,0.5,1.0,1.5])
            axs[i].set_yticklabels([r'$0$',r'$0.5$',r'$1.0$',r'$1.5$'])
            minorLocator = MultipleLocator(0.10)
            axs[i].yaxis.set_minor_locator(minorLocator)
   

    axs[7].set_xlabel(r'\boldmath$z$',size=40)   
    axs[7].xaxis.set_label_coords(0.97,0.00)
    axs[8].set_xlabel(r'\boldmath$z$',size=40)   
    axs[8].xaxis.set_label_coords(0.97,0.00)
    axs[9].set_xlabel(r'\boldmath$z$',size=40)   
    axs[9].xaxis.set_label_coords(0.97,0.00)


    axs[1].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{u}$'       , transform=axs[1].transAxes,size=40)
    axs[2].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{d}$'       , transform=axs[2].transAxes,size=40)
    axs[3].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{s}$'       , transform=axs[3].transAxes,size=40)
    axs[4].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{\bar{u}}$' , transform=axs[4].transAxes,size=40)
    axs[5].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{\bar{d}}$' , transform=axs[5].transAxes,size=40)
    axs[6].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{\bar{s}}$' , transform=axs[6].transAxes,size=40)
    axs[7].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{c}$'       , transform=axs[7].transAxes,size=40)
    axs[8].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{b}$'       , transform=axs[8].transAxes,size=40)
    axs[9].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{g}$'       , transform=axs[9].transAxes,size=40)

    if logy: axs[4].text(0.75,0.85,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[4].transAxes,size=25)
    else:    axs[4].text(0.05,0.05,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[4].transAxes,size=25)

    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy)
    handles.append(hand['JAM20'])
    handles.append(hand['MAP10'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{JAM20-SIDIS}')
    labels.append(r'\textrm{MAPFF1.0}')

    legend1 = axs[3].legend(handles,labels,loc=(0.00,0.00),fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5)
    axs[3].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.03)

    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/ff-pion-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
    if logx: filename+='-logx'
    if logy: filename+='-logy'
    filename += '.png'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

    if logx==False and logy==False:
        plot_pion_CSV(wdir,Q2,mode)

def plot_xf_kaon(wdir,Q2=100,mode=1,logx=False,logy=False):
   
    nrows,ncols=3,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    fflabel = 'ffkaon'
    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('%s not in active distribution' % fflabel)
            return

    filename = '%s/data/%s-Q2=%3.5f.dat' % (wdir, fflabel,Q2)
    #--load data if it exists
    try:
        data = load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,fflabel[2:],Q2=Q2)
        data = load(filename)


    hand = {}

    flavs = ['u','d','s','ub','db','sb','c','b','g']

    JAM20 = QPDCALC('JAM20-SIDIS_FF_kaon_nlo',ismc=True)
    MAP10 = QPDCALC('MAPFF10NNLOKAp',ismc=True)


    for flav in flavs:

        X=data['X']
        if   flav=='u':  ax = axs[1]
        elif flav=='d':  ax = axs[2]
        elif flav=='s':  ax = axs[3]
        elif flav=='ub': ax = axs[4]
        elif flav=='db': ax = axs[5]
        elif flav=='sb': ax = axs[6]
        elif flav=='c':  ax = axs[7]
        elif flav=='b':  ax = axs[8]
        elif flav=='g':  ax = axs[9]


        #--plot average and standard deviation
        if mode==0:
            for i in range(len(data['XF'][flav])):
                thy = ax.plot(X,data['XF'][flav][i],color='red',alpha=0.2,zorder=1.2)
        if mode==1:
            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)
            thy = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=0.9,zorder=1.2)
        if mode==2:
            ci_min, ci_max = core.get_ci(data['XF'][flav])
            thy = ax.fill_between(X,ci_min,ci_max,color='red',alpha=0.9,zorder=1.2)

        #--plot JAM20
        ff = JAM20.get_xpdf(flav,X,Q2=Q2) 

        color = 'blue'
        alpha = 1.0
        hand['JAM20'] = ax.fill_between(X[X>0.2],ff['xfmin'][X>0.2],ff['xfmax'][X>0.2],fc=color,alpha=alpha,lw=1.5,zorder=1.3)

        #--plot MAP10
        ff = MAP10.get_xpdf(flav,X,Q2=Q2) 

        color = 'green'
        alpha = 0.8
        hand['MAP10'] = ax.fill_between(X,ff['xfmin'],ff['xfmax'],fc=color,alpha=alpha,zorder=1.1,lw=1.5)



   
    for i in range(N):
        if logx:
            axs[i+1].semilogx()
            axs[i+1].set_xlim(2e-2,0.9)
        else:
            axs[i+1].set_xlim(2e-2,0.9)
            axs[i+1].set_xticks([0.2,0.4,0.6,0.8])
            minorLocator = MultipleLocator(0.05)
            axs[i+1].xaxis.set_minor_locator(minorLocator)

        axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
        axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)

    axs [1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axs [5].tick_params(labelbottom=False)
    axs [6].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axs [3].tick_params(labelleft=False)
    axs [5].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axs [8].tick_params(labelleft=False)
    axs [9].tick_params(labelleft=False)


    for i in [1,2,3,4,5,6,7,8,9]:
        if logy:
            axs[i].semilogy()
            axs[i].set_ylim(2e-5,0.9)
        else:
            axs[i].set_ylim(0.00,0.60)
            axs[i].set_yticks([0.1,0.2,0.3,0.4,0.5])
            minorLocator = MultipleLocator(0.05)
            axs[i].yaxis.set_minor_locator(minorLocator)


    axs[7].set_xlabel(r'\boldmath$z$',size=40)   
    axs[7].xaxis.set_label_coords(0.97,0.00)
    axs[8].set_xlabel(r'\boldmath$z$',size=40)   
    axs[8].xaxis.set_label_coords(0.97,0.00)
    axs[9].set_xlabel(r'\boldmath$z$',size=40)   
    axs[9].xaxis.set_label_coords(0.97,0.00)

    axs[1].text(0.80,0.83,r'\boldmath$z D^{K^+}_{u}$'       , transform=axs[1].transAxes,size=40)
    axs[2].text(0.80,0.83,r'\boldmath$z D^{K^+}_{d}$'       , transform=axs[2].transAxes,size=40)
    axs[3].text(0.80,0.83,r'\boldmath$z D^{K^+}_{s}$'       , transform=axs[3].transAxes,size=40)
    axs[4].text(0.80,0.83,r'\boldmath$z D^{K^+}_{\bar{u}}$' , transform=axs[4].transAxes,size=40)
    axs[5].text(0.80,0.83,r'\boldmath$z D^{K^+}_{\bar{d}}$' , transform=axs[5].transAxes,size=40)
    if logy: axs[6].text(0.05,0.05,r'\boldmath$z D^{K^+}_{\bar{s}}$' , transform=axs[6].transAxes,size=40)
    else:    axs[6].text(0.80,0.83,r'\boldmath$z D^{K^+}_{\bar{s}}$' , transform=axs[6].transAxes,size=40)
    axs[7].text(0.80,0.83,r'\boldmath$z D^{K^+}_{c}$'       , transform=axs[7].transAxes,size=40)
    axs[8].text(0.80,0.83,r'\boldmath$z D^{K^+}_{b}$'       , transform=axs[8].transAxes,size=40)
    axs[9].text(0.80,0.83,r'\boldmath$z D^{K^+}_{g}$'       , transform=axs[9].transAxes,size=40)

    if logy: axs[4].text(0.05,0.05,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[4].transAxes,size=25)
    else:    axs[4].text(0.07,0.85,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[4].transAxes,size=25)

    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy)
    handles.append(hand['JAM20'])
    handles.append(hand['MAP10'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{JAM20-SIDIS}')
    labels.append(r'\textrm{MAPFF1.0}')

    if logy: loc='lower left'
    else:    loc='upper left'
    legend1 = axs[3].legend(handles,labels,loc=loc,fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5)
    axs[3].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.03)

    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/ff-kaon-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
    if logx: filename+='-logx'
    if logy: filename+='-logy'
    filename += '.png'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

def plot_xf_hadron(wdir,Q2=100,mode=1,logx=False,logy=False):
   
    nrows,ncols=3,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    fflabel = 'ffhadron'
    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('%s not in active distribution' % fflabel)
            return

    filename = '%s/data/%s-Q2=%3.5f.dat' % (wdir, fflabel,Q2)
    #--load data if it exists
    try:
        data = load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,fflabel[2:],Q2=Q2)
        data = load(filename)

    hand = {}

    flavs = ['u','d','s','ub','db','sb','c','b','g']

    JAM20 = QPDCALC('JAM20-SIDIS_FF_hadron_nlo',ismc=True)

    for flav in flavs:

        X=data['X']
        if   flav=='u':  ax = axs[1]
        elif flav=='d':  ax = axs[2]
        elif flav=='s':  ax = axs[3]
        elif flav=='ub': ax = axs[4]
        elif flav=='db': ax = axs[5]
        elif flav=='sb': ax = axs[6]
        elif flav=='c':  ax = axs[7]
        elif flav=='b':  ax = axs[8]
        elif flav=='g':  ax = axs[9]


        #--plot average and standard deviation
        if mode==0:
            for i in range(len(data['XF'][flav])):
                thy = ax.plot(X,data['XF'][flav][i],color='red',alpha=0.2,zorder=1)
        if mode==1:
            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)
            thy = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=0.9,zorder=1.2)
        if mode==2:
            ci_min, ci_max = core.get_ci(data['XF'][flav])
            thy = ax.fill_between(X,ci_min,ci_max,color='red',alpha=0.9,zorder=1.2)

        #--plot JAM20
        ff = JAM20.get_xpdf(flav,X,Q2=Q2) 

        color = 'blue'
        alpha = 1.0
        hand['JAM20'] = ax.fill_between(X[X>0.2],ff['xfmin'][X>0.2],ff['xfmax'][X>0.2],fc=color,alpha=alpha,zorder=1.3,lw=1.5)
        #hand['JAM20'] = ax.fill_between(X,ff['xfmin'],ff['xfmax'],fc=color,alpha=alpha,zorder=5,ec='lightgray',lw=1.5)
        #ax .plot(X,ff['xfmax'],color=ec,alpha=0.4,zorder=5,lw=2.0)
        #ax .plot(X,ff['xfmin'],color=ec,alpha=0.4,zorder=5,lw=2.0)




   
    for i in range(N):
        if logx:
            axs[i+1].semilogx()
            axs[i+1].set_xlim(2e-2,0.9)
        else:
            axs[i+1].set_xlim(2e-2,0.9)
            axs[i+1].set_xticks([0.2,0.4,0.6,0.8])
            minorLocator = MultipleLocator(0.05)
            axs[i+1].xaxis.set_minor_locator(minorLocator)

        axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
        axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)



    axs [1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axs [5].tick_params(labelbottom=False)
    axs [6].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axs [3].tick_params(labelleft=False)
    axs [5].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axs [8].tick_params(labelleft=False)
    axs [9].tick_params(labelleft=False)


    #for i in [1,2,3]:
    #    axs[i].set_ylim(0,0.90)

    #    axs[i].set_yticks([0.2,0.4,0.6,0.8])
    #    minorLocator = MultipleLocator(0.05)
    #    axs[i].yaxis.set_minor_locator(minorLocator)
   
    #for i in [4,5,6]:
    #    axs[i].set_ylim(0,0.85)

    #    axs[i].set_yticks([0.2,0.4,0.6,0.8])
    #    minorLocator = MultipleLocator(0.05)
    #    axs[i].yaxis.set_minor_locator(minorLocator)

    #for i in [7,8,9]:
    #    axs[i].set_ylim(0,0.85)

    #    axs[i].set_yticks([0.2,0.4,0.6,0.8])
    #    minorLocator = MultipleLocator(0.05)
    #    axs[i].yaxis.set_minor_locator(minorLocator)

    for i in [1,2,3,4,5,6,7,8,9]:
        if logy:
            axs[i].semilogy()
            axs[i].set_ylim(2e-4,5)
        else:
            axs[i].set_ylim(0,2.0)
            axs[i].set_yticks([0.5,1.0,1.5])
            minorLocator = MultipleLocator(0.10)
            axs[i].yaxis.set_minor_locator(minorLocator)

    axs[7].set_xlabel(r'\boldmath$z$',size=40)   
    axs[7].xaxis.set_label_coords(0.97,0.00)
    axs[8].set_xlabel(r'\boldmath$z$',size=40)   
    axs[8].xaxis.set_label_coords(0.97,0.00)
    axs[9].set_xlabel(r'\boldmath$z$',size=40)   
    axs[9].xaxis.set_label_coords(0.97,0.00)

    axs[1].text(0.80,0.83,r'\boldmath$z D^{h^+}_{u}$'       , transform=axs[1].transAxes,size=40)
    axs[2].text(0.80,0.83,r'\boldmath$z D^{h^+}_{d}$'       , transform=axs[2].transAxes,size=40)
    axs[3].text(0.80,0.83,r'\boldmath$z D^{h^+}_{s}$'       , transform=axs[3].transAxes,size=40)
    axs[4].text(0.80,0.83,r'\boldmath$z D^{h^+}_{\bar{u}}$' , transform=axs[4].transAxes,size=40)
    axs[5].text(0.80,0.83,r'\boldmath$z D^{h^+}_{\bar{d}}$' , transform=axs[5].transAxes,size=40)
    axs[6].text(0.80,0.83,r'\boldmath$z D^{h^+}_{\bar{s}}$' , transform=axs[6].transAxes,size=40)
    axs[7].text(0.80,0.83,r'\boldmath$z D^{h^+}_{c}$'       , transform=axs[7].transAxes,size=40)
    axs[8].text(0.80,0.83,r'\boldmath$z D^{h^+}_{b}$'       , transform=axs[8].transAxes,size=40)
    axs[9].text(0.80,0.83,r'\boldmath$z D^{h^+}_{g}$'       , transform=axs[9].transAxes,size=40)

    if logy: axs[4].text(0.02,0.02,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[4].transAxes,size=25)
    else:    axs[4].text(0.02,0.02,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[4].transAxes,size=25)


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy)
    handles.append(hand['JAM20'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{JAM20-SIDIS}')

    if logy: loc = 'lower left'
    else:    loc = 'upper left'
    legend1 = axs[3].legend(handles,labels,loc=loc,fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5)
    axs[3].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.03)

    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/ff-hadron-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
    if logx: filename+='-logx'
    if logy: filename+='-logy'
    filename += '.png'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)


#--charge symmetry violation
def plot_pion_CSV(wdir,Q2=100,mode=1):
   

    nrows,ncols=1,1
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)


    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    fflabel = 'ffpion'
    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('%s not in active distribution' % fflabel)
            return

    filename = '%s/data/%s-Q2=%3.5f.dat' % (wdir, fflabel,Q2)
    #--load data if it exists
    try:
        data = load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,fflabel[2:],Q2=Q2)
        data = load(filename)

    hand = {}

    MAP10 = lhapdf.mkPDFs('MAPFF10NNLOPIp')

    ax = axs[1]
    X=data['X']

    u  = np.array(data['XF']['u'])
    db = np.array(data['XF']['db'])

    #--no filtering
    ff = (u-db)/(u+db)
    
    #--with filtering
    #_num = u - db
    #_den = u + db
    #N0 = np.shape(_den)[0] 
    #num, den = [],[]
    #for i in range(len(_den)):
    #    flag = False
    #    for k in range(len(_den[i])):
    #        if _den[i][k] <= 0: flag = True
    #    if flag==False:
    #        num.append(_num[i])
    #        den.append(_den[i])

    #N1 = np.shape(den)[0]
    #print(N0,N1)
    #num, den = np.array(num), np.array(den)
    #ff = num/den
    

    #--plot average and standard deviation
    if mode==0:
        for i in range(len(ff)):
            thy = ax.plot(X,ff[i],color='red',alpha=0.2,zorder=2)
    if mode==1:
        mean = np.mean(ff,axis=0)
        std = np.std(ff,axis=0)
        thy = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
    if mode==2:
        ci_min, ci_max = core.get_ci(ff)
        thy = ax.fill_between(X,ci_min,ci_max,color='red',alpha=0.9,zorder=1.2)

    #--plot MAP10
    u  = np.array([MAP10[i].xfxQ2( 2,X,Q2*np.ones(len(X))) for i in range(len(MAP10))])
    db = np.array([MAP10[i].xfxQ2(-1,X,Q2*np.ones(len(X))) for i in range(len(MAP10))])
    ff = (u-db)/(u+db)
    mean = np.mean(ff,axis=0)
    std  = np.std (ff,axis=0)

    color = 'green'
    alpha = 0.8
    hand['MAP10'] = ax.fill_between(X,(mean-std),(mean+std),fc=color,alpha=alpha,zorder=4,lw=1.5)




   
    for i in range(N):
        axs[i+1].set_xlim(0.02,0.9)

        axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
        axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
        axs[i+1].set_xticks([0.2,0.4,0.6,0.8])

        axs[i+1].axhline(0,0,1,color='black')
        minorLocator = MultipleLocator(0.05)
        axs[i+1].xaxis.set_minor_locator(minorLocator)

    #axs [1].tick_params(labelbottom=False)
    #axs [2].tick_params(labelbottom=False)
    #axs [3].tick_params(labelbottom=False)
    #axs [4].tick_params(labelbottom=False)
    #axs [5].tick_params(labelbottom=False)
    #axs [6].tick_params(labelbottom=False)


    #axs [2].tick_params(labelleft=False)
    #axs [3].tick_params(labelleft=False)
    #axs [5].tick_params(labelleft=False)
    #axs [6].tick_params(labelleft=False)
    #axs [8].tick_params(labelleft=False)
    #axs [9].tick_params(labelleft=False)


    for i in [1]:
        axs[i].set_ylim(-1.1,1.1)

        axs[i].set_yticks([-1.0,-0.5,0,0.5,1.0])
        axs[i].set_yticklabels([r'$-1.0$',r'$-0.5$',r'$0$',r'$0.5$',r'$1.0$'])
        minorLocator = MultipleLocator(0.1)
        axs[i].yaxis.set_minor_locator(minorLocator)
   

    axs[1].set_xlabel(r'\boldmath$z$',size=40)   
    axs[1].xaxis.set_label_coords(0.95,0.00)

    #axs[1].text(0.10,0.10,r'\boldmath$D^{\pi^+}_{(u-\bar{d})/(u+\bar{d})}$'       , transform=axs[1].transAxes,size=45)
    axs[1].text(0.10,0.10,r'\boldmath$R_{\rm CSV}$'       , transform=axs[1].transAxes,size=60)

    axs[1].text(0.03,0.92,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[1].transAxes,size=20)

    #blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy)
    handles.append(hand['MAP10'])

    labels.append(r'\textrm{\textbf{JAM25 (CSV)}}')
    labels.append(r'\textrm{MAP1.0}')

    axs[1].legend(handles,labels,loc=(0.40,0.00),fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/ff-pion-CSV-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
    filename+='.png'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

#--SU3 violation
def plot_pion_SU3(wdir,Q2=100,mode=1):
   

    nrows,ncols=1,1
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)


    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    fflabel = 'ffpion'
    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('%s not in active distribution' % fflabel)
            return

    filename = '%s/data/%s-Q2=%3.5f.dat' % (wdir, fflabel,Q2)
    #--load data if it exists
    try:
        data = load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,fflabel[2:],Q2=Q2)
        data = load(filename)

    hand = {}

    MAP10 = lhapdf.mkPDFs('MAPFF10NNLOPIp')

    ax = axs[1]
    X=data['X']

    s  = np.array(data['XF']['s'])
    d  = np.array(data['XF']['d'])
    ff = (s-d)/(s+d)

    #--plot average and standard deviation
    if mode==0:
        for i in range(len(ff)):
            thy = ax.plot(X,ff[i],color='red',alpha=0.2,zorder=2)
    if mode==1:
        mean = np.mean(ff,axis=0)
        std = np.std(ff,axis=0)
        thy = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
    if mode==2:
        ci_min, ci_max = core.get_ci(ff)
        thy = ax.fill_between(X,ci_min,ci_max,color='red',alpha=0.9,zorder=1.2)

    #--plot MAP10
    s  = np.array([MAP10[i].xfxQ2( 3,X,Q2*np.ones(len(X))) for i in range(len(MAP10))])
    d  = np.array([MAP10[i].xfxQ2( 1,X,Q2*np.ones(len(X))) for i in range(len(MAP10))])
    ff = (s-d)/(s+d)
    mean = np.mean(ff,axis=0)
    std  = np.std (ff,axis=0)

    color = 'green'
    alpha = 0.8
    hand['MAP10'] = ax.fill_between(X,(mean-std),(mean+std),fc=color,alpha=alpha,zorder=4,lw=1.5)




   
    for i in range(N):
        axs[i+1].set_xlim(0.02,1.0)

        axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
        axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
        axs[i+1].set_xticks([0.2,0.4,0.6,0.8])

        axs[i+1].axhline(0,0,1,color='black')
        minorLocator = MultipleLocator(0.05)
        axs[i+1].xaxis.set_minor_locator(minorLocator)

    #axs [1].tick_params(labelbottom=False)
    #axs [2].tick_params(labelbottom=False)
    #axs [3].tick_params(labelbottom=False)
    #axs [4].tick_params(labelbottom=False)
    #axs [5].tick_params(labelbottom=False)
    #axs [6].tick_params(labelbottom=False)


    #axs [2].tick_params(labelleft=False)
    #axs [3].tick_params(labelleft=False)
    #axs [5].tick_params(labelleft=False)
    #axs [6].tick_params(labelleft=False)
    #axs [8].tick_params(labelleft=False)
    #axs [9].tick_params(labelleft=False)


    for i in [1]:
        axs[i].set_ylim(-1.1,1.1)

        axs[i].set_yticks([-1.0,-0.5,0,0.5,1.0])
        axs[i].set_yticklabels([r'$-1.0$',r'$-0.5$',r'$0$',r'$0.5$',r'$1.0$'])
        minorLocator = MultipleLocator(0.1)
        axs[i].yaxis.set_minor_locator(minorLocator)
   

    axs[1].set_xlabel(r'\boldmath$z$',size=40)   
    axs[1].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.05,0.10,r'\boldmath$D^{\pi^+}_{(s-d)/(s+d)}$'       , transform=axs[1].transAxes,size=45)

    axs[1].text(0.40,0.90,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[1].transAxes,size=25)

    #blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy)
    handles.append(hand['MAP10'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{MAP}')

    axs[1].legend(handles,labels,loc='upper left',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/ff-pion-SU3-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
    filename+='.png'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)






