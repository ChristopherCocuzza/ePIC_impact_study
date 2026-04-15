import sys,os
import numpy as np

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py

#--from scipy stack 
from scipy.integrate import quad

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.qpdlib.qpdcalc import QPDCALC
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

FLAV=[]
FLAV.append('uv')
FLAV.append('dv')

FLAV2=[]
FLAV2.append('uv')
FLAV2.append('dv')
FLAV2.append('uv0')
FLAV2.append('dv0')
FLAV2.append('uv1')
FLAV2.append('dv1')

def gen_xf(wdir,Q2=None):
    
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    if Q2==None: Q2 = conf['Q20']
    print('\ngenerating offshell pdf at Q2 = %s from %s'%(Q2,wdir))
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'off pdf' not in conf['steps'][istep]['active distributions']:
        if 'off pdf' not in conf['steps'][istep]['passive distributions']:
                print('off pdf is not an active or passive distribution')
                return 

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['off pdf']
    #--setup kinematics
    X=10**np.linspace(-6,-1,200)
    X=np.append(X,np.linspace(0.1,0.99,200))

    pdf.evolve(Q2)

    #--compute XF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in XF:  XF[flav]=[]
            if flav=='uv':
                 func=lambda x: pdf.get_xF(x,Q2,'uv')
            elif flav=='dv':
                 func=lambda x: pdf.get_xF(x,Q2,'dv')
            elif flav=='d/u':
                 func=lambda x: pdf.get_xF(x,Q2,'d')/pdf.get_xF(x,Q2,'u')
            elif flav=='db+ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') + pdf.get_xF(x,Q2,'ub')
            elif flav=='db-ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') - pdf.get_xF(x,Q2,'ub')
            elif flav=='db/ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') / pdf.get_xF(x,Q2,'ub')
            elif flav=='s+sb':
                 func=lambda x: pdf.get_xF(x,Q2,'s') + pdf.get_xF(x,Q2,'sb')
            elif flav=='s-sb':
                 func=lambda x: pdf.get_xF(x,Q2,'s') - pdf.get_xF(x,Q2,'sb')
            elif flav=='Rs':
                 func=lambda x: (pdf.get_xF(x,Q2,'s') + pdf.get_xF(x,Q2,'sb'))\
                                /(pdf.get_xF(x,Q2,'db') + pdf.get_xF(x,Q2,'ub'))
            else:
                 func=lambda x: pdf.get_xF(x,Q2,flav) 

            XF[flav].append(np.array([func(x) for x in X]))

    print() 
    checkdir('%s/data'%wdir)
    filename='%s/data/off-pdf-Q2=%3.5f.dat'%(wdir,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_xf(wdir,Q2=None,mode=0):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=3,2
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax21=py.subplot(nrows,ncols,3)
    ax22=py.subplot(nrows,ncols,4)
    ax31=py.subplot(nrows,ncols,5)
    ax32=py.subplot(nrows,ncols,6)

    cmap = matplotlib.cm.get_cmap('hot')

    hand = {}
    thy  = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if Q2==None: Q2 = conf['Q20']

    #idx = [10041,10051]
    idx = [10041,10050,10051]

    #scale = classifier.get_scale(wdir,'idis',idx)

    if 'off pdf' not in conf['steps'][istep]['active distributions']:
        if 'off pdf' not in conf['steps'][istep]['passive distributions']:
                print('off pdf is not an active or passive distribution')
                return 

    #--load data if it exists
    filename='%s/data/off-pdf-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,Q2)
        
    #--load pdf data
    filename='%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2)
    pdfs=load(filename)

    replicas=core.get_replicas(wdir)
    #cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    #best_cluster=cluster_order[0]

    X=data['X']

    flavs = ['u','d']
    for flav in flavs:
        if flav=='u': pdf = np.array(data['XF']['uv'])
        if flav=='d': pdf = np.array(data['XF']['dv'])
        
        mean = np.mean(pdf,axis=0)
        std  = np.std (pdf,axis=0)

        if   flav=='u': ax = ax11
        elif flav=='d': ax = ax12
        else: continue
        if   flav=='u':    color = 'firebrick'
        elif flav=='d':    color = 'darkgreen'

        #--plot each replica
        if mode==0:
            for i in range(len(pdf)):
                hand[flav] ,= ax.plot(X,pdf[i],color='red',alpha=0.2)
 
        #--plot average and standard deviation
        if mode==1:
            hand[flav]  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=0.8,zorder=5)

    uv = np.array(data['XF']['uv'])
    dv = np.array(data['XF']['dv'])
    p  = uv + dv
    m  = uv - dv

    meanp = np.mean(p,axis=0)
    stdp  = np.std (p,axis=0)
    meanm = np.mean(m,axis=0)
    stdm  = np.std (m,axis=0)

    #--plot each replica
    if mode==0:
        for i in range(len(p)):
            hand[flav] ,= ax21.plot(X,p[i],color='red',alpha=0.2)
            hand[flav] ,= ax22.plot(X,m[i],color='red',alpha=0.2)
 
    #--plot average and standard deviation
    if mode==1:
        hand[flav]  = ax21.fill_between(X,meanp-stdp,meanp+stdp,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax22.fill_between(X,meanm-stdm,meanm+stdm,color='red',alpha=0.8,zorder=5)

    uvon = np.array(pdfs['XF']['uv'])
    dvon = np.array(pdfs['XF']['dv'])
    ratu = uv/uvon
    ratd = dv/dvon

    meanu = np.mean(ratu,axis=0)
    stdu  = np.std (ratu,axis=0)
    meand = np.mean(ratd,axis=0)
    stdd  = np.std (ratd,axis=0)

    #--plot each replica
    if mode==0:
        for i in range(len(ratu)):
            hand[flav] ,= ax31.plot(X,ratu[i],color='red',alpha=0.2)
            hand[flav] ,= ax32.plot(X,ratd[i],color='red',alpha=0.2)
 
    #--plot average and standard deviation
    if mode==1:
        hand[flav]  = ax31.fill_between(X,meanu-stdu,meanu+stdu,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax32.fill_between(X,meand-stdd,meand+stdd,color='red',alpha=0.8,zorder=5)


    for ax in [ax11,ax12,ax21,ax22,ax31,ax32]:
          ax.set_xlim(0,0.9)
          ax.axhline(0.0,ls='--',color='black',alpha=0.5)
          minorLocator = MultipleLocator(0.04)
          majorLocator = MultipleLocator(0.2)
          ax.xaxis.set_minor_locator(minorLocator)
          ax.xaxis.set_major_locator(majorLocator)
          ax.set_xticks([0.2,0.4,0.6,0.8])
            
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)

    if mode==0:
        ax11.set_ylim(-0.95,0.95)
        ax12.set_ylim(-0.95,0.95)
        ax21.set_ylim(-0.95,0.95)
        ax22.set_ylim(-0.95,0.95)
        ax31.set_ylim(-1.10,1.10)
        ax32.set_ylim(-1.10,1.10)
    if mode==1:
        ax11.set_ylim(-0.25,0.25)
        ax12.set_ylim(-0.25,0.25)
        ax21.set_ylim(-0.35,0.35)
        ax22.set_ylim(-0.35,0.35)
        ax31.set_ylim(-1.10,1.10)
        ax32.set_ylim(-1.10,1.10)


    for ax in [ax31,ax32]:
        ax.set_xlabel(r'\boldmath$x$' ,size=30)
        ax.xaxis.set_label_coords(0.98,0.00)

    if Q2 == 1.27**2: ax11.text(0.65,0.85,r'$Q^2 = m_c^2$'                                  , transform=ax11.transAxes,size=30)
    else:             ax11.text(0.65,0.85,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=30)


    ax11.text(0.05,0.07,r'\boldmath$x \delta u_v$'     ,transform=ax11.transAxes,size=30)
    ax12.text(0.05,0.07,r'\boldmath$x \delta d_v$'     ,transform=ax12.transAxes,size=30)
    ax21.text(0.05,0.07,r'\boldmath$x (\delta u_v + \delta d_v)$' ,transform=ax21.transAxes,size=30)
    ax22.text(0.05,0.07,r'\boldmath$x (\delta u_v - \delta d_v)$' ,transform=ax22.transAxes,size=30)
    ax31.text(0.05,0.07,r'\boldmath$  \delta u_v/u_v$'     ,transform=ax31.transAxes,size=30)
    ax32.text(0.05,0.07,r'\boldmath$  \delta d_v/d_v$'     ,transform=ax32.transAxes,size=30)

    #sm   = py.cm.ScalarMappable(cmap=cmap)
    #sm.set_array([])
    #cax = fig.add_axes([0.72,0.92,0.25,0.05])
    #cax.tick_params(axis='both',which='both',labelsize=20,direction='in')
    #cax.xaxis.set_label_coords(0.65,-0.5)
    #cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    #cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=30)


    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.20)

    filename = '%s/gallery/off-pdfs-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)



def gen_xf2(wdir,Q2=None):
    
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    if Q2==None: Q2 = conf['Q20']
    print('\ngenerating offshell pdf at Q2 = %s from %s'%(Q2,wdir))
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'off pdf0' not in conf['steps'][istep]['active distributions']:
        if 'off pdf0' not in conf['steps'][istep]['passive distributions']:
                print('off pdf0 is not an active or passive distribution')
                return 

    if 'off pdf1' not in conf['steps'][istep]['active distributions']:
        if 'off pdf1' not in conf['steps'][istep]['passive distributions']:
                print('off pdf1 is not an active or passive distribution')
                return 

    if 'offshell model' in conf: model = conf['offshell model']
    else:                        model = 2

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf0=conf['off pdf0']
    pdf1=conf['off pdf1']
    #--setup kinematics
    X=10**np.linspace(-6,-1,200)
    X=np.append(X,np.linspace(0.1,0.99,200))

    pdf0.evolve(Q2)
    pdf1.evolve(Q2)

    #--compute XF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV2:
            if flav not in XF:  XF[flav]=[]
            if flav in ['uv0','dv0']:
                 func=lambda x: pdf0.get_xF(x,Q2,flav[:2])
            elif flav in ['uv1']:
                 func=lambda x: pdf1.get_xF(x,Q2,flav[:2])
            elif flav in ['dv1']:
                 if model==2: func=lambda x:  pdf1.get_xF(x,Q2,flav[:2])
                 if model==3: func=lambda x: -pdf1.get_xF(x,Q2,'uv')
            elif flav in ['uv']:
                 func=lambda x: pdf0.get_xF(x,Q2,flav[:2]) + pdf1.get_xF(x,Q2,flav[:2])
            elif flav in ['dv']:
                 if model==2: func=lambda x: pdf0.get_xF(x,Q2,flav[:2]) + pdf1.get_xF(x,Q2,flav[:2])
                 if model==3: func=lambda x: pdf0.get_xF(x,Q2,flav[:2]) - pdf1.get_xF(x,Q2,'uv')

            XF[flav].append(np.array([func(x) for x in X]))

    print() 
    checkdir('%s/data'%wdir)
    filename='%s/data/off-pdf2-Q2=%3.5f.dat'%(wdir,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_xf2(wdir,Q2=None,mode=0):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=6,2
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax21=py.subplot(nrows,ncols,3)
    ax22=py.subplot(nrows,ncols,4)
    ax31=py.subplot(nrows,ncols,5)
    ax32=py.subplot(nrows,ncols,6)
    ax41=py.subplot(nrows,ncols,7)
    ax42=py.subplot(nrows,ncols,8)
    ax51=py.subplot(nrows,ncols,9)
    ax52=py.subplot(nrows,ncols,10)
    ax61=py.subplot(nrows,ncols,11)
    ax62=py.subplot(nrows,ncols,12)

    cmap = matplotlib.cm.get_cmap('cool')

    hand = {}
    thy  = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if Q2==None: Q2 = conf['Q20']

    #idx = [10041,10051]
    idx = [10052,10053]

    scale = classifier.get_scale(wdir,'idis',idx)

    if 'off pdf0' not in conf['steps'][istep]['active distributions']:
        if 'off pdf0' not in conf['steps'][istep]['passive distributions']:
                print('off pdf0 is not an active or passive distribution')
                return 

    if 'off pdf1' not in conf['steps'][istep]['active distributions']:
        if 'off pdf1' not in conf['steps'][istep]['passive distributions']:
                print('off pdf1 is not an active or passive distribution')
                return 

    #--load data if it exists
    filename='%s/data/off-pdf2-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,Q2)
        
    #--load pdf data
    filename='%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2)
    pdfs=load(filename)

    replicas=core.get_replicas(wdir)
    #cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    #best_cluster=cluster_order[0]

    X=data['X']

    flavs = ['uv0','dv0','uv1','dv1','uv','dv']
    for flav in flavs:
        if   flav=='uv': pdf = np.array(data['XF']['uv0']) + np.array(data['XF']['uv1'])
        elif flav=='dv': pdf = np.array(data['XF']['dv0']) + np.array(data['XF']['dv1'])
        else           : pdf = np.array(data['XF'][flav])
        
        mean = np.mean(pdf,axis=0)
        std  = np.std (pdf,axis=0)

        y0,y1 = 16,2.5
        perc_min0 = np.percentile(pdf,y0    ,axis=0)
        perc_max0 = np.percentile(pdf,100-y0,axis=0)

        perc_min1 = np.percentile(pdf,y1    ,axis=0)
        perc_max1 = np.percentile(pdf,100-y1,axis=0)

        if   flav=='uv0': ax = ax11
        elif flav=='dv0': ax = ax12
        elif flav=='uv1': ax = ax21
        elif flav=='dv1': ax = ax22
        #elif flav=='uv':  ax = ax31
        #elif flav=='dv':  ax = ax32
        else: continue
        if   'u' in flav:    color = 'firebrick'
        elif 'd' in flav:    color = 'darkgreen'

        #--plot each replica
        if mode==0:
            for i in range(len(pdf)):
                hand[flav] ,= ax.plot(X,pdf[i],color=cmap(scale[i]),alpha=0.5)
 
        #--plot average and standard deviation
        if mode==1:
            hand[flav]  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=0.8,zorder=5)

        #--plot percentiles
        if mode==2:
            hand[flav] = ax.fill_between(X,perc_min0,perc_max0,color='red',alpha=0.5,zorder=5)
            #hand[flav] = ax.fill_between(X,perc_min1,perc_max1,color='red',alpha=0.5,zorder=5)

    uv0 = np.array(data['XF']['uv0'])
    dv0 = np.array(data['XF']['dv0'])
    uv1 = np.array(data['XF']['uv1'])
    dv1 = np.array(data['XF']['dv1'])
    uv = np.array(data['XF']['uv'])
    dv = np.array(data['XF']['dv'])
    p0  = uv0 + dv0
    m0  = uv0 - dv0
    p1  = uv1 + dv1
    m1  = uv1 - dv1
    p   = uv  + dv
    m   = uv  - dv

    meanp0 = np.mean(p0,axis=0)
    stdp0  = np.std (p0,axis=0)
    meanm0 = np.mean(m0,axis=0)
    stdm0  = np.std (m0,axis=0)

    meanp1 = np.mean(p1,axis=0)
    stdp1  = np.std (p1,axis=0)
    meanm1 = np.mean(m1,axis=0)
    stdm1  = np.std (m1,axis=0)

    meanp = np.mean(p,axis=0)
    stdp  = np.std (p,axis=0)
    meanm = np.mean(m,axis=0)
    stdm  = np.std (m,axis=0)

    perc_minp00 = np.percentile(p0,y0    ,axis=0)
    perc_maxp00 = np.percentile(p0,100-y0,axis=0)
    perc_minm00 = np.percentile(m0,y0    ,axis=0)
    perc_maxm00 = np.percentile(m0,100-y0,axis=0)

    perc_minp01 = np.percentile(p0,y1    ,axis=0)
    perc_maxp01 = np.percentile(p0,100-y1,axis=0)
    perc_minm01 = np.percentile(m0,y1    ,axis=0)
    perc_maxm01 = np.percentile(m0,100-y1,axis=0)

    perc_minp10 = np.percentile(p1,y0    ,axis=0)
    perc_maxp10 = np.percentile(p1,100-y0,axis=0)
    perc_minm10 = np.percentile(m1,y0    ,axis=0)
    perc_maxm10 = np.percentile(m1,100-y0,axis=0)

    perc_minp11 = np.percentile(p1,y1    ,axis=0)
    perc_maxp11 = np.percentile(p1,100-y1,axis=0)
    perc_minm11 = np.percentile(m1,y1    ,axis=0)
    perc_maxm11 = np.percentile(m1,100-y1,axis=0)

    perc_minp0 = np.percentile(p,y0    ,axis=0)
    perc_maxp0 = np.percentile(p,100-y0,axis=0)
    perc_minm0 = np.percentile(m,y0    ,axis=0)
    perc_maxm0 = np.percentile(m,100-y0,axis=0)

    perc_minp1 = np.percentile(p,y1    ,axis=0)
    perc_maxp1 = np.percentile(p,100-y1,axis=0)
    perc_minm1 = np.percentile(m,y1    ,axis=0)
    perc_maxm1 = np.percentile(m,100-y1,axis=0)

    #--plot each replica
    if mode==0:
        for i in range(len(p)):
            hand[flav] ,= ax31.plot(X,p0[i],color=cmap(scale[i]),alpha=0.5)
            hand[flav] ,= ax32.plot(X,m0[i],color=cmap(scale[i]),alpha=0.5)
            hand[flav] ,= ax41.plot(X,p1[i],color=cmap(scale[i]),alpha=0.5)
            hand[flav] ,= ax42.plot(X,m1[i],color=cmap(scale[i]),alpha=0.5)
            #hand[flav] ,= ax61.plot(X,p [i],color=cmap(scale[i]),alpha=0.5)
            #hand[flav] ,= ax62.plot(X,m [i],color=cmap(scale[i]),alpha=0.5)
 
    #--plot average and standard deviation
    if mode==1:
        hand[flav]  = ax31.fill_between(X,meanp0-stdp0,meanp0+stdp0,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax32.fill_between(X,meanm0-stdm0,meanm0+stdm0,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax41.fill_between(X,meanp1-stdp1,meanp1+stdp1,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax42.fill_between(X,meanm1-stdm1,meanm1+stdm1,color='red',alpha=0.8,zorder=5)
        #hand[flav]  = ax61.fill_between(X,meanp -stdp ,meanp +stdp ,color='red',alpha=0.8,zorder=5)
        #hand[flav]  = ax62.fill_between(X,meanm -stdm ,meanm +stdm ,color='red',alpha=0.8,zorder=5)

    #--plot percentiles
    if mode==2:
        hand[flav]  = ax31.fill_between(X,perc_minp00,perc_maxp00,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax32.fill_between(X,perc_minm00,perc_maxm00,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax41.fill_between(X,perc_minp10,perc_maxp10,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax42.fill_between(X,perc_minm10,perc_maxm10,color='red',alpha=0.5,zorder=5)
        #hand[flav]  = ax61.fill_between(X,perc_minp0 ,perc_maxp0 ,color='red',alpha=0.5,zorder=5)
        #hand[flav]  = ax62.fill_between(X,perc_minm0 ,perc_maxm0 ,color='red',alpha=0.5,zorder=5)

        hand[flav]  = ax31.fill_between(X,perc_minp01,perc_maxp01,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax32.fill_between(X,perc_minm01,perc_maxm01,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax41.fill_between(X,perc_minp11,perc_maxp11,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax42.fill_between(X,perc_minm11,perc_maxm11,color='red',alpha=0.5,zorder=5)
        #hand[flav]  = ax61.fill_between(X,perc_minp1 ,perc_maxp1 ,color='red',alpha=0.5,zorder=5)
        #hand[flav]  = ax62.fill_between(X,perc_minm1 ,perc_maxm1 ,color='red',alpha=0.5,zorder=5)

    uvon = np.array(pdfs['XF']['uv'])
    dvon = np.array(pdfs['XF']['dv'])
    ratu0 = uv0/uvon
    ratd0 = dv0/dvon
    ratu1 = uv1/uvon
    ratd1 = dv1/dvon

    meanu0 = np.mean(ratu0,axis=0)
    stdu0  = np.std (ratu0,axis=0)
    meand0 = np.mean(ratd0,axis=0)
    stdd0  = np.std (ratd0,axis=0)
    meanu1 = np.mean(ratu1,axis=0)
    stdu1  = np.std (ratu1,axis=0)
    meand1 = np.mean(ratd1,axis=0)
    stdd1  = np.std (ratd1,axis=0)

    perc_min_u00 = np.percentile(ratu0,y0    ,axis=0)
    perc_max_u00 = np.percentile(ratu0,100-y0,axis=0)
    perc_min_u01 = np.percentile(ratu0,y1    ,axis=0)
    perc_max_u01 = np.percentile(ratu0,100-y1,axis=0)

    perc_min_d00 = np.percentile(ratd0,y0    ,axis=0)
    perc_max_d00 = np.percentile(ratd0,100-y0,axis=0)
    perc_min_d01 = np.percentile(ratd0,y1    ,axis=0)
    perc_max_d01 = np.percentile(ratd0,100-y1,axis=0)

    perc_min_u10 = np.percentile(ratu1,y0    ,axis=0)
    perc_max_u10 = np.percentile(ratu1,100-y0,axis=0)
    perc_min_u11 = np.percentile(ratu1,y1    ,axis=0)
    perc_max_u11 = np.percentile(ratu1,100-y1,axis=0)

    perc_min_d10 = np.percentile(ratd1,y0    ,axis=0)
    perc_max_d10 = np.percentile(ratd1,100-y0,axis=0)
    perc_min_d11 = np.percentile(ratd1,y1    ,axis=0)
    perc_max_d11 = np.percentile(ratd1,100-y1,axis=0)

    #--plot each replica
    if mode==0:
        for i in range(len(p)):
            hand[flav] ,= ax51.plot(X,ratu0[i],color=cmap(scale[i]),alpha=0.5)
            hand[flav] ,= ax52.plot(X,ratd0[i],color=cmap(scale[i]),alpha=0.5)
            hand[flav] ,= ax61.plot(X,ratu1[i],color=cmap(scale[i]),alpha=0.5)
            hand[flav] ,= ax62.plot(X,ratd1[i],color=cmap(scale[i]),alpha=0.5)
 
    #--plot average and standard deviation
    if mode==1:
        hand[flav]  = ax51.fill_between(X,meanu0-stdu0,meanu0+stdu0,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax52.fill_between(X,meand0-stdd0,meand0+stdd0,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax61.fill_between(X,meanu1-stdu1,meanu1+stdu1,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax62.fill_between(X,meand1-stdd1,meand1+stdd1,color='red',alpha=0.8,zorder=5)

    #--plot percentiles
    if mode==2:
        hand[flav]  = ax51.fill_between(X,perc_min_u00,perc_max_u00,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax52.fill_between(X,perc_min_d00,perc_max_d00,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax61.fill_between(X,perc_min_u10,perc_max_u10,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax62.fill_between(X,perc_min_d10,perc_max_d10,color='red',alpha=0.5,zorder=5)
                                                                 
        hand[flav]  = ax51.fill_between(X,perc_min_u01,perc_max_u01,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax52.fill_between(X,perc_min_d01,perc_max_d01,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax61.fill_between(X,perc_min_u11,perc_max_u11,color='red',alpha=0.5,zorder=5)
        hand[flav]  = ax62.fill_between(X,perc_min_d11,perc_max_d11,color='red',alpha=0.5,zorder=5)

    for ax in [ax11,ax12,ax21,ax22,ax31,ax32,ax41,ax42,ax51,ax52,ax61,ax62]:
          ax.set_xlim(0.2,0.9)
          ax.axhline(0.0,ls='--',color='black',alpha=0.5)
          minorLocator = MultipleLocator(0.04)
          majorLocator = MultipleLocator(0.2)
          ax.xaxis.set_minor_locator(minorLocator)
          ax.xaxis.set_major_locator(majorLocator)
          ax.set_xticks([0.4,0.6,0.8])
            
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=40,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=40,length=5)

    for ax in [ax12,ax22,ax32,ax42,ax52,ax62]:
        ax.tick_params(labelleft=False)

    for ax in [ax11,ax12]:
        ax.set_ylim(-0.30,0.30)
    for ax in [ax21,ax22]:
        ax.set_ylim(-0.80,0.80)
    for ax in [ax31,ax32]:
        ax.set_ylim(-0.80,0.80)
    for ax in [ax41,ax42]:
        ax.set_ylim(-0.80,0.80)
    for ax in [ax51,ax52]:
        ax.set_ylim(-4.00,4.00)
    for ax in [ax61,ax62]:
        ax.set_ylim(-4.00,4.00)


    for ax in [ax61,ax62]:
        ax.set_xlabel(r'\boldmath$x$' ,size=50)
        ax.xaxis.set_label_coords(0.98,0.00)

    if Q2 == 1.27**2: ax12.text(0.60,0.85,r'$Q^2 = m_c^2$'                                  , transform=ax12.transAxes,size=30)
    else:             ax12.text(0.60,0.85,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax12.transAxes,size=30)

    if mode==2:
        ax11.text(0.03,0.80,r'\textrm{\textbf{68/95$\%$ CI}}',transform=ax11.transAxes,size=50,zorder=5)

    ax11.text(0.70,0.07,r'\boldmath$x \delta u_0$'                 ,transform=ax11.transAxes,size=50,zorder=5)
    ax12.text(0.70,0.07,r'\boldmath$x \delta d_0$'                 ,transform=ax12.transAxes,size=50,zorder=5)
    ax21.text(0.70,0.07,r'\boldmath$x \delta u_1$'                 ,transform=ax21.transAxes,size=50,zorder=5)
    ax22.text(0.70,0.07,r'\boldmath$x \delta d_1$'                 ,transform=ax22.transAxes,size=50,zorder=5)
    ax31.text(0.40,0.07,r'\boldmath$x (\delta u_0 + \delta d_0)$'  ,transform=ax31.transAxes,size=50,zorder=5)
    ax32.text(0.40,0.07,r'\boldmath$x (\delta u_0 - \delta d_0)$'  ,transform=ax32.transAxes,size=50,zorder=5)
    ax41.text(0.40,0.07,r'\boldmath$x (\delta u_1 + \delta d_1)$'  ,transform=ax41.transAxes,size=50,zorder=5)
    ax42.text(0.40,0.07,r'\boldmath$x (\delta u_1 - \delta d_1)$'  ,transform=ax42.transAxes,size=50,zorder=5)
    ax51.text(0.07,0.07,r'\boldmath$\delta u_0/u_v$'               ,transform=ax51.transAxes,size=50,zorder=5)
    ax52.text(0.07,0.07,r'\boldmath$\delta d_0/d_v$'               ,transform=ax52.transAxes,size=50,zorder=5)
    ax61.text(0.07,0.07,r'\boldmath$\delta u_1/u_v$'               ,transform=ax61.transAxes,size=50,zorder=5)
    ax62.text(0.07,0.07,r'\boldmath$\delta d_1/d_v$'               ,transform=ax62.transAxes,size=50,zorder=5)

    if mode==0:
        sm   = py.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.10,0.84,0.20,0.03])
        cax.tick_params(axis='both',which='both',labelsize=20,direction='in')
        cax.xaxis.set_label_coords(0.65,-0.5)
        cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
        cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=30)


    py.tight_layout()
    py.subplots_adjust(hspace = 0.01, wspace = 0.01, top = 0.99, right = 0.99)

    filename = '%s/gallery/off-pdfs2-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)


def plot_dqNA(wdir,Q2=None,mode=0):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)

    cmap = matplotlib.cm.get_cmap('hot')

    hand = {}
    thy  = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if Q2==None: Q2 = conf['Q20']

    #idx = [10041,10051]
    idx = [10041,10050,10051]

    scale = classifier.get_scale(wdir,'idis',idx)

    if 'off pdf0' not in conf['steps'][istep]['active distributions']:
        if 'off pdf0' not in conf['steps'][istep]['passive distributions']:
                print('off pdf0 is not an active or passive distribution')
                return 

    if 'off pdf1' not in conf['steps'][istep]['active distributions']:
        if 'off pdf1' not in conf['steps'][istep]['passive distributions']:
                print('off pdf1 is not an active or passive distribution')
                return 

    #--load data if it exists
    filename='%s/data/off-pdf2-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,Q2)
        
    #--load pdf data
    filename='%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2)
    pdfs=load(filename)

    replicas=core.get_replicas(wdir)
    #cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    #best_cluster=cluster_order[0]

    X=data['X']

    flavs = ['uv0','dv0','uv1','dv1','uv','dv']

    q0 = np.array(data['XF']['uv0'])
    u1 = np.array(data['XF']['uv1'])
    d1 = np.array(data['XF']['dv1'])

    du_p_t = q0 + u1
    dd_p_t = q0 + d1
    dq_p_h = q0

    du_p_t_mean = np.mean(du_p_t,axis=0)
    dd_p_t_mean = np.mean(dd_p_t,axis=0)
    dq_p_h_mean = np.mean(dq_p_h,axis=0)

    du_p_t_std = np.std(du_p_t,axis=0)
    dd_p_t_std = np.std(dd_p_t,axis=0)
    dq_p_h_std = np.std(dq_p_h,axis=0)

    y0,y1 = 16,2.5
    du_p_t_perc_min0 = np.percentile(du_p_t,y0    ,axis=0)
    dd_p_t_perc_min0 = np.percentile(dd_p_t,y0    ,axis=0)
    dq_p_h_perc_min0 = np.percentile(dq_p_h,y0    ,axis=0)
    
    du_p_t_perc_max0 = np.percentile(du_p_t,100-y0,axis=0)
    dd_p_t_perc_max0 = np.percentile(dd_p_t,100-y0,axis=0)
    dq_p_h_perc_max0 = np.percentile(dq_p_h,100-y0,axis=0)

    du_p_t_perc_min1 = np.percentile(du_p_t,y1    ,axis=0)
    dd_p_t_perc_min1 = np.percentile(dd_p_t,y1    ,axis=0)
    dq_p_h_perc_min1 = np.percentile(dq_p_h,y1    ,axis=0)
    
    du_p_t_perc_max1 = np.percentile(du_p_t,100-y1,axis=0)
    dd_p_t_perc_max1 = np.percentile(dd_p_t,100-y1,axis=0)
    dq_p_h_perc_max1 = np.percentile(dq_p_h,100-y1,axis=0)

    color = 'firebrick'

    #--plot each replica
    if mode==0:
        for i in range(len(du_p_t)):
            hand['q'] ,= ax11.plot(X,du_p_t[i],color='red',alpha=0.2)
            hand['q'] ,= ax12.plot(X,dd_p_t[i],color='red',alpha=0.2)
            hand['q'] ,= ax13.plot(X,dq_p_h[i],color='red',alpha=0.2)
 
    #--plot average and standard deviation
    if mode==1:
        hand['q']  = ax11.fill_between(X,du_p_t_mean-du_p_t_std,du_p_t_mean+du_p_t_std,color='red',alpha=0.8,zorder=5)
        hand['q']  = ax12.fill_between(X,dd_p_t_mean-dd_p_t_std,dd_p_t_mean+dd_p_t_std,color='red',alpha=0.8,zorder=5)
        hand['q']  = ax13.fill_between(X,dq_p_h_mean-dq_p_h_std,dq_p_h_mean+dq_p_h_std,color='red',alpha=0.8,zorder=5)

    #--plot percentiles
    if mode==2:
        hand['q']  = ax11.fill_between(X,du_p_t_perc_min0,du_p_t_perc_max0,color='red',alpha=0.8,zorder=5)
        hand['q']  = ax12.fill_between(X,dd_p_t_perc_min0,dd_p_t_perc_max0,color='red',alpha=0.8,zorder=5)
        hand['q']  = ax13.fill_between(X,dq_p_h_perc_min0,dq_p_h_perc_max0,color='red',alpha=0.8,zorder=5)

        hand['q']  = ax11.fill_between(X,du_p_t_perc_min1,du_p_t_perc_max1,color='red',alpha=0.8,zorder=5)
        hand['q']  = ax12.fill_between(X,dd_p_t_perc_min1,dd_p_t_perc_max1,color='red',alpha=0.8,zorder=5)
        hand['q']  = ax13.fill_between(X,dq_p_h_perc_min1,dq_p_h_perc_max1,color='red',alpha=0.8,zorder=5)

    for ax in [ax11,ax12,ax13]:
        ax.set_xlim(0,0.9)
        ax.axhline(0.0,ls='--',color='black',alpha=0.5)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticks([0.2,0.4,0.6,0.8])
          
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)

        ax.set_ylim(-0.90,0.90)

    for ax in [ax12,ax13]:
        ax.tick_params(labelleft=False)

    for ax in [ax11,ax12,ax13]:
        ax.set_xlabel(r'\boldmath$x$' ,size=30)
        ax.xaxis.set_label_coords(0.98,0.00)

    if Q2 == 1.27**2: ax11.text(0.65,0.85,r'$Q^2 = m_c^2$'                                  , transform=ax11.transAxes,size=30)
    else:             ax11.text(0.65,0.85,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=30)


    ax11.text(0.05,0.08,r'\boldmath$x \delta u_{p/t} = 2 x \delta u_{p/D} = x \delta q_0 + x \delta u_1$',transform=ax11.transAxes,size=30)
    ax12.text(0.05,0.08,r'\boldmath$x \delta d_{p/t} = 2 x \delta d_{p/D} = x \delta q_0 + x \delta d_1$',transform=ax12.transAxes,size=30)
    ax13.text(0.05,0.08,r'\boldmath$x \delta u_{p/h} =   x \delta d_{p/h} = x \delta q_0 $'              ,transform=ax13.transAxes,size=30)


    py.tight_layout()
    py.subplots_adjust(hspace = 0.01, wspace = 0.01, top = 0.99, right = 0.99)

    filename = '%s/gallery/off-dqNA-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-percentiles'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)



