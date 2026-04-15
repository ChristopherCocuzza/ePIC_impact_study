import sys,os
import numpy as np
import time
import argparse
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import pylab as py

#--from scipy stack 
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid as cumtrapz

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

from analysis.obslib import ht

import kmeanconf as kc

TAR = ['p','n','d','h','t']
STF = ['F2','FL','F3','W2+','WL+','W3+','W2-','WL-','W3-']

#--unpolarized NC and CC generation
def gen_stf(wdir,Q2=None,kind=None,HT=True,TMC=True):
 
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
    _replicas=core.get_replicas(wdir)
    core.mod_conf(istep,_replicas[0]) #--set conf as specified in istep   
    print('\ngenerating STFs from %s at Q2 = %3.5f with HT = %s and TMC = %s'%(wdir,Q2,HT,TMC))

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    if 'pdf' in conf['steps'][istep]['active distributions']:
        passive = False
    else:
        passive = True

    if kind=='offshell' and 'offpdf' not in conf: return
    if kind=='offshell' and 'offpdf' in conf and conf['offpdf']==False: return

    if HT ==False: conf['ht']  = False
    if TMC==False: conf['tmc'] = False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    M2 = conf['aux'].M2
    parman=resman.parman
    resman.setup_idis()
    parman.order=_replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    idis = conf['idis']
    #--setup kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.99,100))

    Q2 = Q2*np.ones(len(X))

    #--compute X*STF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,_replicas[cnt])
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for tar in TAR:
            if tar not in XF: XF[tar] = {}
            for stf in STF:
                if stf not in XF[tar]: XF[tar][stf] = []

                xf = X*idis.get_FX(stf,X,Q2,tar,kind=kind,idx=None)
                XF[tar][stf].append(xf)

    print() 
    checkdir('%s/data'%wdir)
    if kind==None: filename ='%s/data/stf-Q2=%3.5f'%(wdir,Q2[0])
    else:          filename ='%s/data/stf-%s-Q2=%3.5f'%(wdir,kind,Q2[0])

    if HT==False:  filename+='_noHT'
    if TMC==False: filename+='_noTMC'

    filename+='.dat'

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_stf(wdir,Q2=None,mode=1):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 
  
    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
  
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
  
    #cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    #best_cluster=cluster_order[0]
  
    hand = {}
  
    filename ='%s/data/stf-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_stf(wdir,Q2)
        data=load(filename)
  
    X    = data['X']
    data = data['XF']
  
    for tar in TAR:
        if tar not in hand: hand[tar] = {}
        for stf in STF:
            mean = np.mean(data[tar][stf],axis=0)
            std  = np.std (data[tar][stf],axis=0)
  
            if tar=='p':   color='red'
            elif tar=='n': color='green'
            elif tar=='d': color='blue'
            elif tar=='h': color='magenta'
            elif tar=='t': color='orange'
            else: continue
  
            label = None
            if stf =='F2':   ax = ax11
            elif stf =='FL': ax = ax12
            elif stf =='F3': ax = ax13
            else: continue
  
            #--plot each replica
            if mode==0:
                for i in range(len(data[tar][stf])):
                    hand[tar][stf] ,= ax.plot(X,data[tar][stf][i],color=color,alpha=0.1)
      
            #--plot average and standard deviation
            if mode==1:
                ax.plot(X,mean,color=color)
                hand[tar][stf] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)
  
  
    for ax in [ax11,ax12,ax13]:
          ax.set_xlim(0,0.9)
            
          ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)
  
  
    ax11.text(0.80,0.70,r'\boldmath$xF_2$',transform=ax11.transAxes,size=40)
    ax12.text(0.80,0.70,r'\boldmath$xF_L$',transform=ax12.transAxes,size=40)
    ax13.text(0.60,0.70,r'\boldmath$xF_3$',transform=ax13.transAxes,size=40)
  
    handles,labels = [],[]
  
    if 'p' in hand: handles.append(hand['p']['F2'])
    if 'n' in hand: handles.append(hand['n']['F2'])
    if 'd' in hand: handles.append(hand['d']['F2'])
    if 'h' in hand: handles.append(hand['h']['F2'])
    if 't' in hand: handles.append(hand['t']['F2'])
  
    if 'p' in hand: labels.append(r'\boldmath$p$')
    if 'n' in hand: labels.append(r'\boldmath$n$')
    if 'd' in hand: labels.append(r'\boldmath$D$')
    if 'h' in hand: labels.append(r'\boldmath$^3{\rm He}$')
    if 't' in hand: labels.append(r'\boldmath$^3{\rm H}$')
  
    ax13.legend(handles,labels,loc='upper right',fontsize=20, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)
  
    ax11.set_ylim(0,0.1)      
    ax12.set_ylim(0,0.004)    
    ax13.set_ylim(0,0.00045)  
  
    for ax in [ax11,ax12,ax13]:
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xlabel(r'\boldmath$x$' ,size=35)   
        ax.xaxis.set_label_coords(0.97,0)
  
  
    if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
    else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)
  
    py.tight_layout()
  
    filename = '%s/gallery/stfs-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
  
    filename+='.png'
  
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_F2_rat(wdir,Q2=None,num='d',den='p',mode=1):

  #--plot F2 ratios
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,1
  fig = py.figure(figsize=(ncols*7,nrows*5))
  ax11=py.subplot(nrows,ncols,1)

  replicas=core.get_replicas(wdir)

  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  stf = 'F2'
  filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,num,stf,Q2)
  #--load data if it exists
  try:
      NUM=load(filename)
  #--generate data and then load it if it does not exist
  except:
      gen_stf(wdir,Q2,num,stf)
      NUM=load(filename)

  filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,den,stf,Q2)
  #--load data if it exists
  try:
      DEN=load(filename)
  #--generate data and then load it if it does not exist
  except:
      gen_stf(wdir,Q2,den,stf)
      DEN=load(filename)


  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  X  = NUM['X']

  F2num = np.array(NUM['XF'])
  F2den = np.array(DEN['XF'])

  rat = F2num/F2den

  mean = np.mean(rat,axis=0)
  std  = np.std (rat,axis=0)

  #--plot each replica
  if mode==0:
      for i in range(len(rat)):
          ax11.plot(X,rat[i],color='red',alpha=0.1)
  
  #--plot average and standard deviation
  if mode==1:
      ax11.fill_between(X,mean-std,mean+std,color='red',alpha=0.7)


  for ax in [ax11,ax11]:
        ax.set_xlim(0,1.0)
        ax.set_xticks([0.2,0.4,0.6,0.8])


  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)

  ax11.set_ylim(0.24,1.05)

  ax11.text(0.05, 0.10, r'\boldmath$F_2^%s/F_2^%s$'%(num,den),transform=ax11.transAxes,size=40)
  ax11.set_xlabel(r'\boldmath$x$'          ,size=30)
  ax11.xaxis.set_label_coords(0.98,0)

  if Q2 == 1.27**2: ax11.text(0.30,0.85,r'$Q^2 = m_c^2$',              transform=ax11.transAxes,size=30)
  else:             ax11.text(0.30,0.85,r'$Q^2 = %s ~ \rm{GeV^2}$'%Q2, transform=ax11.transAxes,size=30)

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.04)
      majorLocator = MultipleLocator(0.2)
      ax.xaxis.set_minor_locator(minorLocator)
      ax.xaxis.set_major_locator(majorLocator)
      ax.xaxis.set_tick_params(which='major',length=6)
      ax.xaxis.set_tick_params(which='minor',length=3)
      ax.yaxis.set_tick_params(which='major',length=6)
      ax.yaxis.set_tick_params(which='minor',length=3)

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.04)
      majorLocator = MultipleLocator(0.2)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  ax.set_xticks([0.2,0.4,0.6,0.8])

  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/stf-F2%s-F2%s-rat-Q2=%3.5f'%(wdir,num,den,Q2)
  if mode==1: filename += '-bands'
  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

def plot_EMC_rat(wdir,Q2=None,mode=0):

  #--plot EMC ratios for F2D, F2H, and F2T.  As well as super ratio for F2H/F2T
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,1
  fig = py.figure(figsize=(ncols*9,nrows*5))
  ax11=py.subplot(nrows,ncols,1)

  replicas=core.get_replicas(wdir)

  hand = {}
  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  stf = 'F2'
  TAR = ['p','n','d','h','t']
  filename ='%s/data/stf-Q2=%3.5f.dat'%(wdir,Q2)
  #--load data if it exists
  try:
      data=load(filename)
  #--generate data and then load it if it does not exist
  except:
      gen_stf(wdir,Q2)
      data=load(filename)

  X  = data['X']

  F2p = np.array(data['XF']['p']['F2'])
  F2n = np.array(data['XF']['n']['F2'])
  F2d = np.array(data['XF']['d']['F2'])
  F2h = np.array(data['XF']['h']['F2'])
  F2t = np.array(data['XF']['t']['F2'])

  ratDN = 2*F2d/(F2p+F2n)
  ratHN = 3*F2h/(2*F2p+F2n)
  ratTN = 3*F2t/(F2p+2*F2n)

  ratHT = ratHN/ratTN

  meanDN = np.mean(ratDN,axis=0)
  stdDN  = np.std (ratDN,axis=0)
  meanHN = np.mean(ratHN,axis=0)
  stdHN  = np.std (ratHN,axis=0)
  meanTN = np.mean(ratTN,axis=0)
  stdTN  = np.std (ratTN,axis=0)
  meanHT = np.mean(ratHT,axis=0)
  stdHT  = np.std (ratHT,axis=0)

  #--plot each replica
  if mode==0:
      for i in range(len(ratHN)):
          hand['DN'] ,= ax11.plot(X,ratDN[i],color='firebrick',alpha=0.1)
          hand['HN'] ,= ax11.plot(X,ratHN[i],color='darkgreen',alpha=0.1)
          hand['TN'] ,= ax11.plot(X,ratTN[i],color='blue',alpha=0.1)
          #hand['HT'] ,= ax11.plot(X,ratHT[i],color='magenta',alpha=0.1)
  
  #--plot average and standard deviation
  if mode==1:
      hand['DN'] = ax11.fill_between(X,meanDN-stdDN,meanDN+stdDN,color='firebrick',alpha=0.5,hatch='+')
      hand['HN'] = ax11.fill_between(X,meanHN-stdHN,meanHN+stdHN,color='darkgreen',alpha=0.4,hatch=None)
      hand['TN'] = ax11.fill_between(X,meanTN-stdTN,meanTN+stdTN,color='blue'     ,alpha=0.4,hatch=None)
      #hand['HT'] = ax11.fill_between(X,meanHT-stdHT,meanHT+stdHT,color='magenta'  ,alpha=0.4,hatch=None)
      #hand['DN'] ,= ax11.plot(X,meanDN+stdDN,color='firebrick',alpha=1.0,ls='-')
      #hand['HN'] ,= ax11.plot(X,meanHN+stdHN,color='darkgreen',alpha=1.0,ls='--')
      #hand['TN'] ,= ax11.plot(X,meanTN+stdTN,color='blue'     ,alpha=1.0,ls=':')
      #hand['HT'] ,= ax11.plot(X,meanHT+stdHT,color='magenta'  ,alpha=1.0,ls='-.')
      #ax11              .plot(X,meanDN-stdDN,color='firebrick',alpha=1.0,ls='-')
      #ax11              .plot(X,meanHN-stdHN,color='darkgreen',alpha=1.0,ls='--')
      #ax11              .plot(X,meanTN-stdTN,color='blue'     ,alpha=1.0,ls=':')
      #ax11              .plot(X,meanHT-stdHT,color='magenta'  ,alpha=1.0,ls='-.')


  for ax in [ax11]:
        ax.set_xlim(0,0.9)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        #ax.set_xlim(1e-4,0.9)
        #ax.semilogx()


  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)

  ax11.axhline(1,0,1,ls='--',color='black',alpha=0.5)

  ax11.set_ylim(0.93,1.09)

  ax11.text(0.05, 0.15, r'\textrm{\textbf{EMC Ratios}}',size=30, transform=ax11.transAxes)
  ax11.set_xlabel(r'\boldmath$x$'          ,size=30)
  ax11.xaxis.set_label_coords(0.97,0)

  if Q2 == 1.27**2: ax11.text(0.05,0.05,r'$Q^2 = m_c^2$',              transform=ax11.transAxes,size=30)
  else:             ax11.text(0.05,0.05,r'$Q^2 = %s ~ \rm{GeV^2}$'%Q2, transform=ax11.transAxes,size=30)

  #for ax in [ax11]:

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.01)
      majorLocator = MultipleLocator(0.05)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  handles,labels = [],[]
  handles.append(hand['DN'])
  handles.append(hand['HN'])
  handles.append(hand['TN'])
  #handles.append(hand['HT'])
  labels.append(r'\boldmath$R(D)$')
  labels.append(r'\boldmath$R(^3{\rm He})$')
  labels.append(r'\boldmath$R(^3{\rm H})$')
  #labels.append(r'\boldmath$R(^3{\rm He})/R(^3{\rm H})$')
  #labels.append(r'\boldmath$\mathcal{R}$')
  ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=25, handletextpad = 0.5, handlelength = 1.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/stf-EMC-rat-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'
  filename+='.png'

  ax11.set_rasterized(True)

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

#--charged current
def plot_CCstf(wdir,Q2=None,mode=0):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,3
  fig = py.figure(figsize=(ncols*7,nrows*4))
  ax11=py.subplot(nrows,ncols,1)
  ax12=py.subplot(nrows,ncols,2)
  ax13=py.subplot(nrows,ncols,3)

  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  hand = {}

  tar = 'p'
  STF = ['W2+','WL+','W3+','W2-','WL-','W3-']  

  for stf in STF:
      filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)
      #--load data if it exists
      try:
          data=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_stf(wdir,Q2,tar,stf)
          data=load(filename)
      X    = data['X']
      data = data['XF']
      mean = np.mean(data,axis=0)
      std  = np.std (data,axis=0)

      if stf[-1]=='+': color='red'
      if stf[-1]=='-': color='blue'

      label = None
      if stf =='W2+':   ax = ax11
      elif stf =='WL+': ax = ax12
      elif stf =='W3+': ax = ax13
      elif stf =='W2-': ax = ax11
      elif stf =='WL-': ax = ax12
      elif stf =='W3-': ax = ax13
      else: continue

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              hand[stf] ,= ax.plot(X,data[i],color=color,alpha=0.1)
  
      #--plot average and standard deviation
      if mode==1:
          ax.plot(X,mean,color=color)
          hand[stf] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)


  for ax in [ax11,ax12,ax13]:
        ax.set_xlim(2e-4,1)
        ax.semilogx()
        ax.set_xlabel(r'\boldmath$x$'          ,size=30)
        ax.xaxis.set_label_coords(0.90,0)
          
        ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20,pad=7.0)
        ax.set_xticks([0.001,0.01,0.1,1])
        ax.set_xticklabels([r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$1$'])


  ax11.text(0.40,0.70,r'\boldmath$xF_2^p$',transform=ax11.transAxes,size=40)
  ax12.text(0.40,0.70,r'\boldmath$xF_L^p$',transform=ax12.transAxes,size=40)
  ax13.text(0.40,0.70,r'\boldmath$xF_3^p$',transform=ax13.transAxes,size=40)
  
  ax13.axhline(0,0,1,ls='--',color='black',alpha=0.5)

  handles,labels = [],[]

  handles.append(hand['W2+'])
  handles.append(hand['W2-'])

  labels.append(r'\boldmath$W^+$')
  labels.append(r'\boldmath$W^-$')
  ax11.legend(handles,labels,loc='upper left',fontsize=30, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)

  ax11.set_ylim(0,0.4)      #,ax11.set_yticks([0,0.2,0.4,0.6,0.8])
  ax12.set_ylim(0,0.015)    #,ax12.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
  ax13.set_ylim(-1.0,2.0)   #,ax13.set_yticks([0,0.2,0.4,0.6])

  if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
  else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)

  py.tight_layout()

  filename = '%s/gallery/CCstfs-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()


#--as function of Q2 with fixed X
def gen_stf_func_Q2(wdir,X=0.3,kind=None,HT=True,TMC=True):
 
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    _replicas=core.get_replicas(wdir)
    core.mod_conf(istep,_replicas[0]) #--set conf as specified in istep   
    print('\ngenerating STFs from %s at X = %3.5f with HT = %s and TMC = %s'%(wdir,X,HT,TMC))

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    if 'pdf' in conf['steps'][istep]['active distributions']:
        passive = False
    else:
        passive = True

    if kind=='offshell' and 'offpdf' not in conf: return
    if kind=='offshell' and 'offpdf' in conf and conf['offpdf']==False: return

    if HT ==False: conf['ht']  = False
    if TMC==False: conf['tmc'] = False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    M2 = conf['aux'].M2
    parman=resman.parman
    resman.setup_idis()
    parman.order=_replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    idis = conf['idis']
    #--setup kinematics
    Q2 = np.geomspace(conf['Q20'],1000,100)

    X = X*np.ones(len(Q2))

    #--compute X*STF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,_replicas[cnt])
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for tar in TAR:
            if tar not in XF: XF[tar] = {}
            for stf in STF:
                if stf not in XF[tar]: XF[tar][stf] = []

                xf = X*idis.get_FX(stf,X,Q2,tar,kind=kind,idx=None)
                XF[tar][stf].append(xf)

    print() 
    checkdir('%s/data'%wdir)
    if kind==None: filename ='%s/data/stf-funcQ2-X=%3.5f'%(wdir,X[0])
    else:          filename ='%s/data/stf-funcQ2-%s-X=%3.5f'%(wdir,kind,X[0])

    if HT==False:  filename+='_noHT'
    if TMC==False: filename+='_noTMC'

    filename+='.dat'

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

#--as function of X with fixed W2
def gen_stf_fixed_W2(wdir,W2=3.5,kind=None,HT=True,TMC=True):
 
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    _replicas=core.get_replicas(wdir)
    core.mod_conf(istep,_replicas[0]) #--set conf as specified in istep   
    print('\ngenerating STFs from %s at W2 = %3.5f with HT = %s and TMC = %s'%(wdir,W2,HT,TMC))

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    if 'pdf' in conf['steps'][istep]['active distributions']:
        passive = False
    else:
        passive = True

    if kind=='offshell' and 'offpdf' not in conf: return
    if kind=='offshell' and 'offpdf' in conf and conf['offpdf']==False: return

    if HT ==False: conf['ht']  = False
    if TMC==False: conf['tmc'] = False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    M2 = conf['aux'].M2
    parman=resman.parman
    resman.setup_idis()
    parman.order=_replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    idis = conf['idis']
    #--setup kinematics
    #--setup kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.99,100))

    M2 = 0.938**2
    Q2 = (W2 - M2) * X / (1-X) 

    #--compute X*STF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,_replicas[cnt])
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for tar in TAR:
            if tar not in XF: XF[tar] = {}
            for stf in STF:
                if stf not in XF[tar]: XF[tar][stf] = []

                xf = X*idis.get_FX(stf,X,Q2,tar,kind=kind,idx=None)
                XF[tar][stf].append(xf)

    print() 
    checkdir('%s/data'%wdir)
    if kind==None: filename ='%s/data/stf-fixedW2-W2=%3.5f'%(wdir,W2)
    else:          filename ='%s/data/stf-fixedW2-%s-W2=%3.5f'%(wdir,kind,W2)

    if HT==False:  filename+='_noHT'
    if TMC==False: filename+='_noTMC'

    filename+='.dat'

    save({'X':X,'Q2':Q2,'XF':XF,'W2':W2},filename)
    print ('Saving data to %s'%filename)


#--comparisons with Marathon (F2 only)
def gen_marathon_stf(wdir):
 
    tars = ['p','n','d','h','t']
     
    print('\ngenerating STF at MARATHON kinematics from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
   
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    idis  = conf['idis']

    #--marathon kinematics
    X   = np.array([0.195,0.225,0.255,0.285,0.315,0.345,0.375,0.405,0.435,0.465,0.495,0.525,0.555,0.585,0.615,0.645,0.675,0.705,0.735,0.765,0.795,0.825])
    Q2  = 14*X

    #--compute X*STF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
       
        for tar in tars:
            if tar not in XF: XF[tar] = []
            xf = X*idis.get_FX('F2',X,Q2,tar,idx=None)
            XF[tar].append(xf)

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/stf-marathon.dat'%(wdir)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)


#--off-shell structure functions (F2 only)
def gen_off_stf(wdir,Q2=None,nucleus='d'):
   
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating offshell STF from %s for %s at Q2=%3.5f'%(wdir,nucleus,Q2))

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    #--setup kinematics for structure functions to be calculated at
    Xgrid = np.geomspace(1e-5,1e-1,20)
    Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    Q2grid = [Q2*0.99,Q2*1.01]
    conf['idis grid'] = {}
    conf['idis grid']['X']  = Xgrid 
    conf['idis grid']['Q2'] = Q2grid 
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
   
    if 'off pdf' in conf:
        idis.data['p']['F2off d'] = np.zeros(idis.X.size)
        idis.data['n']['F2off d'] = np.zeros(idis.X.size)
        idis.data['p']['F2off h'] = np.zeros(idis.X.size)
        idis.data['n']['F2off h'] = np.zeros(idis.X.size)
        idis.data['p']['F2off t'] = np.zeros(idis.X.size)
        idis.data['n']['F2off t'] = np.zeros(idis.X.size)
    else: return

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics for structure functions to be interpolated to
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))
    XM , gXM = np.meshgrid(X , idis.gX)
    Q2M, gWM = np.meshgrid(Q2, idis.gW)
    a = XM

    #--compute X*STF for all replicas        
    XF=[]
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        pdf.evolve(Q2)
        idis._update()

        if   nucleus=='d': p, n = 1,1
        elif nucleus=='h': p, n = 2,1
        elif nucleus=='t': p, n = 1,2


        #--retrieve smearing class and upper limit of integration 
        if   nucleus=='d': smf, b = idis.dsmf,idis.ymaxD #--deuterium
        elif nucleus=='h': smf, b = idis.hsmf,idis.ymaxH #--helium
        elif nucleus=='t': smf, b = idis.hsmf,idis.ymaxT #--tritium

        switch = False
        #--tritium takes helium and switches p <--> n
        if nucleus=='t': switch = True

        YM   = 0.5*(b-a)*gXM+0.5*(a+b)
        JM   = 0.5*(b-a) 
        XM_YM = XM/YM
       
        if p==n:
            fof22p = smf.get_fXX2('f22','offshell',XM,Q2M,YM)
            fof22n = fof22p
        else:
            fof22p = smf.get_fXX2('f22p','offshell',XM,Q2M,YM)
            fof22n = smf.get_fXX2('f22n','offshell',XM,Q2M,YM)
        if switch: 
            fof22p, fof22n = fof22n[:], fof22p[:]
        F2poff = idis.get_stf(XM_YM,Q2M,stf='F2',tar='p',off=True,nucleus=nucleus)
        F2noff = idis.get_stf(XM_YM,Q2M,stf='F2',tar='n',off=True,nucleus=nucleus)
        integ  = p*fof22p*F2poff + n*fof22n*F2noff

        result = X*np.einsum('ij,ij,ij->j',gWM,JM,integ)/(p+n)
        XF.append(result)

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/off-stf-%s-Q2=%3.5f.dat'%(wdir,nucleus,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_off_stf(wdir,Q2=None,mode=0):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=2,3
  fig = py.figure(figsize=(ncols*7,nrows*4))
  ax11=py.subplot(nrows,ncols,1)
  ax12=py.subplot(nrows,ncols,2)
  ax13=py.subplot(nrows,ncols,3)
  ax21=py.subplot(nrows,ncols,4)
  ax22=py.subplot(nrows,ncols,5)
  ax23=py.subplot(nrows,ncols,6)


  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  TAR = ['d','h','t']
  stf = 'F2'

  for tar in TAR:
      filename ='%s/data/off-stf-%s-Q2=%3.5f.dat'%(wdir,tar,Q2)
      #--load data if it exists
      try:
          data=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_off_stf(wdir,Q2,tar)
          data=load(filename)
      filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)
      #--load data if it exists
      try:
          STF=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_stf(wdir,Q2,tar,stf)
          STF=load(filename)

      X    = data['X']
      data = data['XF']
      STF  = STF['XF']
      mean = np.mean(data,axis=0)
      std  = np.std (data,axis=0)

      if tar == 'd': ax,color=ax11,'firebrick'
      if tar == 'h': ax,color=ax12,'darkgreen'
      if tar == 't': ax,color=ax13,'darkblue'
      

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              ax.plot(X,data[i],color=color,alpha=0.3)
    
      #--plot average and standard deviation
      if mode==1:
          ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)

      #--plot ratios to onshell structure functions 
      if tar == 'd': ax,color=ax21,'firebrick'
      if tar == 'h': ax,color=ax22,'darkgreen'
      if tar == 't': ax,color=ax23,'darkblue'

      mean = mean/np.mean(STF,axis=0)
      std  = std /np.mean(STF,axis=0)

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              ax.plot(X,data[i]/STF[i],color=color,alpha=0.3)
    
      #--plot average and standard deviation
      if mode==1:
          ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)


  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelbottom=False)
  ax12.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelbottom=False,labelleft=False)
  ax13.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelbottom=False,labelleft=False)
  ax21.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)
  ax22.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelleft=False)
  ax23.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelleft=False)

  ax11.text(0.05, 0.08, r'\boldmath$x F_2^{d({\rm off})}$'                        ,size=30, transform=ax11.transAxes)
  ax12.text(0.05, 0.08, r'\boldmath$x F_2^{^3{\rm He}({\rm off})}$'               ,size=30, transform=ax12.transAxes)
  ax13.text(0.05, 0.08, r'\boldmath$x F_2^{^3{\rm H} ({\rm off})}$'               ,size=30, transform=ax13.transAxes)
  ax21.text(0.05, 0.08, r'\boldmath$F_2^{d({\rm off})}/F_2^d$'                    ,size=30, transform=ax21.transAxes)
  ax22.text(0.05, 0.08, r'\boldmath$F_2^{^3{\rm He}({\rm off})}/F_2^{^3{\rm He}}$',size=30, transform=ax22.transAxes)
  ax23.text(0.05, 0.08, r'\boldmath$F_2^{^3{\rm H} ({\rm off})}/F_2^{^3{\rm H}}$' ,size=30, transform=ax23.transAxes)

  #if Q2 == 1.27**2: ax11.text(0.05,0.05,r'$Q^2 = m_c^2$',              transform=ax11.transAxes,size=30)
  #else:             ax11.text(0.05,0.05,r'$Q^2 = %s ~ \rm{GeV^2}$'%Q2, transform=ax11.transAxes,size=30)

  for ax in [ax11,ax12,ax13,ax21,ax22,ax23]:
      ax.set_xlim(0,0.9)
      ax.axhline(0,0,1,ls=':',color='black',alpha=0.5)
      minorLocator = MultipleLocator(0.02)
      majorLocator = MultipleLocator(0.2)
      ax.xaxis.set_minor_locator(minorLocator)
      ax.xaxis.set_major_locator(majorLocator)
      ax.xaxis.set_tick_params(which='major',length=6)
      ax.xaxis.set_tick_params(which='minor',length=3)
      ax.yaxis.set_tick_params(which='major',length=6)
      ax.yaxis.set_tick_params(which='minor',length=3)
      ax.set_xticks([0.2,0.4,0.6,0.8])

  for ax in [ax11,ax12,ax13]:
      ax.set_ylim(-0.005,0.005)
      minorLocator = MultipleLocator(0.0004)
      majorLocator = MultipleLocator(0.002)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  for ax in [ax21,ax22,ax23]:
      ax.set_ylim(-0.19,0.19)
      minorLocator = MultipleLocator(0.02)
      majorLocator = MultipleLocator(0.10)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  for ax in [ax21,ax22,ax23]:
      ax.set_xlabel(r'\boldmath$x$'          ,size=30)
      ax.xaxis.set_label_coords(0.97,0)

  handles,labels = [],[]
  #handles.append(hand['HT'])
  #labels.append(r'\boldmath$R(^3{\rm He})/R(^3{\rm H})$')
  #labels.append(r'\boldmath$\mathcal{R}$')
  #ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=25, handletextpad = 0.5, handlelength = 1.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0,wspace=0)

  filename = '%s/gallery/off-stfs-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'
  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()




#--F2 moments as function of Q2
def gen_F2_mom(wdir,tar='p',W2cut=3.5,xmin=10e-9,xmax=0.99):
 
    stf = 'F2' 
    load_config('%s/input.py'%wdir)
    Q20 = conf['Q20']
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    M2 = 0.9389**2

    #--setup kinematics for structure functions to be calculated at
    Xgrid = np.geomspace(1e-5,1e-1,20)
    Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    Q2grid = np.linspace(Q20*0.99,8*1.01,10)
    conf['idis grid'] = {}
    conf['idis grid']['X']  = Xgrid 
    conf['idis grid']['Q2'] = Q2grid 
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
    
    if tar in ['p','n']:
        idis.data[tar] = {}
        idis.data[tar][stf] = np.zeros(idis.X.size)
    else:
        idis.data[tar] = {}
        idis.data['p'][stf] = np.zeros(idis.X.size)
        idis.data['n'][stf] = np.zeros(idis.X.size)
        idis.data[tar][stf] = np.zeros(idis.X.size)
        if stf=='FL':
            idis.data['p']['F2'] = np.zeros(idis.X.size)
            idis.data['n']['F2'] = np.zeros(idis.X.size)
            idis.data[tar]['F2'] = np.zeros(idis.X.size)

    if 'off pdf' in conf:
        idis.data['p']['F2off d'] = np.zeros(idis.X.size)
        idis.data['n']['F2off d'] = np.zeros(idis.X.size)
        idis.data['p']['F2off h'] = np.zeros(idis.X.size)
        idis.data['n']['F2off h'] = np.zeros(idis.X.size)
        idis.data['p']['F2off t'] = np.zeros(idis.X.size)
        idis.data['n']['F2off t'] = np.zeros(idis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    ## setup kinematics
    Q2 = np.linspace(Q20,8,5) 
    if xmax==None: print('\ngenerating first moment for F2 %s from %s from %3.9f' % (tar, wdir, xmin))
    else:          print('\ngenerating first moment for F2 %s from %s from %3.9f to %3.2f' % (tar, wdir, xmin, xmax))


    ## compute moments for all replicas
    MOM = []
    cnt = 0

    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        idis._update()

        mom = []
        for q2 in Q2:
            if xmax==None: _xmax = q2/(W2cut - M2 + q2)
            else:          _xmax = xmax
            xs = np.geomspace(xmin,0.1,100)
            xs = np.append(xs, np.linspace(0.1, _xmax, 100))
            func = lambda x: idis.get_stf(x,q2,stf='F2',tar=tar)

            function_values = [func(_) for _ in xs]
            moment_temp = cumtrapz(function_values, xs, initial = 0.0)
            moment_temp = np.array(moment_temp)
            moment_max = moment_temp[-1]
            moment = moment_max - moment_temp
            mom.append(moment[0])

        MOM.append(mom)

    print()

    MOM = np.array(MOM)

    checkdir('%s/data'%wdir)
    if xmax==None: filename ='%s/data/stf-F2-%s-mom-xmin-%3.9f.dat'%(wdir,tar,xmin)
    else:          filename ='%s/data/stf-F2-%s-mom-xmin-%3.9f-xmax-%3.5f.dat'%(wdir,tar,xmin,xmax)

    save({'MOM':MOM,'xmin':xmin,'xmax':xmax,'Q2':Q2},filename)
    print ('Saving data to %s'%filename)

def plot_F2_mom(wdir,Q2=None,mode=1,W2cut=3.5,xmin=10e-9,xmax=0.99):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  TAR = ['p','n']
  nrows,ncols=1,1
  fig = py.figure(figsize=(ncols*7,nrows*5))
  ax11=py.subplot(nrows,ncols,1)

  load_config('%s/input.py'%wdir)
  istep=core.get_istep()

  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  hand = {}

  data = {}

  for tar in TAR:
      if xmax==None: filename ='%s/data/stf-F2-%s-mom-xmin-%3.9f.dat'%(wdir,tar,xmin)
      else:          filename ='%s/data/stf-F2-%s-mom-xmin-%3.9f-xmax-%3.5f.dat'%(wdir,tar,xmin,xmax)
      #--load data if it exists
      try:
          data[tar]=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_F2_mom(wdir,tar,)
          data[tar]=load(filename)
      Q2    = data[tar]['Q2']
      value = data[tar]['MOM']
      mean  = np.mean(value,axis=0)
      std   = np.std (value,axis=0)

      if tar=='p':     color='red'
      elif tar=='n':   color='purple'
      else: continue

      ax = ax11

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              hand[tar] ,= ax.plot(Q2,value[i],color=color,alpha=0.1)
    
      #--plot average and standard deviation
      if mode==1:
          #ax.plot(Q2,mean,color=color)
          hand[tar] = ax.fill_between(Q2,mean-std,mean+std,color=color,alpha=0.9)

  #--plot p - n
  value = data['p']['MOM'] - data['n']['MOM']
  mean  = np.mean(value,axis=0)
  std   = np.std (value,axis=0)
  #--plot each replica
  if mode==0:
      for i in range(len(data)):
          hand['p-n'] ,= ax.plot(Q2,data[i],color='blue',alpha=0.1)
  
  #--plot average and standard deviation
  if mode==1:
      #ax.plot(Q2,mean,color=color)
      hand['p-n'] = ax.fill_between(Q2,mean-std,mean+std,color='blue',alpha=0.9)

  for ax in [ax11]:
        ax.set_xlim(0.5,7.8)
          
        ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)


  #ax11.text(0.50,0.70,r'\boldmath$\int_{x_{\rm min}}^{x_{\rm max}}F_2 dx$',transform=ax11.transAxes,size=30)
  if xmax==None: ax11.text(0.20,0.80,r'\boldmath$\int_{%3.9f}^{x_{\rm max}} F_2^N~dx$'%(xmin),transform=ax11.transAxes,size=30)
  else:          ax11.text(0.20,0.80,r'\boldmath$\int_{%3.9f}^{%3.2f} F_2^N~dx$'%(xmin,xmax) ,transform=ax11.transAxes,size=30)

  handles,labels = [],[]

  handles.append(hand['p'])
  handles.append(hand['n'])
  handles.append(hand['p-n'])

  labels.append(r'\boldmath$p$')
  labels.append(r'\boldmath$n$')
  labels.append(r'\boldmath$p-n$')

  ax11.legend(handles,labels,loc='upper right',fontsize=20, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)

  ax11.set_ylim(0,0.3)      
  minorLocator = MultipleLocator(0.01)
  majorLocator = MultipleLocator(0.05)
  ax11.yaxis.set_minor_locator(minorLocator)
  ax11.yaxis.set_major_locator(majorLocator)

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.2)
      majorLocator = MultipleLocator(1.0)
      ax.xaxis.set_minor_locator(minorLocator)
      ax.xaxis.set_major_locator(majorLocator)
      ax.xaxis.set_tick_params(which='major',length=6)
      ax.xaxis.set_tick_params(which='minor',length=3)
      ax.yaxis.set_tick_params(which='major',length=6)
      ax.yaxis.set_tick_params(which='minor',length=3)
      #ax.set_xticks([0.2,0.4,0.6,0.8])
      ax.set_xlabel(r'\boldmath$Q^2~[{\rm GeV}^2]$' ,size=25)   
      #ax.xaxis.set_label_coords(0.97,0)


  #if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
  #else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)

  py.tight_layout()

  if xmax==None: filename ='%s/gallery/stf-F2-mom-xmin-%3.9f.dat'%(wdir,xmin)
  else:          filename ='%s/gallery/stf-F2-mom-xmin-%3.9f-xmax-%3.5f.dat'%(wdir,xmin,xmax)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()



#--plot F2A components (full, onshell, offshell, HT)
def plot_F2A_components(wdir,Q2=None,mode=1,regen=False):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 
  
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax14=py.subplot(nrows,ncols,4)
    ax21=py.subplot(nrows,ncols,5)
    ax22=py.subplot(nrows,ncols,6)
    ax23=py.subplot(nrows,ncols,7)
    ax24=py.subplot(nrows,ncols,8)
 
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
 
    if 'offpdf' not in conf: offpdf = False
    else: offpdf = conf['offpdf']
 
    hand = {}
 
    if regen:
        gen_stf(wdir,Q2)
        gen_stf(wdir,Q2,kind='onshell')
        gen_stf(wdir,Q2,kind='offshell')
        ht.gen_htA_Q2fixed(wdir,Q2)
   
    filename ='%s/data/stf-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        full=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_stf(wdir,Q2)
        full=load(filename)

    filename ='%s/data/stf-onshell-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        onshell=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_stf(wdir,Q2,kind='onshell')
        onshell=load(filename)

    if offpdf:
        filename ='%s/data/stf-offshell-Q2=%3.5f.dat'%(wdir,Q2)
        #--load data if it exists
        try:
            offshell=load(filename)
        #--generate data and then load it if it does not exist
        except:
            try:
                gen_stf(wdir,Q2,kind='offshell')
                offshell=load(filename)
            except:
                return
    else:
        pass
    
    filename ='%s/data/htA-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        HTA=load(filename)
    #--generate data and then load it if it does not exist
    except:
        ht.gen_htA_Q2fixed(wdir,Q2)
        HTA=load(filename)


    stf = 'F2'

    X_full = full['X']
    X_onshell = onshell['X']
    if offpdf: X_offshell = offshell['X']
    else:      X_offshell = onshell['X']
    X_HTA = HTA['X']
    full = full['XF']
    onshell  = onshell ['XF']
    if offpdf: offshell = offshell['XF']
    HTA      = HTA['HT'] 
 
    for tar in ['d','h','t']:
        if tar not in hand: hand[tar] = {}
        full_mean = np.mean(full[tar][stf],axis=0)
        full_std  = np.std (full[tar][stf],axis=0)

        onshell_mean = np.mean(onshell[tar][stf],axis=0)
        onshell_std  = np.std (onshell[tar][stf],axis=0)

        if offpdf:
            offshell_mean = np.mean(offshell[tar][stf],axis=0)
            offshell_std  = np.std (offshell[tar][stf],axis=0)
        else:
            offshell_mean = np.zeros(len(X_offshell))
            offshell_std  = np.zeros(len(X_offshell))

        #--need factor of x here
        HTA_mean = X_HTA*np.mean(HTA[tar][stf],axis=0)
        HTA_std  = X_HTA*np.std (HTA[tar][stf],axis=0)


        if offpdf: 
            offonshell_rat  = np.array(offshell[tar][stf] )/np.array(onshell[tar][stf])
        else: 
            offonshell_rat  = np.zeros_like(onshell[tar][stf])

        onshell_rat  = np.array(onshell[tar][stf] )/np.array(full[tar][stf])

        if offpdf:
            offshell_rat = np.array(offshell[tar][stf])/np.array(full[tar][stf])
        else:
            offshell_rat = np.zeros(len(X_offshell))


        HTA_rat      = np.array(HTA[tar][stf]     )/np.array(full[tar][stf])

        if offpdf:
            offonshell_rat_mean = np.mean(offonshell_rat,axis=0)
            offonshell_rat_std  = np.std (offonshell_rat,axis=0)
        else:
            offonshell_rat_mean = np.zeros(len(X_offshell))
            offonshell_rat_std  = np.zeros(len(X_offshell))

        onshell_rat_mean = np.mean(onshell_rat,axis=0)
        onshell_rat_std  = np.std (onshell_rat,axis=0)

        if offpdf:
            offshell_rat_mean = np.mean(offshell_rat,axis=0)
            offshell_rat_std  = np.std (offshell_rat,axis=0)
        else:
            offshell_rat_mean = np.zeros(len(X_offshell))
            offshell_rat_std  = np.zeros(len(X_offshell))

        HTA_rat_mean = X_HTA*np.mean(HTA_rat,axis=0)
        HTA_rat_std  = X_HTA*np.std (HTA_rat,axis=0)
    
        y0, y1 = 16, 2.5
        up_full0 = np.percentile(full[tar][stf],    y0,axis=0) 
        do_full0 = np.percentile(full[tar][stf],100-y0,axis=0) 
        up_full1 = np.percentile(full[tar][stf],    y1,axis=0) 
        do_full1 = np.percentile(full[tar][stf],100-y1,axis=0) 

        up_onshell0 = np.percentile(onshell[tar][stf],    y0,axis=0) 
        do_onshell0 = np.percentile(onshell[tar][stf],100-y0,axis=0) 
        up_onshell1 = np.percentile(onshell[tar][stf],    y1,axis=0) 
        do_onshell1 = np.percentile(onshell[tar][stf],100-y1,axis=0) 

        if offpdf:
            up_offshell0 = np.percentile(offshell[tar][stf],    y0,axis=0) 
            do_offshell0 = np.percentile(offshell[tar][stf],100-y0,axis=0) 
            up_offshell1 = np.percentile(offshell[tar][stf],    y1,axis=0) 
            do_offshell1 = np.percentile(offshell[tar][stf],100-y1,axis=0)
        else: 
            up_offshell0 = np.zeros(len(X_offshell))
            do_offshell0 = np.zeros(len(X_offshell))
            up_offshell1 = np.zeros(len(X_offshell))
            do_offshell1 = np.zeros(len(X_offshell))

        #--need factor of x here
        up_HTA0 = X_HTA*np.percentile(HTA[tar][stf],    y0,axis=0) 
        do_HTA0 = X_HTA*np.percentile(HTA[tar][stf],100-y0,axis=0) 
        up_HTA1 = X_HTA*np.percentile(HTA[tar][stf],    y1,axis=0) 
        do_HTA1 = X_HTA*np.percentile(HTA[tar][stf],100-y1,axis=0) 

        up_offonshell_rat0 = np.percentile(offonshell_rat,    y0,axis=0) 
        do_offonshell_rat0 = np.percentile(offonshell_rat,100-y0,axis=0) 
        up_offonshell_rat1 = np.percentile(offonshell_rat,    y1,axis=0) 
        do_offonshell_rat1 = np.percentile(offonshell_rat,100-y1,axis=0) 

        up_onshell_rat0 = np.percentile(onshell_rat,    y0,axis=0) 
        do_onshell_rat0 = np.percentile(onshell_rat,100-y0,axis=0) 
        up_onshell_rat1 = np.percentile(onshell_rat,    y1,axis=0) 
        do_onshell_rat1 = np.percentile(onshell_rat,100-y1,axis=0) 

        up_offshell_rat0 = np.percentile(offshell_rat,    y0,axis=0) 
        do_offshell_rat0 = np.percentile(offshell_rat,100-y0,axis=0) 
        up_offshell_rat1 = np.percentile(offshell_rat,    y1,axis=0) 
        do_offshell_rat1 = np.percentile(offshell_rat,100-y1,axis=0) 

        up_HTA_rat0 = X_HTA*np.percentile(HTA_rat,    y0,axis=0) 
        do_HTA_rat0 = X_HTA*np.percentile(HTA_rat,100-y0,axis=0) 
        up_HTA_rat1 = X_HTA*np.percentile(HTA_rat,    y1,axis=0) 
        do_HTA_rat1 = X_HTA*np.percentile(HTA_rat,100-y1,axis=0) 


        if tar=='d':   color='red'
        elif tar=='h': color='green'
        elif tar=='t': color='blue'
        else: continue
 
        #--plot each replica
        if mode==0:
            for i in range(len(full[tar][stf])):
                hand[tar][stf] ,= ax11.plot(X_full,full[tar][stf][i],color=color,alpha=0.1)
                hand[tar][stf] ,= ax12.plot(X_onshell,onshell[tar][stf][i],color=color,alpha=0.1)
                if offpdf: hand[tar][stf] ,= ax13.plot(X_offshell,offshell[tar][stf][i],color=color,alpha=0.1)
                hand[tar][stf] ,= ax14.plot(X_HTA,X_HTA*HTA[tar][stf][i],color=color,alpha=0.1)

                hand[tar][stf] ,= ax21.plot(X_onshell,offonshell_rat[i],color=color,alpha=0.1)
                hand[tar][stf] ,= ax22.plot(X_onshell,onshell_rat[i],color=color,alpha=0.1)
                if offpdf: hand[tar][stf] ,= ax23.plot(X_offshell,offshell_rat[i],color=color,alpha=0.1)
                hand[tar][stf] ,= ax24.plot(X_HTA,X_HTA*HTA_rat[i],color=color,alpha=0.1)
     
        #--plot average and standard deviation
        if mode==1:
            #ax11.plot(X_full,full_mean,color=color)
            hand[tar][stf] = ax11.fill_between(X_full,full_mean-full_std,full_mean+full_std,color=color,alpha=0.5)
            hand[tar][stf] = ax12.fill_between(X_onshell,onshell_mean-onshell_std,onshell_mean+onshell_std,color=color,alpha=0.5)
            hand[tar][stf] = ax13.fill_between(X_offshell,offshell_mean-offshell_std,offshell_mean+offshell_std,color=color,alpha=0.5)
            hand[tar][stf] = ax14.fill_between(X_HTA,HTA_mean-HTA_std,HTA_mean+HTA_std,color=color,alpha=0.5)
  
            hand[tar][stf] = ax21.fill_between(X_onshell,offonshell_rat_mean-offonshell_rat_std,offonshell_rat_mean+offonshell_rat_std,color=color,alpha=0.5)
            hand[tar][stf] = ax22.fill_between(X_onshell,onshell_rat_mean-onshell_rat_std,onshell_rat_mean+onshell_rat_std,color=color,alpha=0.5)
            hand[tar][stf] = ax23.fill_between(X_offshell,offshell_rat_mean-offshell_rat_std,offshell_rat_mean+offshell_rat_std,color=color,alpha=0.5)
            hand[tar][stf] = ax24.fill_between(X_HTA,HTA_rat_mean-HTA_rat_std,HTA_rat_mean+HTA_rat_std,color=color,alpha=0.5)

        if mode==2:
            hand[tar][stf] = ax11.fill_between(X_full,do_full0,up_full0,color=color,alpha=0.5)
            hand[tar][stf] = ax12.fill_between(X_onshell,do_onshell0,up_onshell0,color=color,alpha=0.5)
            hand[tar][stf] = ax13.fill_between(X_offshell,do_offshell0,up_offshell0,color=color,alpha=0.5)
            hand[tar][stf] = ax14.fill_between(X_HTA,do_HTA0,up_HTA0,color=color,alpha=0.5)
  
            hand[tar][stf] = ax21.fill_between(X_onshell,do_offonshell_rat0,up_offonshell_rat0,color=color,alpha=0.5)
            hand[tar][stf] = ax22.fill_between(X_onshell,do_onshell_rat0,up_onshell_rat0,color=color,alpha=0.5)
            hand[tar][stf] = ax23.fill_between(X_offshell,do_offshell_rat0,up_offshell_rat0,color=color,alpha=0.5)
            hand[tar][stf] = ax24.fill_between(X_HTA,do_HTA_rat0,up_HTA_rat0,color=color,alpha=0.5)

            #hand[tar][stf] = ax11.fill_between(X_full,do_full1,up_full1,color=color,alpha=0.5)
            #hand[tar][stf] = ax12.fill_between(X_onshell,do_onshell1,up_onshell1,color=color,alpha=0.5)
            #hand[tar][stf] = ax13.fill_between(X_offshell,do_offshell1,up_offshell1,color=color,alpha=0.5)
            #hand[tar][stf] = ax14.fill_between(X_HTA,do_HTA1,up_HTA1,color=color,alpha=0.5)
  
            #hand[tar][stf] = ax21.fill_between(X_onshell,do_offonshell_rat1,up_offonshell_rat1,color=color,alpha=0.5)
            #hand[tar][stf] = ax22.fill_between(X_onshell,do_onshell_rat1,up_onshell_rat1,color=color,alpha=0.5)
            #hand[tar][stf] = ax23.fill_between(X_offshell,do_offshell_rat1,up_offshell_rat1,color=color,alpha=0.5)
            #hand[tar][stf] = ax24.fill_between(X_HTA,do_HTA_rat1,up_HTA_rat1,color=color,alpha=0.5)
  
    for ax in [ax11,ax12,ax13,ax14,ax21,ax22,ax23,ax24]:
        ax.set_xlim(0,0.9)
          
        ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])
  
  
    ax11.text(0.75,0.80,r'\boldmath$xF_2^{A}$',transform=ax11.transAxes,size=40)
    ax12.text(0.65,0.80,r'\boldmath$xF_2^{A ({\rm on})}$',transform=ax12.transAxes,size=40)
    ax13.text(0.03,0.08,r'\boldmath$xF_2^{A ({\rm off})}$',transform=ax13.transAxes,size=40)
    ax14.text(0.03,0.80,r'\boldmath$x{\rm HT}_2^{A}$',transform=ax14.transAxes,size=40)
 
    ax21.text(0.03,0.08,r'\boldmath$F_2^{A ({\rm off})}/F_2^{A ({\rm on})}$',transform=ax21.transAxes,size=40)
    ax22.text(0.03,0.08,r'\boldmath$F_2^{A ({\rm on})}/F_2^A$' ,transform=ax22.transAxes,size=40)
    ax23.text(0.03,0.08,r'\boldmath$F_2^{A ({\rm off})}/F_2^A$',transform=ax23.transAxes,size=40)
    ax24.text(0.03,0.75,r'\boldmath${\rm HT}_2^{A}/F_2^A$'     ,transform=ax24.transAxes,size=40)
 
    handles,labels = [],[]
  
    if 'd' in hand: handles.append(hand['d']['F2'])
    if 'h' in hand: handles.append(hand['h']['F2'])
    if 't' in hand: handles.append(hand['t']['F2'])
  
    if 'd' in hand: labels.append(r'\boldmath$D$')
    if 'h' in hand: labels.append(r'\boldmath$^3{\rm He}$')
    if 't' in hand: labels.append(r'\boldmath$^3{\rm H}$')
  
    ax21.legend(handles,labels,loc=(0.00,0.20),fontsize=30, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)
  
    ax11.set_ylim( 0.001,0.090)
    ax11.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax11.yaxis.set_major_locator(MultipleLocator(0.02))

    ax12.set_ylim( 0.001,0.090)    
    ax12.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax12.yaxis.set_major_locator(MultipleLocator(0.02))
    
    ax13.set_ylim(-0.0025,0.0015)
    ax13.yaxis.set_minor_locator(MultipleLocator(0.0005))
    ax13.yaxis.set_major_locator(MultipleLocator(0.001))
  
    ax14.set_ylim(-0.001,0.0049)
    ax14.yaxis.set_minor_locator(MultipleLocator(0.0005))
    ax14.yaxis.set_major_locator(MultipleLocator(0.001))
  
    for ax in [ax21,ax22,ax23,ax24]:
        ax.set_xlabel(r'\boldmath$x$' ,size=35)   
        ax.xaxis.set_label_coords(0.97,0)

    for ax in [ax13,ax14,ax21,ax23,ax24]:
        ax.axhline(0,0,1,color='black',ls=':',alpha=1.0)

    for ax in [ax22]:
        ax.axhline(1,0,1,color='black',ls=':',alpha=1.0)

    ax21.set_ylim(-0.09,0.03)
    ax21.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax21.yaxis.set_major_locator(MultipleLocator(0.02))
 
    ax22.set_ylim( 0.81,1.14)
    ax22.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax22.yaxis.set_major_locator(MultipleLocator(0.05))


    ax23.set_ylim(-0.09,0.03)
    ax23.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax23.yaxis.set_major_locator(MultipleLocator(0.02))


    ax24.set_ylim(-0.01,0.24)
    ax24.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax24.yaxis.set_major_locator(MultipleLocator(0.05))


  
    if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
    else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)
  
    py.tight_layout()
    py.subplots_adjust(top=0.99,right=0.99,hspace=0.01,wspace=0.15) 
 
    filename = '%s/gallery/F2A-components-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
  
    filename+='.png'
  
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

#--plot R = FL/(2xF1) = FL/(rho2 * F2 - FL)
def plot_R(wdir,Q2=10,mode=2):

  nrows,ncols=1,1
  fig = py.figure(figsize=(ncols*9,nrows*5))
  ax11=py.subplot(nrows,ncols,1)

  replicas=core.get_replicas(wdir)

  hand = {}
  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  filename ='%s/data/stf-Q2=%3.5f.dat'%(wdir,Q2)
  #--load data if it exists
  try:
      data=load(filename)
  #--generate data and then load it if it does not exist
  except:
      stf.gen_stf(wdir,Q2)
      data=load(filename)

  X  = data['X']

  M = 0.938
  M2 = M**2
  rho2 = 1 + 4 * X**2 * M2/Q2

  F2p = np.array(data['XF']['p']['F2'])
  F2n = np.array(data['XF']['n']['F2'])
  F2d = np.array(data['XF']['d']['F2'])
  F2h = np.array(data['XF']['h']['F2'])
  F2t = np.array(data['XF']['t']['F2'])

  FLp = np.array(data['XF']['p']['FL'])
  FLn = np.array(data['XF']['n']['FL'])
  FLd = np.array(data['XF']['d']['FL'])
  FLh = np.array(data['XF']['h']['FL'])
  FLt = np.array(data['XF']['t']['FL'])

  R = lambda F2,FL: FL/(rho2 * F2 - FL)

  Rp = R(F2p,FLp)
  Rn = R(F2n,FLn)
  Rd = R(F2d,FLd)
  Rh = R(F2h,FLh)
  Rt = R(F2t,FLt)

  meanRp = np.mean(Rp,axis=0)
  stdRp  = np.std (Rp,axis=0)
  meanRn = np.mean(Rn,axis=0)
  stdRn  = np.std (Rn,axis=0)
  meanRd = np.mean(Rd,axis=0)
  stdRd  = np.std (Rd,axis=0)
  meanRh = np.mean(Rh,axis=0)
  stdRh  = np.std (Rh,axis=0)
  meanRt = np.mean(Rt,axis=0)
  stdRt  = np.std (Rt,axis=0)


  y =5
  doRp  = np.percentile(Rp,y    ,axis=0)
  upRp  = np.percentile(Rp,100-y,axis=0)
  doRn  = np.percentile(Rn,y    ,axis=0)
  upRn  = np.percentile(Rn,100-y,axis=0)
  doRd  = np.percentile(Rd,y    ,axis=0)
  upRd  = np.percentile(Rd,100-y,axis=0)
  doRh  = np.percentile(Rh,y    ,axis=0)
  upRh  = np.percentile(Rh,100-y,axis=0)
  doRt  = np.percentile(Rt,y    ,axis=0)
  upRt  = np.percentile(Rt,100-y,axis=0)

  if mode==2:
      hand['Rp'] = ax11.fill_between(X,doRp,upRp,color='firebrick',alpha=0.9,zorder=3,hatch=None)
      hand['Rn'] = ax11.fill_between(X,doRn,upRn,color='darkgreen',alpha=0.9,zorder=2,hatch=None)
      #hand['Rd'] = ax11.fill_between(X,doRd,upRd,color='darkblue',alpha=0.9,zorder=3,hatch=None)
      #hand['Rh'] = ax11.fill_between(X,doRh,upRh,color='magenta',alpha=0.9,zorder=3,hatch=None)
      #hand['Rt'] = ax11.fill_between(X,doRt,upRt,color='orange',alpha=0.9,zorder=3,hatch=None)

  if mode==1:
      hand['Rp'] = ax11.fill_between(X,meanRp-stdRp,meanRp+stdRp,color='firebrick',alpha=0.9,zorder=3,hatch=None)
      hand['Rn'] = ax11.fill_between(X,meanRn-stdRn,meanRn+stdRn,color='darkgreen',alpha=0.9,zorder=2,hatch=None)
      #hand['Rd'] = ax11.fill_between(X,meanRd-stdRd,meanRd+stdRd,color='darkblue',alpha=0.9,zorder=3,hatch=None)
      #hand['Rh'] = ax11.fill_between(X,meanRh-stdRh,meanRh+stdRh,color='magenta',alpha=0.9,zorder=3,hatch=None)
      #hand['Rt'] = ax11.fill_between(X,meanRt-stdRt,meanRt+stdRt,color='orange',alpha=0.9,zorder=3,hatch=None)

  if mode==0:
      for i in range(len(Rp)):
          hand['Rp'] ,= ax11.plot(X,Rp[i],color='firebrick',alpha=0.2,zorder=3)
          hand['Rn'] ,= ax11.plot(X,Rn[i],color='darkgreen',alpha=0.2,zorder=2)

  for ax in [ax11]:
      ax.set_xlim(0,0.9)
      minorLocator = MultipleLocator(0.02)
      majorLocator = MultipleLocator(0.2)
      ax.xaxis.set_minor_locator(minorLocator)
      ax.xaxis.set_major_locator(majorLocator)
      ax.xaxis.set_tick_params(which='major',length=6)
      ax.xaxis.set_tick_params(which='minor',length=3)
      ax.yaxis.set_tick_params(which='major',length=6)
      ax.yaxis.set_tick_params(which='minor',length=3)
      ax.set_xticks([0.2,0.4,0.6,0.8])
      #ax.set_xlim(1e-4,0.9)
      #ax.semilogx()


      ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)

      ax.set_xlabel(r'\boldmath$x$'          ,size=40)
      ax.xaxis.set_label_coords(0.96,0)

  #ax11.axhline(0,0,1,ls='--',color='black',alpha=0.5)
  ax11.set_ylim(0,0.49)
  minorLocator = MultipleLocator(0.05)
  majorLocator = MultipleLocator(0.10)
  ax11.yaxis.set_minor_locator(minorLocator)
  ax11.yaxis.set_major_locator(majorLocator)

  #ax12.axhline(0,0,1,ls='--',color='black',alpha=0.5)
  #ax12.axhline(1,0,1,ls='--',color='black',alpha=0.5)
  #ax12.set_ylim(-0.05,1.49)
  #minorLocator = MultipleLocator(0.1)
  #majorLocator = MultipleLocator(0.2)
  #ax12.yaxis.set_minor_locator(minorLocator)
  #ax12.yaxis.set_major_locator(majorLocator)

  ax11.text(0.10, 0.87, r'\boldmath$R=\sigma_L/\sigma_T$',size=40, transform=ax11.transAxes)

  ax11.text(0.02,0.05,r'$Q^2 = %s ~ \rm{GeV^2}$'%Q2, transform=ax11.transAxes,size=20)

  handles,labels = [],[]
  handles.append(hand['Rp'])
  handles.append(hand['Rn'])
  labels.append(r'\boldmath$p$')
  labels.append(r'\boldmath$n$')
  ax11.legend(handles,labels,frameon=False,loc=(0.75,0.55),fontsize=30, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

  #handles,labels = [],[]
  #handles.append(hand['F2p'])
  #handles.append(hand['FLp'])
  #labels.append(r'\boldmath$F_2$'+r'\textrm{\textbf{ term/total}}')
  #labels.append(r'\boldmath$F_L$'+r'\textrm{\textbf{ term/total}}')
  #ax12.legend(handles,labels,frameon=False,loc=(0.50,0.10),fontsize=30, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)



  py.tight_layout()
  py.subplots_adjust(hspace=0,top=0.99,right=0.99)

  filename = '%s/gallery/stf_R'%(wdir)
  if mode==1: filename += '_bands'
  if mode==2: filename += '_ci'
  filename += '.png'

  ax11.set_rasterized(True)

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()



def gen_F2_off_components(wdir,Q2=None):
 
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
    _replicas=core.get_replicas(wdir)
    core.mod_conf(istep,_replicas[0]) #--set conf as specified in istep   
    print('\ngenerating F2 off components from %s at Q2 = %3.5f'%(wdir,Q2))

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    if 'pdf' in conf['steps'][istep]['active distributions']:
        passive = False
    else:
        passive = True

    if 'offpdf' not in conf: return
    if 'offpdf' in conf and conf['offpdf']==False: return

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    M2 = conf['aux'].M2
    parman=resman.parman
    resman.setup_idis()
    parman.order=_replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    idis = conf['idis']
    #--setup kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.99,100))

    Q2 = Q2*np.ones(len(X))

    T = ['d','h','t']
    N = ['p','n']
    funcs = ['u0','d0','u1','d1']

    #--compute X*STF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,_replicas[cnt])
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for tar in T:
            if tar not in XF: XF[tar] = {}
            for n in N:
                if n not in XF[tar]: XF[tar][n] = {}
                for func in funcs:
                    if func not in XF[tar][n]: XF[tar][n][func] = []

                    xf = X*idis.get_FXA_dqx('F2',func,X,Q2,n,tar,idx=None)
                    XF[tar][n][func].append(xf)

    for tar in T:
        for n in N:
            for func in funcs:
                XF[tar][n][func] = np.array(XF[tar][n][func])

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/F2-off-components-Q2=%3.5f.dat'%(wdir,Q2[0])

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_F2A_off_components(wdir,Q2=None,mode=1,regen=False):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 
 
    plot_F2A_off_components_ratio(wdir,Q2,mode,regen)
 
    nrows,ncols=3,6
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax14=py.subplot(nrows,ncols,4)
    ax15=py.subplot(nrows,ncols,5)
    ax16=py.subplot(nrows,ncols,6)
    ax21=py.subplot(nrows,ncols,7)
    ax22=py.subplot(nrows,ncols,8)
    ax23=py.subplot(nrows,ncols,9)
    ax24=py.subplot(nrows,ncols,10)
    ax25=py.subplot(nrows,ncols,11)
    ax26=py.subplot(nrows,ncols,12)
    ax31=py.subplot(nrows,ncols,13)
    ax32=py.subplot(nrows,ncols,14)
    ax33=py.subplot(nrows,ncols,15)
    ax34=py.subplot(nrows,ncols,16)
    ax35=py.subplot(nrows,ncols,17)
    ax36=py.subplot(nrows,ncols,18)
 
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
 
    if 'offpdf' not in conf: offpdf = False
    else: offpdf = conf['offpdf']
 
    hand = {}
 
    if regen:
        gen_stf(wdir,Q2)
        gen_stf(wdir,Q2,kind='onshell')
        gen_stf(wdir,Q2,kind='offshell')
        ht.gen_htA_Q2fixed(wdir,Q2)
   
    filename ='%s/data/F2-off-components-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_F2_off_components(wdir,Q2)
        data=load(filename)


    stf = 'F2'

    X = data['X']
    STF = data['XF']
    D = data['XF']['d']
    H = data['XF']['h']
    T = data['XF']['t']

    eu2 = 4/9
    ed2 = 1/9

    D_u0 = 0.5 * eu2 * D['p']['u0'] + 0.5 * ed2 * D['n']['u0'] 
    D_d0 = 0.5 * ed2 * D['p']['d0'] + 0.5 * eu2 * D['n']['d0'] 
    D_u1 = 0.5 * eu2 * D['p']['u1'] + 0.5 * ed2 * D['n']['u1'] 
    D_d1 = 0.5 * ed2 * D['p']['d1'] + 0.5 * eu2 * D['n']['d1']

    H_u0 = eu2 * H['p']['u0'] + ed2 * H['n']['u0'] 
    H_d0 = ed2 * H['p']['d0'] + eu2 * H['n']['d0']
    H_u1 = eu2 * H['n']['u1']
    H_d1 = ed2 * H['n']['d1']

    T_u0 = eu2 * T['p']['u0'] + ed2 * T['n']['u0'] 
    T_d0 = eu2 * T['n']['d0'] + ed2 * T['p']['d0']
    T_u1 = eu2 * T['p']['u1']
    T_d1 = ed2 * T['p']['d1']

    D_0 = D_u0 + D_d0
    D_1 = D_u1 + D_d1

    H_0 = H_u0 + H_d0
    H_1 = H_u1 + H_d1

    T_0 = T_u0 + T_d0
    T_1 = T_u1 + T_d1

    y0, y1 = 16, 2.5

    color = 'red'
    #--plot each replica
    if mode==0:
        for i in range(len(D_u0)):
            ax11.plot(X,D_u0[i],color=color,alpha=0.1)
            ax12.plot(X,D_d0[i],color=color,alpha=0.1)
            ax13.plot(X,D_u1[i],color=color,alpha=0.1)
            ax14.plot(X,D_d1[i],color=color,alpha=0.1)
            ax15.plot(X,D_0 [i],color=color,alpha=0.1)
            ax16.plot(X,D_1 [i],color=color,alpha=0.1)

            ax21.plot(X,H_u0[i],color=color,alpha=0.1)
            ax22.plot(X,H_d0[i],color=color,alpha=0.1)
            ax23.plot(X,H_u1[i],color=color,alpha=0.1)
            ax24.plot(X,H_d1[i],color=color,alpha=0.1)
            ax25.plot(X,H_0 [i],color=color,alpha=0.1)
            ax26.plot(X,H_1 [i],color=color,alpha=0.1)

            ax31.plot(X,T_u0[i],color=color,alpha=0.1)
            ax32.plot(X,T_d0[i],color=color,alpha=0.1)
            ax33.plot(X,T_u1[i],color=color,alpha=0.1)
            ax34.plot(X,T_d1[i],color=color,alpha=0.1)
            ax35.plot(X,T_0 [i],color=color,alpha=0.1)
            ax36.plot(X,T_1 [i],color=color,alpha=0.1)
    
    #--plot average and standard deviation
    #if mode==1:
    #    #ax11.plot(X_full,full_mean,color=color)
    #    ax11.fill_between(X_full,full_mean-full_std,full_mean+full_std,color=color,alpha=0.5)

    if mode==2:
        ax11.fill_between(X,np.percentile(D_u0,y0,axis=0),np.percentile(D_u0,100-y0,axis=0),color=color,alpha=0.5)
        ax12.fill_between(X,np.percentile(D_d0,y0,axis=0),np.percentile(D_d0,100-y0,axis=0),color=color,alpha=0.5)
        ax13.fill_between(X,np.percentile(D_u1,y0,axis=0),np.percentile(D_u1,100-y0,axis=0),color=color,alpha=0.5)
        ax14.fill_between(X,np.percentile(D_d1,y0,axis=0),np.percentile(D_d1,100-y0,axis=0),color=color,alpha=0.5)
        ax15.fill_between(X,np.percentile(D_0 ,y0,axis=0),np.percentile(D_0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax16.fill_between(X,np.percentile(D_1 ,y0,axis=0),np.percentile(D_1 ,100-y0,axis=0),color=color,alpha=0.5)

        ax21.fill_between(X,np.percentile(H_u0,y0,axis=0),np.percentile(H_u0,100-y0,axis=0),color=color,alpha=0.5)
        ax22.fill_between(X,np.percentile(H_d0,y0,axis=0),np.percentile(H_d0,100-y0,axis=0),color=color,alpha=0.5)
        ax23.fill_between(X,np.percentile(H_u1,y0,axis=0),np.percentile(H_u1,100-y0,axis=0),color=color,alpha=0.5)
        ax24.fill_between(X,np.percentile(H_d1,y0,axis=0),np.percentile(H_d1,100-y0,axis=0),color=color,alpha=0.5)
        ax25.fill_between(X,np.percentile(H_0 ,y0,axis=0),np.percentile(H_0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax26.fill_between(X,np.percentile(H_1 ,y0,axis=0),np.percentile(H_1 ,100-y0,axis=0),color=color,alpha=0.5)

        ax31.fill_between(X,np.percentile(T_u0,y0,axis=0),np.percentile(T_u0,100-y0,axis=0),color=color,alpha=0.5)
        ax32.fill_between(X,np.percentile(T_d0,y0,axis=0),np.percentile(T_d0,100-y0,axis=0),color=color,alpha=0.5)
        ax33.fill_between(X,np.percentile(T_u1,y0,axis=0),np.percentile(T_u1,100-y0,axis=0),color=color,alpha=0.5)
        ax34.fill_between(X,np.percentile(T_d1,y0,axis=0),np.percentile(T_d1,100-y0,axis=0),color=color,alpha=0.5)
        ax35.fill_between(X,np.percentile(T_0 ,y0,axis=0),np.percentile(T_0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax36.fill_between(X,np.percentile(T_1 ,y0,axis=0),np.percentile(T_1 ,100-y0,axis=0),color=color,alpha=0.5)

        ax11.fill_between(X,np.percentile(D_u0,y1,axis=0),np.percentile(D_u0,100-y1,axis=0),color=color,alpha=0.5)
        ax12.fill_between(X,np.percentile(D_d0,y1,axis=0),np.percentile(D_d0,100-y1,axis=0),color=color,alpha=0.5)
        ax13.fill_between(X,np.percentile(D_u1,y1,axis=0),np.percentile(D_u1,100-y1,axis=0),color=color,alpha=0.5)
        ax14.fill_between(X,np.percentile(D_d1,y1,axis=0),np.percentile(D_d1,100-y1,axis=0),color=color,alpha=0.5)
        ax15.fill_between(X,np.percentile(D_0 ,y1,axis=0),np.percentile(D_0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax16.fill_between(X,np.percentile(D_1 ,y1,axis=0),np.percentile(D_1 ,100-y1,axis=0),color=color,alpha=0.5)

        ax21.fill_between(X,np.percentile(H_u0,y1,axis=0),np.percentile(H_u0,100-y1,axis=0),color=color,alpha=0.5)
        ax22.fill_between(X,np.percentile(H_d0,y1,axis=0),np.percentile(H_d0,100-y1,axis=0),color=color,alpha=0.5)
        ax23.fill_between(X,np.percentile(H_u1,y1,axis=0),np.percentile(H_u1,100-y1,axis=0),color=color,alpha=0.5)
        ax24.fill_between(X,np.percentile(H_d1,y1,axis=0),np.percentile(H_d1,100-y1,axis=0),color=color,alpha=0.5)
        ax25.fill_between(X,np.percentile(H_0 ,y1,axis=0),np.percentile(H_0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax26.fill_between(X,np.percentile(H_1 ,y1,axis=0),np.percentile(H_1 ,100-y1,axis=0),color=color,alpha=0.5)

        ax31.fill_between(X,np.percentile(T_u0,y1,axis=0),np.percentile(T_u0,100-y1,axis=0),color=color,alpha=0.5)
        ax32.fill_between(X,np.percentile(T_d0,y1,axis=0),np.percentile(T_d0,100-y1,axis=0),color=color,alpha=0.5)
        ax33.fill_between(X,np.percentile(T_u1,y1,axis=0),np.percentile(T_u1,100-y1,axis=0),color=color,alpha=0.5)
        ax34.fill_between(X,np.percentile(T_d1,y1,axis=0),np.percentile(T_d1,100-y1,axis=0),color=color,alpha=0.5)
        ax35.fill_between(X,np.percentile(T_0 ,y1,axis=0),np.percentile(T_0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax36.fill_between(X,np.percentile(T_1 ,y1,axis=0),np.percentile(T_1 ,100-y1,axis=0),color=color,alpha=0.5)

    for ax in [ax11,ax12,ax13,ax14,ax15,ax16,ax21,ax22,ax23,ax24,ax25,ax26,ax31,ax32,ax33,ax34,ax35,ax36]:
        ax.set_xlim(0,0.9)
          
        ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])
 
        ax.axhline(0,0,1,color='black',ls=':',alpha=1.0)

        ax.set_ylim(-0.007,0.007) 

    for ax in [ax12,ax13,ax14,ax15,ax16,ax22,ax23,ax24,ax25,ax26,ax32,ax33,ax34,ax35,ax36]:
        ax.tick_params(labelleft=False)
  
    ax11.text(0.05,0.80,r'\boldmath$xF_2^{D}(\delta u_0)$'             ,transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.80,r'\boldmath$xF_2^{D}(\delta d_0)$'             ,transform=ax12.transAxes,size=40)
    ax13.text(0.05,0.80,r'\boldmath$xF_2^{D}(\delta u_1)$'             ,transform=ax13.transAxes,size=40)
    ax14.text(0.05,0.80,r'\boldmath$xF_2^{D}(\delta d_1)$'             ,transform=ax14.transAxes,size=40)
    ax15.text(0.05,0.80,r'\boldmath$xF_2^{D}(\delta u_0 , \delta d_0)$',transform=ax15.transAxes,size=40)
    ax16.text(0.05,0.80,r'\boldmath$xF_2^{D}(\delta u_1 , \delta d_1)$',transform=ax16.transAxes,size=40)

    ax21.text(0.05,0.80,r'\boldmath$xF_2^{h}(\delta u_0)$'             ,transform=ax21.transAxes,size=40)
    ax22.text(0.05,0.80,r'\boldmath$xF_2^{h}(\delta d_0)$'             ,transform=ax22.transAxes,size=40)
    ax23.text(0.05,0.80,r'\boldmath$xF_2^{h}(\delta u_1)$'             ,transform=ax23.transAxes,size=40)
    ax24.text(0.05,0.80,r'\boldmath$xF_2^{h}(\delta d_1)$'             ,transform=ax24.transAxes,size=40)
    ax25.text(0.05,0.80,r'\boldmath$xF_2^{h}(\delta u_0 , \delta d_0)$',transform=ax25.transAxes,size=40)
    ax26.text(0.05,0.80,r'\boldmath$xF_2^{h}(\delta u_1 , \delta d_1)$',transform=ax26.transAxes,size=40)

    ax31.text(0.05,0.80,r'\boldmath$xF_2^{t}(\delta u_0)$'             ,transform=ax31.transAxes,size=40)
    ax32.text(0.05,0.80,r'\boldmath$xF_2^{t}(\delta d_0)$'             ,transform=ax32.transAxes,size=40)
    ax33.text(0.05,0.80,r'\boldmath$xF_2^{t}(\delta u_1)$'             ,transform=ax33.transAxes,size=40)
    ax34.text(0.05,0.80,r'\boldmath$xF_2^{t}(\delta d_1)$'             ,transform=ax34.transAxes,size=40)
    ax35.text(0.05,0.80,r'\boldmath$xF_2^{t}(\delta u_0 , \delta d_0)$',transform=ax35.transAxes,size=40)
    ax36.text(0.05,0.80,r'\boldmath$xF_2^{t}(\delta u_1 , \delta d_1)$',transform=ax36.transAxes,size=40)
 
    #handles,labels = [],[]
  
    #if 'd' in hand: handles.append(hand['d']['F2'])
    #if 'h' in hand: handles.append(hand['h']['F2'])
    #if 't' in hand: handles.append(hand['t']['F2'])
  
    #if 'd' in hand: labels.append(r'\boldmath$D$')
    #if 'h' in hand: labels.append(r'\boldmath$^3{\rm He}$')
    #if 't' in hand: labels.append(r'\boldmath$^3{\rm H}$')
  
    #ax21.legend(handles,labels,loc=(0.00,0.20),fontsize=30, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)
  
    #ax11.set_ylim( 0.001,0.090)
    #ax11.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax11.yaxis.set_major_locator(MultipleLocator(0.02))

    #ax12.set_ylim( 0.001,0.090)    
    #ax12.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax12.yaxis.set_major_locator(MultipleLocator(0.02))
    #
    #ax13.set_ylim(-0.0025,0.0015)
    #ax13.yaxis.set_minor_locator(MultipleLocator(0.0005))
    #ax13.yaxis.set_major_locator(MultipleLocator(0.001))
  
    #ax14.set_ylim(-0.001,0.0049)
    #ax14.yaxis.set_minor_locator(MultipleLocator(0.0005))
    #ax14.yaxis.set_major_locator(MultipleLocator(0.001))
  
    for ax in [ax31,ax32,ax33,ax34,ax35,ax36]:
        ax.set_xlabel(r'\boldmath$x$' ,size=35)   
        ax.xaxis.set_label_coords(0.97,0)

    #for ax in [ax13,ax14,ax21,ax23,ax24]:
    #    ax.axhline(0,0,1,color='black',ls=':',alpha=1.0)

    #for ax in [ax22]:
    #    ax.axhline(1,0,1,color='black',ls=':',alpha=1.0)

    #ax21.set_ylim(-0.09,0.03)
    #ax21.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax21.yaxis.set_major_locator(MultipleLocator(0.02))
 
    #ax22.set_ylim( 0.81,1.14)
    #ax22.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax22.yaxis.set_major_locator(MultipleLocator(0.05))


    #ax23.set_ylim(-0.09,0.03)
    #ax23.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax23.yaxis.set_major_locator(MultipleLocator(0.02))


    #ax24.set_ylim(-0.01,0.24)
    #ax24.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax24.yaxis.set_major_locator(MultipleLocator(0.05))


  
    if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
    else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)
  
    py.tight_layout()
    py.subplots_adjust(top=0.99,right=0.99,hspace=0.01,wspace=0.01) 
 
    filename = '%s/gallery/F2A-off-components-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
  
    filename+='.png'
  
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_F2A_off_components_ratio(wdir,Q2=None,mode=2,regen=False):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 
  
    nrows,ncols=3,7
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax14=py.subplot(nrows,ncols,4)
    ax15=py.subplot(nrows,ncols,5)
    ax16=py.subplot(nrows,ncols,6)
    ax17=py.subplot(nrows,ncols,7)
    ax21=py.subplot(nrows,ncols,8)
    ax22=py.subplot(nrows,ncols,9)
    ax23=py.subplot(nrows,ncols,10)
    ax24=py.subplot(nrows,ncols,11)
    ax25=py.subplot(nrows,ncols,12)
    ax26=py.subplot(nrows,ncols,13)
    ax27=py.subplot(nrows,ncols,14)
    ax31=py.subplot(nrows,ncols,15)
    ax32=py.subplot(nrows,ncols,16)
    ax33=py.subplot(nrows,ncols,17)
    ax34=py.subplot(nrows,ncols,18)
    ax35=py.subplot(nrows,ncols,19)
    ax36=py.subplot(nrows,ncols,20)
    ax37=py.subplot(nrows,ncols,21)
 
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
 
    if 'offpdf' not in conf: offpdf = False
    else: offpdf = conf['offpdf']
 
    hand = {}
 
    if regen:
        gen_F2_off_components(wdir,Q2)
        gen_stf(wdir,Q2)
   
    filename ='%s/data/F2-off-components-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_F2_off_components(wdir,Q2)
        data=load(filename)

    #--entire structure func
    filename ='%s/data/stf-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        full=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_stf(wdir,Q2)
        full=load(filename)

    stf = 'F2'

    F2_D = full['XF']['d']['F2']
    F2_h = full['XF']['h']['F2']
    F2_t = full['XF']['t']['F2']

    X = data['X']
    STF = data['XF']
    D = data['XF']['d']
    H = data['XF']['h']
    T = data['XF']['t']

    eu2 = 4/9
    ed2 = 1/9

    D_u0 = (0.5 * eu2 * D['p']['u0'] + 0.5 * ed2 * D['n']['u0'])/F2_D 
    D_d0 = (0.5 * ed2 * D['p']['d0'] + 0.5 * eu2 * D['n']['d0'])/F2_D 
    D_u1 = (0.5 * eu2 * D['p']['u1'] + 0.5 * ed2 * D['n']['u1'])/F2_D 
    D_d1 = (0.5 * ed2 * D['p']['d1'] + 0.5 * eu2 * D['n']['d1'])/F2_D

    H_u0 = (eu2 * H['p']['u0'] + ed2 * H['n']['u0'])/F2_h 
    H_d0 = (ed2 * H['p']['d0'] + eu2 * H['n']['d0'])/F2_h 
    H_u1 = (ed2 * H['n']['u1'])/F2_h 
    H_d1 = (eu2 * H['n']['d1'])/F2_h 

    T_u0 = (eu2 * T['p']['u0'] + ed2 * T['n']['u0'])/F2_t  
    T_d0 = (eu2 * T['n']['d0'] + ed2 * T['p']['d0'])/F2_t
    T_u1 = (eu2 * T['p']['u1'])/F2_t
    T_d1 = (ed2 * T['p']['d1'])/F2_t

    D_0 = D_u0 + D_d0
    D_1 = D_u1 + D_d1

    H_0 = H_u0 + H_d0
    H_1 = H_u1 + H_d1

    T_0 = T_u0 + T_d0
    T_1 = T_u1 + T_d1

    D_tot = D_0 + D_1
    H_tot = H_0 + H_1
    T_tot = T_0 + T_1

    y0, y1 = 16, 2.5

    color = 'red'
    #--plot each replica
    if mode==0:
        for i in range(len(D_u0)):
            ax11.plot(X,D_u0 [i],color=color,alpha=0.1)
            ax12.plot(X,D_d0 [i],color=color,alpha=0.1)
            ax13.plot(X,D_u1 [i],color=color,alpha=0.1)
            ax14.plot(X,D_d1 [i],color=color,alpha=0.1)
            ax15.plot(X,D_0  [i],color=color,alpha=0.1)
            ax16.plot(X,D_1  [i],color=color,alpha=0.1)
            ax17.plot(X,D_tot[i],color=color,alpha=0.1)

            ax21.plot(X,H_u0 [i],color=color,alpha=0.1)
            ax22.plot(X,H_d0 [i],color=color,alpha=0.1)
            ax23.plot(X,H_u1 [i],color=color,alpha=0.1)
            ax24.plot(X,H_d1 [i],color=color,alpha=0.1)
            ax25.plot(X,H_0  [i],color=color,alpha=0.1)
            ax26.plot(X,H_1  [i],color=color,alpha=0.1)
            ax27.plot(X,H_tot[i],color=color,alpha=0.1)

            ax31.plot(X,T_u0 [i],color=color,alpha=0.1)
            ax32.plot(X,T_d0 [i],color=color,alpha=0.1)
            ax33.plot(X,T_u1 [i],color=color,alpha=0.1)
            ax34.plot(X,T_d1 [i],color=color,alpha=0.1)
            ax35.plot(X,T_0  [i],color=color,alpha=0.1)
            ax36.plot(X,T_1  [i],color=color,alpha=0.1)
            ax37.plot(X,T_tot[i],color=color,alpha=0.1)
    
    #--plot average and standard deviation
    #if mode==1:
    #    #ax11.plot(X_full,full_mean,color=color)
    #    ax11.fill_between(X_full,full_mean-full_std,full_mean+full_std,color=color,alpha=0.5)

    if mode==2:
        ax11.fill_between(X,np.percentile(D_u0 ,y0,axis=0),np.percentile(D_u0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax12.fill_between(X,np.percentile(D_d0 ,y0,axis=0),np.percentile(D_d0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax13.fill_between(X,np.percentile(D_u1 ,y0,axis=0),np.percentile(D_u1 ,100-y0,axis=0),color=color,alpha=0.5)
        ax14.fill_between(X,np.percentile(D_d1 ,y0,axis=0),np.percentile(D_d1 ,100-y0,axis=0),color=color,alpha=0.5)
        ax15.fill_between(X,np.percentile(D_0  ,y0,axis=0),np.percentile(D_0  ,100-y0,axis=0),color=color,alpha=0.5)
        ax16.fill_between(X,np.percentile(D_1  ,y0,axis=0),np.percentile(D_1  ,100-y0,axis=0),color=color,alpha=0.5)
        ax17.fill_between(X,np.percentile(D_tot,y0,axis=0),np.percentile(D_tot,100-y0,axis=0),color=color,alpha=0.5)

        ax21.fill_between(X,np.percentile(H_u0 ,y0,axis=0),np.percentile(H_u0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax22.fill_between(X,np.percentile(H_d0 ,y0,axis=0),np.percentile(H_d0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax23.fill_between(X,np.percentile(H_u1 ,y0,axis=0),np.percentile(H_u1 ,100-y0,axis=0),color=color,alpha=0.5)
        ax24.fill_between(X,np.percentile(H_d1 ,y0,axis=0),np.percentile(H_d1 ,100-y0,axis=0),color=color,alpha=0.5)
        ax25.fill_between(X,np.percentile(H_0  ,y0,axis=0),np.percentile(H_0  ,100-y0,axis=0),color=color,alpha=0.5)
        ax26.fill_between(X,np.percentile(H_1  ,y0,axis=0),np.percentile(H_1  ,100-y0,axis=0),color=color,alpha=0.5)
        ax27.fill_between(X,np.percentile(H_tot,y0,axis=0),np.percentile(H_tot,100-y0,axis=0),color=color,alpha=0.5)

        ax31.fill_between(X,np.percentile(T_u0 ,y0,axis=0),np.percentile(T_u0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax32.fill_between(X,np.percentile(T_d0 ,y0,axis=0),np.percentile(T_d0 ,100-y0,axis=0),color=color,alpha=0.5)
        ax33.fill_between(X,np.percentile(T_u1 ,y0,axis=0),np.percentile(T_u1 ,100-y0,axis=0),color=color,alpha=0.5)
        ax34.fill_between(X,np.percentile(T_d1 ,y0,axis=0),np.percentile(T_d1 ,100-y0,axis=0),color=color,alpha=0.5)
        ax35.fill_between(X,np.percentile(T_0  ,y0,axis=0),np.percentile(T_0  ,100-y0,axis=0),color=color,alpha=0.5)
        ax36.fill_between(X,np.percentile(T_1  ,y0,axis=0),np.percentile(T_1  ,100-y0,axis=0),color=color,alpha=0.5)
        ax37.fill_between(X,np.percentile(T_tot,y0,axis=0),np.percentile(T_tot,100-y0,axis=0),color=color,alpha=0.5)

        ax11.fill_between(X,np.percentile(D_u0 ,y1,axis=0),np.percentile(D_u0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax12.fill_between(X,np.percentile(D_d0 ,y1,axis=0),np.percentile(D_d0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax13.fill_between(X,np.percentile(D_u1 ,y1,axis=0),np.percentile(D_u1 ,100-y1,axis=0),color=color,alpha=0.5)
        ax14.fill_between(X,np.percentile(D_d1 ,y1,axis=0),np.percentile(D_d1 ,100-y1,axis=0),color=color,alpha=0.5)
        ax15.fill_between(X,np.percentile(D_0  ,y1,axis=0),np.percentile(D_0  ,100-y1,axis=0),color=color,alpha=0.5)
        ax16.fill_between(X,np.percentile(D_1  ,y1,axis=0),np.percentile(D_1  ,100-y1,axis=0),color=color,alpha=0.5)
        ax17.fill_between(X,np.percentile(D_tot,y1,axis=0),np.percentile(D_tot,100-y1,axis=0),color=color,alpha=0.5)

        ax21.fill_between(X,np.percentile(H_u0 ,y1,axis=0),np.percentile(H_u0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax22.fill_between(X,np.percentile(H_d0 ,y1,axis=0),np.percentile(H_d0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax23.fill_between(X,np.percentile(H_u1 ,y1,axis=0),np.percentile(H_u1 ,100-y1,axis=0),color=color,alpha=0.5)
        ax24.fill_between(X,np.percentile(H_d1 ,y1,axis=0),np.percentile(H_d1 ,100-y1,axis=0),color=color,alpha=0.5)
        ax25.fill_between(X,np.percentile(H_0  ,y1,axis=0),np.percentile(H_0  ,100-y1,axis=0),color=color,alpha=0.5)
        ax26.fill_between(X,np.percentile(H_1  ,y1,axis=0),np.percentile(H_1  ,100-y1,axis=0),color=color,alpha=0.5)
        ax27.fill_between(X,np.percentile(H_tot,y1,axis=0),np.percentile(H_tot,100-y1,axis=0),color=color,alpha=0.5)

        ax31.fill_between(X,np.percentile(T_u0 ,y1,axis=0),np.percentile(T_u0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax32.fill_between(X,np.percentile(T_d0 ,y1,axis=0),np.percentile(T_d0 ,100-y1,axis=0),color=color,alpha=0.5)
        ax33.fill_between(X,np.percentile(T_u1 ,y1,axis=0),np.percentile(T_u1 ,100-y1,axis=0),color=color,alpha=0.5)
        ax34.fill_between(X,np.percentile(T_d1 ,y1,axis=0),np.percentile(T_d1 ,100-y1,axis=0),color=color,alpha=0.5)
        ax35.fill_between(X,np.percentile(T_0  ,y1,axis=0),np.percentile(T_0  ,100-y1,axis=0),color=color,alpha=0.5)
        ax36.fill_between(X,np.percentile(T_1  ,y1,axis=0),np.percentile(T_1  ,100-y1,axis=0),color=color,alpha=0.5)
        ax37.fill_between(X,np.percentile(T_tot,y1,axis=0),np.percentile(T_tot,100-y1,axis=0),color=color,alpha=0.5)

    for ax in [ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax21,ax22,ax23,ax24,ax25,ax26,ax27,ax31,ax32,ax33,ax34,ax35,ax36,ax37]:
        ax.set_xlim(0.2,0.9)
          
        ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.4,0.6,0.8])
 
        ax.axhline(0,0,1,color='black',ls=':',alpha=1.0)

        ax.set_ylim(-0.30,0.30) 

    for ax in [ax12,ax13,ax14,ax15,ax16,ax17,ax22,ax23,ax24,ax25,ax26,ax27,ax32,ax33,ax34,ax35,ax36,ax37]:
        ax.tick_params(labelleft=False)
  
    ax11.text(0.05,0.80,r'\boldmath$F_2^{D}(\delta u_0)/F_2^{D}$'             ,transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.80,r'\boldmath$F_2^{D}(\delta d_0)/F_2^{D}$'             ,transform=ax12.transAxes,size=40)
    ax13.text(0.05,0.80,r'\boldmath$F_2^{D}(\delta u_1)/F_2^{D}$'             ,transform=ax13.transAxes,size=40)
    ax14.text(0.05,0.80,r'\boldmath$F_2^{D}(\delta d_1)/F_2^{D}$'             ,transform=ax14.transAxes,size=40)
    ax15.text(0.05,0.80,r'\boldmath$F_2^{D}(\delta u_0 , \delta d_0)/F_2^{D}$',transform=ax15.transAxes,size=40)
    ax16.text(0.05,0.80,r'\boldmath$F_2^{D}(\delta u_1 , \delta d_1)/F_2^{D}$',transform=ax16.transAxes,size=40)
    ax17.text(0.05,0.80,r'\boldmath$F_2^{D}({\rm all})/F_2^{D}$'              ,transform=ax17.transAxes,size=40)

    ax21.text(0.05,0.80,r'\boldmath$F_2^{h}(\delta u_0)/F_2^{h}$'             ,transform=ax21.transAxes,size=40)
    ax22.text(0.05,0.80,r'\boldmath$F_2^{h}(\delta d_0)/F_2^{h}$'             ,transform=ax22.transAxes,size=40)
    ax23.text(0.05,0.80,r'\boldmath$F_2^{h}(\delta u_1)/F_2^{h}$'             ,transform=ax23.transAxes,size=40)
    ax24.text(0.05,0.80,r'\boldmath$F_2^{h}(\delta d_1)/F_2^{h}$'             ,transform=ax24.transAxes,size=40)
    ax25.text(0.05,0.80,r'\boldmath$F_2^{h}(\delta u_0 , \delta d_0)/F_2^{h}$',transform=ax25.transAxes,size=40)
    ax26.text(0.05,0.80,r'\boldmath$F_2^{h}(\delta u_1 , \delta d_1)/F_2^{h}$',transform=ax26.transAxes,size=40)
    ax27.text(0.05,0.80,r'\boldmath$F_2^{h}({\rm all})/F_2^{h}$'              ,transform=ax27.transAxes,size=40)

    ax31.text(0.05,0.80,r'\boldmath$F_2^{t}(\delta u_0)/F_2^{t}$'             ,transform=ax31.transAxes,size=40)
    ax32.text(0.05,0.80,r'\boldmath$F_2^{t}(\delta d_0)/F_2^{t}$'             ,transform=ax32.transAxes,size=40)
    ax33.text(0.05,0.80,r'\boldmath$F_2^{t}(\delta u_1)/F_2^{t}$'             ,transform=ax33.transAxes,size=40)
    ax34.text(0.05,0.80,r'\boldmath$F_2^{t}(\delta d_1)/F_2^{t}$'             ,transform=ax34.transAxes,size=40)
    ax35.text(0.05,0.80,r'\boldmath$F_2^{t}(\delta u_0 , \delta d_0)/F_2^{t}$',transform=ax35.transAxes,size=40)
    ax36.text(0.05,0.80,r'\boldmath$F_2^{t}(\delta u_1 , \delta d_1)/F_2^{t}$',transform=ax36.transAxes,size=40)
    ax37.text(0.05,0.80,r'\boldmath$F_2^{t}({\rm all})/F_2^{t}$'              ,transform=ax37.transAxes,size=40)
 
    #handles,labels = [],[]
  
    #if 'd' in hand: handles.append(hand['d']['F2'])
    #if 'h' in hand: handles.append(hand['h']['F2'])
    #if 't' in hand: handles.append(hand['t']['F2'])
  
    #if 'd' in hand: labels.append(r'\boldmath$D$')
    #if 'h' in hand: labels.append(r'\boldmath$^3{\rm He}$')
    #if 't' in hand: labels.append(r'\boldmath$^3{\rm H}$')
  
    #ax21.legend(handles,labels,loc=(0.00,0.20),fontsize=30, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)
  
    #ax11.set_ylim( 0.001,0.090)
    #ax11.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax11.yaxis.set_major_locator(MultipleLocator(0.02))

    #ax12.set_ylim( 0.001,0.090)    
    #ax12.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax12.yaxis.set_major_locator(MultipleLocator(0.02))
    #
    #ax13.set_ylim(-0.0025,0.0015)
    #ax13.yaxis.set_minor_locator(MultipleLocator(0.0005))
    #ax13.yaxis.set_major_locator(MultipleLocator(0.001))
  
    #ax14.set_ylim(-0.001,0.0049)
    #ax14.yaxis.set_minor_locator(MultipleLocator(0.0005))
    #ax14.yaxis.set_major_locator(MultipleLocator(0.001))
  
    for ax in [ax31,ax32,ax33,ax34,ax35,ax36]:
        ax.set_xlabel(r'\boldmath$x$' ,size=35)   
        ax.xaxis.set_label_coords(0.97,0)

    #for ax in [ax13,ax14,ax21,ax23,ax24]:
    #    ax.axhline(0,0,1,color='black',ls=':',alpha=1.0)

    #for ax in [ax22]:
    #    ax.axhline(1,0,1,color='black',ls=':',alpha=1.0)

    #ax21.set_ylim(-0.09,0.03)
    #ax21.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax21.yaxis.set_major_locator(MultipleLocator(0.02))
 
    #ax22.set_ylim( 0.81,1.14)
    #ax22.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax22.yaxis.set_major_locator(MultipleLocator(0.05))


    #ax23.set_ylim(-0.09,0.03)
    #ax23.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax23.yaxis.set_major_locator(MultipleLocator(0.02))


    #ax24.set_ylim(-0.01,0.24)
    #ax24.yaxis.set_minor_locator(MultipleLocator(0.01))
    #ax24.yaxis.set_major_locator(MultipleLocator(0.05))


  
    if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
    else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)
  
    py.tight_layout()
    py.subplots_adjust(top=0.99,right=0.99,hspace=0.01,wspace=0.01) 
 
    filename = '%s/gallery/F2A-off-components-ratio-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    if mode==2: filename += '-ci'
  
    filename+='.png'
  
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

if __name__=="__main__":

    
    ap = argparse.ArgumentParser()

    ap.add_argument('task'                ,type=int                       ,help='0 to generate STFs')
    ap.add_argument('-d'   ,'--directory' ,type=str   ,default='unamed'   ,help='directory name to store results')
    ap.add_argument('-Q2'  ,'--Q2'        ,type=float ,default='unamed'   ,help='Q2 value')
    ap.add_argument('-t'   ,'--tar'       ,type=str   ,default='unamed'   ,help='target')
    ap.add_argument('-s'   ,'--stf'       ,type=str   ,default='unamed'   ,help='structure function')
    args = ap.parse_args()

    if args.task==0:
        gen_stf(args.directory,Q2=args.Q2,tar=args.tar,stf=args.stf)

    if args.task==1:
        gen_marathon_stf(args.directory,tar=args.tar)

    if args.task==2:
        gen_off_stf(args.directory,Q2=args.Q2,nucleus=args.tar)









