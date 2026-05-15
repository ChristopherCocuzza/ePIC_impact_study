import sys,os
import numpy as np
import time
import argparse

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import pylab as py

#--from scipy stack 
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

#--polarized
def gen_pstf(wdir,Q2=10,tar='p',stf='g1',LT_only=False,TMC=True,order='NLO'):

    print('\ngenerating pstf from %s for %s %s at Q2 = %3.5f (LT_only=%s, order = %s)'%(wdir,stf,tar,Q2,LT_only,order))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = True

    if LT_only: conf['pidis gXres'] = False
    if TMC==False: conf['pidis tmc'] = False

    conf['order'] = order 
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    #resman.setup_idis()
    resman.setup_pidis()
    pidis = conf['pidis']
    parman=resman.parman

    parman.order=replicas[0]['order'][istep]

    ppdf=conf['ppdf']
    #--setup kinematics
    X=10**np.linspace(-6,-1,200)
    X=np.append(X,np.linspace(0.101,0.99,200))
    Q2 = Q2*np.ones(len(X))

    #--compute d2 for all replicas
    STF=[]
    cnt=0
    n_replicas = len(replicas)
    for i in range(n_replicas):
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        STF.append(pidis.get_gX(stf,X,Q2,tar,idx=None))


    STF = np.array(STF)
    print()
    checkdir('%s/data'%wdir)
    filename='%s/data/pstf-%s-%s-Q2=%3.5f'%(wdir,stf,tar,Q2[0])
    if LT_only: filename += '-LT'
    if TMC==False: filename += '-noTMC'
    if order=='LO': filename += '-LO'
    filename += '.dat'

    save({'X':X,'Q2':Q2[0],'STF':STF},filename)
    print('Saving data to %s'%filename)

def plot_pstf(wdir,Q2=None,mode=0,TAR=['p','n','d','h'],STF=['g1','g2']):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=2,1
  N = nrows*ncols
  fig = py.figure(figsize=(ncols*7,nrows*4))
  axs, axLs = {},{}
  for i in range(N):
      axs[i+1] = py.subplot(nrows,ncols,i+1)
      divider = make_axes_locatable(axs[i+1])
      axLs[i+1] = divider.append_axes("right",size=3.00,pad=0,sharey=axs[i+1])
      axLs[i+1].set_xlim(0.1,0.9)
      axLs[i+1].spines['left'].set_visible(False)
      axLs[i+1].yaxis.set_ticks_position('right')
      py.setp(axLs[i+1].get_xticklabels(),visible=True)

      axs[i+1].spines['right'].set_visible(False)


  load_config('%s/input.py'%wdir)
  istep=core.get_istep()

  if Q2==None: Q2 = conf['Q20']
      
  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  hand = {}
  for tar in TAR:
      for stf in STF:
          filename ='%s/data/pstf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)
          #--load data if it exists
          try:
              data=load(filename)
          #--generate data and then load it if it does not exist
          except:
              gen_pstf(wdir,Q2,tar,stf)
              data=load(filename)

          X    = data['X']
          data = data['XF'] 
          mean = np.mean(data,axis=0)
          std  = np.std (data,axis=0)

          if tar=='p': color='red'
          if tar=='n': color='green'
          if tar=='d': color='blue'
          if tar=='h': color='magenta'

          label = None
          if stf =='g1': ax,axL = axs[1],axLs[1]
          if stf =='g2': ax,axL = axs[2],axLs[2]

          #--plot each replica
          if mode==0:
              for i in range(len(data)):
                  hand[tar] ,= ax .plot(X,data[i],color=color,alpha=0.1)
                  hand[tar] ,= axL.plot(X,data[i],color=color,alpha=0.1)
    
          #--plot average and standard deviation
          if mode==1:
              hand[tar] = ax .fill_between(X,(mean-std),(mean+std),color=color,alpha=0.8)
              hand[tar] = axL.fill_between(X,(mean-std),(mean+std),color=color,alpha=0.8)


  for i in range(N):
        axs[i+1].set_xlim(8e-3,0.1)
        axs[i+1].semilogx()

        axs[i+1].tick_params(axis='both', which='both', top=True, direction='in',labelsize=20)
        axs[i+1].set_xticks([0.01,0.1])
        axs[i+1].set_xticklabels([r'$0.01$',r'$0.1$'])
        axs[i+1].axhline(0,0,1,ls='--',color='black',alpha=0.5)
        axs[i+1].axvline(0.1,0,1,ls=':' ,color='black',alpha=0.5)

        axLs[i+1].set_xlim(0.1,1.0)

        axLs[i+1].tick_params(axis='both', which='both', top=True, right=True, left=False, labelright=False, direction='in',labelsize=20)
        axs[i+1] .tick_params(axis='both', which='minor', length = 3)
        axs[i+1] .tick_params(axis='both', which='major', length = 6)
        axLs[i+1].tick_params(axis='both', which='minor', length = 3)
        axLs[i+1].tick_params(axis='both', which='major', length = 6)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.2)
        axLs[i+1].xaxis.set_minor_locator(minorLocator)
        axLs[i+1].xaxis.set_major_locator(majorLocator)
        axLs[i+1].set_xticks([0.3,0.5,0.7])
        axLs[i+1].set_xticklabels([r'$0.3$',r'$0.5$',r'$0.7$'])
        axLs[i+1].axhline(0,0,1,ls='--',color='black',alpha=0.5)
        axLs[i+1].axvline(0.1,0,1,ls=':' ,color='black',alpha=0.5)


  minorLocator = MultipleLocator(0.01)
  majorLocator = MultipleLocator(0.02)
  axs[1].yaxis.set_minor_locator(minorLocator)
  axs[1].yaxis.set_major_locator(majorLocator)
  minorLocator = MultipleLocator(0.005)
  majorLocator = MultipleLocator(0.01)
  axs[2].yaxis.set_minor_locator(minorLocator)
  axs[2].yaxis.set_major_locator(majorLocator)

  axs[1] .tick_params(labelbottom=False)
  axLs[1].tick_params(labelbottom=False)
  axLs[2].set_xlabel(r'\boldmath$x$' ,size=30)
  axLs[2].xaxis.set_label_coords(0.95,0.00)

  axs[1].text(0.10,0.40,r'\boldmath$xg_1$',transform=axs[1].transAxes,size=40)
  axs[2].text(0.10,0.25,r'\boldmath$xg_2$',transform=axs[2].transAxes,size=40)

  axs[1].set_ylim(-0.025,0.085)
  axs[2].set_ylim(-0.050,0.019)

  axs[1].set_yticks([-0.02,0,0.02,0.04,0.06,0.08])
  axs[2].set_yticks([-0.04,-0.03,-0.02,-0.01,0,0.01])

  if Q2 == 1.27**2: axs[2].text(0.05,0.05,r'$Q^2 = m_c^2$',             transform=axs[2].transAxes,size=30)
  else:             axs[2].text(0.05,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=axs[2].transAxes,size=25)

  handles, labels = [],[]
  if 'p' in hand: handles.append(hand['p'])
  if 'n' in hand: handles.append(hand['n'])
  if 'd' in hand: handles.append(hand['d'])
  if 'h' in hand: handles.append(hand['h'])
  if 'p' in hand: labels.append(r'\boldmath$p$')
  if 'n' in hand: labels.append(r'\boldmath$n$')
  if 'd' in hand: labels.append(r'\boldmath$D$')
  if 'h' in hand: labels.append(r'\boldmath$^3 {\rm He}$')
  axs[1].legend(handles,labels,loc='upper left',fontsize=25, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/pstfs-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

def plot_EMC(wdir,Q2=None,mode=0):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,2
  N = nrows*ncols
  fig = py.figure(figsize=(ncols*7,nrows*4))
  ax11 = py.subplot(nrows,ncols,1)
  ax12 = py.subplot(nrows,ncols,2)


  load_config('%s/input.py'%wdir)
  istep=core.get_istep()

  if Q2==None: Q2 = conf['Q20']
      

  color='red'
  hand = {}
  #--deuteron g1
  g1p = load('%s/data/pstf-p-g1-Q2=%3.5f.dat'%(wdir,Q2))
  g1n = load('%s/data/pstf-n-g1-Q2=%3.5f.dat'%(wdir,Q2))
  g1d = load('%s/data/pstf-d-g1-Q2=%3.5f.dat'%(wdir,Q2))
  g1h = load('%s/data/pstf-h-g1-Q2=%3.5f.dat'%(wdir,Q2))
  #g2p = load('%s/data/pstf-p-g2-Q2=%3.5f.dat'%(wdir,Q2))
  #g2n = load('%s/data/pstf-n-g2-Q2=%3.5f.dat'%(wdir,Q2))
  #g2d = load('%s/data/pstf-d-g2-Q2=%3.5f.dat'%(wdir,Q2))
  #g2h = load('%s/data/pstf-h-g2-Q2=%3.5f.dat'%(wdir,Q2))

  X = g1p['X']
  ratio_g1d = 2*g1d['XF']/(g1p['XF'] + g1n['XF'])
  ratio_g1h = 3*g1h['XF']/(2*g1p['XF'] + g1n['XF'])
  #ratio_g2d = 2*g2d['XF']/(g2p['XF'] + g2n['XF'])
  #ratio_g2h = 3*g2h['XF']/(2*g2p['XF'] + g2n['XF'])


  mean_g1d = np.mean(ratio_g1d,axis=0)
  std_g1d  = np.std (ratio_g1d,axis=0)
  mean_g1h = np.mean(ratio_g1h,axis=0)
  std_g1h  = np.std (ratio_g1h,axis=0)

  #mean_g2d = np.mean(ratio_g2d,axis=0)
  #std_g2d  = np.std (ratio_g2d,axis=0)
  #mean_g2h = np.mean(ratio_g2h,axis=0)
  #std_g2h  = np.std (ratio_g2h,axis=0)

  #--plot each replica
  if mode==0:
      for i in range(len(ratio_g1d)):
          hand['g1d'] ,= ax11.plot(X,ratio_g1d[i],color=color,alpha=1)
          hand['g1h'] ,= ax12.plot(X,ratio_g1h[i],color=color,alpha=1)
          #hand['g2d'] ,= ax12.plot(X,ratio_g2d[i],color=color,alpha=0.1)
          #hand['g2h'] ,= ax22.plot(X,ratio_g2h[i],color=color,alpha=0.1)
  
  #--plot average and standard deviation
  if mode==1:
      hand['g1d'] = ax11.fill_between(X,(mean_g1d-std_g1d),(mean_g1d+std_g1d),color=color,alpha=0.8)
      hand['g1h'] = ax12.fill_between(X,(mean_g1h-std_g1h),(mean_g1h+std_g1h),color=color,alpha=0.8)
      #hand['g2d'] = ax12.fill_between(X,(mean_g2d-std_g2d),(mean_g2d+std_g2d),color=color,alpha=0.8)
      #hand['g2h'] = ax22.fill_between(X,(mean_g2h-std_g2h),(mean_g2h+std_g2h),color=color,alpha=0.8)


  for ax in [ax11,ax12]:
        ax.set_xlim(0.1,0.9)

        ax.tick_params(axis='both', which='both', top=True, direction='in',labelsize=20)
        #ax.set_xticks([0.01,0.1])
        #ax.set_xticklabels([r'$0.01$',r'$0.1$'])
        ax.axhline(1,0,1,ls='--',color='black',alpha=0.5)
        ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)

        ax.tick_params(axis='both', which='both', top=True, right=True, left=False, labelright=False, direction='in',labelsize=20)
        ax .tick_params(axis='both', which='minor', length = 3)
        ax .tick_params(axis='both', which='major', length = 6)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        #ax.set_xticks([0.3,0.5,0.7])
        #ax.set_xticklabels([r'$0.3$',r'$0.5$',r'$0.7$'])

  ax11.set_ylim( 0.80,1.10)
  ax12.set_ylim(-0.40,0.20)

  #minorLocator = MultipleLocator(0.01)
  #majorLocator = MultipleLocator(0.02)
  #axs[1].yaxis.set_minor_locator(minorLocator)
  #axs[1].yaxis.set_major_locator(majorLocator)
  #minorLocator = MultipleLocator(0.005)
  #majorLocator = MultipleLocator(0.01)
  #axs[2].yaxis.set_minor_locator(minorLocator)
  #axs[2].yaxis.set_major_locator(majorLocator)

  ax11.set_xlabel(r'\boldmath$x$' ,size=30)
  ax11.xaxis.set_label_coords(0.95,0.00)
  ax12.set_xlabel(r'\boldmath$x$' ,size=30)
  ax12.xaxis.set_label_coords(0.95,0.00)

  ax11.text(0.02,0.80,r'\boldmath$R(g_1^{D})$'         ,transform=ax11.transAxes,size=40)
  ax12.text(0.02,0.80,r'\boldmath$R(g_1^{^3{\rm He}})$',transform=ax12.transAxes,size=40)

  #axs[1].set_ylim(-0.025,0.085)
  #axs[2].set_ylim(-0.050,0.019)

  #axs[1].set_yticks([-0.02,0,0.02,0.04,0.06,0.08])
  #axs[2].set_yticks([-0.04,-0.03,-0.02,-0.01,0,0.01])

  ax11.text(0.05,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)

  #handles, labels = [],[]
  #if 'p' in hand: handles.append(hand['p'])
  #if 'n' in hand: handles.append(hand['n'])
  #if 'd' in hand: handles.append(hand['d'])
  #if 'h' in hand: handles.append(hand['h'])
  #if 'p' in hand: labels.append(r'\boldmath$p$')
  #if 'n' in hand: labels.append(r'\boldmath$n$')
  #if 'd' in hand: labels.append(r'\boldmath$D$')
  #if 'h' in hand: labels.append(r'\boldmath$^3 {\rm He}$')
  #axs[1].legend(handles,labels,loc='upper left',fontsize=25, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/pstfs-EMC-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

#--higher twists
def gen_gXres(wdir,Q2=10,tar='p',stf='g1'):

    print('\ngenerating gXres from %s for %s %s at Q2 = %3.5f'%(wdir,stf,tar,Q2))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = True
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = True

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    resman.setup_pidis()
    pidis = conf['pidis']
    parman=resman.parman

    parman.order=replicas[0]['order'][istep]

    gXres = conf['gXres']      
    #--setup kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.99,100))
    Q2 = Q2*np.ones(len(X))

    #--compute d2 for all replicas
    STF=[]
    cnt=0
    n_replicas = len(replicas)
    for i in range(n_replicas):
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        STF.append(gXres.get_gXres(stf,tar,X,Q2))


    STF = np.array(STF)
    print()
    checkdir('%s/data'%wdir)
    filename='%s/data/gXres-%s-%s-Q2=%3.5f.dat'%(wdir,stf,tar,Q2[0])

    save({'X':X,'Q2':Q2[0],'STF':STF},filename)
    print('Saving data to %s'%filename)

#--with fixed W2 rather than Q2
def gen_gXres_W2(wdir,W2=4,tar='p',stf='g1'):

    print('\ngenerating gXres from %s for %s %s at W2 = %3.5f'%(wdir,stf,tar,W2))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = True
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = True

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    resman.setup_pidis()
    pidis = conf['pidis']
    parman=resman.parman

    parman.order=replicas[0]['order'][istep]

    gXres = conf['gXres']
    M2 = conf['aux'].M2
    #--setup kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.99,100))
    Q2 = X/(1-X)*(W2-M2)

    #--compute d2 for all replicas
    STF=[]
    cnt=0
    n_replicas = len(replicas)
    for i in range(n_replicas):
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        STF.append(gXres.get_gXres(stf,tar,X,Q2))


    STF = np.array(STF)
    print()
    checkdir('%s/data'%wdir)
    filename='%s/data/gXres-%s-%s-W2=%3.5f.dat'%(wdir,stf,tar,W2)

    save({'X':X,'Q2':Q2[0],'STF':STF},filename)
    print('Saving data to %s'%filename)

#--polarized moments
def gen_d2(wdir,tar='p',LT_only=False,TMC=True):

    print('\ngenerating d2 from %s for %s (LT_only=%s, TMC=%s)'%(wdir,tar,LT_only,TMC))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = True
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = True
  
    if LT_only: conf['pidis gXres'] = False
    if TMC==False: conf['pidis tmc'] = False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    #resman.setup_idis()
    resman.setup_pidis()
    pidis = conf['pidis']
    parman=resman.parman

    parman.order=replicas[0]['order'][istep]

    ppdf=conf['ppdf']
    #--setup kinematics
    #--Gaussian quadrature
    npts = 99
    z,w = np.polynomial.legendre.leggauss(npts) 
    jac = 0.5
    x   = 0.5*(z+1)
    q2 = np.linspace(1.27**2,10,20)

    X,Q2 = np.meshgrid(x,q2)

    shape = Q2.shape

    X  = X.flatten()
    Q2 = Q2.flatten()

    gXres = conf['gXres']
    #--compute d2 for all replicas
    D2=[]
    cnt=0
    n_replicas = len(replicas)
    for i in range(n_replicas):
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        g1   =  X**2*pidis.get_gX('g1',X,Q2,tar,idx=None)
        g2   =  X**2*pidis.get_gX('g2',X,Q2,tar,idx=None)
        func = 2*g1 + 3*g2

        func = func.reshape(shape)

        d2 = np.sum(w*jac*func,axis=1)
        #d2 = [np.sum(w*jac*func[i]) for i in range(len(func))]
        D2.append(d2)

    print()
    checkdir('%s/data'%wdir)
    filename='%s/data/d2-%s'%(wdir,tar)
    if LT_only: filename += '-LT'
    if TMC==False: filename += '-noTMC'
    filename += '.dat'

    save({'Q2':q2,'D2':D2},filename)
    print('Saving data to %s'%filename)
        
def gen_d2_trunc_dep(wdir, tar='p', Q2 = 10, xmin = 1e-2, xmax = 0.98, LT_only = False):
    ## get truncated second moment integrated from a range of x_min to 1
    load_config('%s/input.py' % wdir)
    istep = core.get_istep()

    replicas = core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0])
    ## 'conf' will be modified for each replica individually later in the loop over 'replicas'
    ## the reason for doing this is that 'fix parameters' has to be set correctly for each replica

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
            print('ppdf-proton not an active or passive distribution')
            return

    conf['bootstrap'] = False
    resman = RESMAN(nworkers = 1, parallel = False, datasets = False)
    resman.setup_idis()
    resman.setup_pidis()
    parman = resman.parman
    parman.order = replicas[0]['order'][istep]
    ## make sure 'parman' uses the same order for active distributions as all the replicas do

    pidis = conf['pidis']

    ## setup kinematics
    xs = np.geomspace(xmin,0.1,100)
    xs = np.append(xs, np.linspace(0.1, xmax, 100))
    if Q2 == None: Q2 = conf['Q20']
    print('\ngenerating d2 from %s at Q2 = %3.2f from %3.6f to %3.6f' % (wdir, Q2, xmin, xmax))

    ## compute moments for all replicas
    moments = []
    n_replicas = len(replicas)

    for i in range(n_replicas): ## using 'scipy.integrate.cumtrapz' takes about 9.984 seconds for 100 x points, 4 flavors and 516 replicas
        lprint('%d/%d' % (i + 1, n_replicas))

        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        g1   =  xs**2*pidis.get_gX('g1',xs,Q2*np.ones(len(xs)),tar=tar,idx=None)
        g2   =  xs**2*pidis.get_gX('g2',xs,Q2*np.ones(len(xs)),tar=tar,idx=None)
        function_values =  2*g1 + 3*g2

        moment_temp = cumulative_trapezoid(function_values, xs, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        moments.append(moment_max - moment_temp)

    moments = np.array(moments)
    checkdir('%s/data' % wdir)
    filename = '%s/data/d2-trunc-%s-Q2=%3.5f-xmin=%3.5f-xmax=%3.5f.dat' % (wdir,tar,Q2,xmin,xmax)
    save({'X': xs, 'Q2': Q2, 'moments': moments}, filename)
    print('Saving data to %s'%filename)

def gen_d2_trunc_funcQ2(wdir, tar='p', xmin = 5e-3, xmax = 0.73,LT_only=False):
    ## get truncated second moment integrated from a range of x_min to 1
    load_config('%s/input.py' % wdir)
    istep = core.get_istep()


    replicas = core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
            print('ppdf-proton not an active or passive distribution')
            return

    if LT_only: conf['pidis gXres'] = False
    resman = RESMAN(nworkers = 1, parallel = False, datasets = False)
    resman.setup_idis()
    resman.setup_pidis()
    parman = resman.parman
    parman.order = replicas[0]['order'][istep]
    ## make sure 'parman' uses the same order for active distributions as all the replicas do

    pidis = conf['pidis']

    ## setup kinematics
    xs = np.geomspace(xmin,0.1,100)
    xs = np.append(xs, np.linspace(0.1, xmax, 100))
    Q2 = np.linspace(1.27**2,10,20)
    print('\ngenerating d2 from %s at from %3.6f to %3.6f (LT_only=%s)' % (wdir, xmin, xmax, LT_only))

    ## compute moments for all replicas
    n_replicas = len(replicas)
    moments = np.zeros((n_replicas,len(Q2)))

    for i in range(n_replicas): ## using 'scipy.integrate.cumtrapz' takes about 9.984 seconds for 100 x points, 4 flavors and 516 replicas
        lprint('%d/%d' % (i + 1, n_replicas))

        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        for j in range(len(Q2)):
            g1   =  xs**2*pidis.get_gX('g1',xs,Q2[j]*np.ones(len(xs)),tar=tar,idx=None)
            g2   =  xs**2*pidis.get_gX('g2',xs,Q2[j]*np.ones(len(xs)),tar=tar,idx=None)
            function_values =  2*g1 + 3*g2

            moment_temp = cumulative_trapezoid(function_values, xs, initial = 0.0)
            moment_temp = np.array(moment_temp)
            moment_max = moment_temp[-1]
            moments[i][j] = moment_max

    checkdir('%s/data' % wdir)
    filename = '%s/data/d2-trunc-funcQ2-%s-xmin=%3.5f-xmax=%3.5f' % (wdir,tar,xmin,xmax)
    if LT_only: filename+='-LT'
    filename += '.dat'
    save({'X': xs, 'Q2': Q2, 'D2': moments}, filename)
    print('Saving data to %s'%filename)

#--BC sum rule
def gen_BCSR(wdir,tar='p'):

    print('\ngenerating BCSR from %s for %s'%(wdir,tar))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = True
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = True
  
    #conf['pidis tmc'] = False
    #conf['pidis gXres'] = False 
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    resman.setup_pidis()
    pidis = conf['pidis']
    parman=resman.parman

    parman.order=replicas[0]['order'][istep]

    ppdf=conf['ppdf']
    #--setup kinematics
    #--Gaussian quadrature
    npts = 99
    z,w = np.polynomial.legendre.leggauss(npts) 
    jac = 0.5
    x   = 0.5*(z+1)
    q2 = np.linspace(1.27**2,10,20)

    X,Q2 = np.meshgrid(x,q2)

    shape = Q2.shape

    X  = X.flatten()
    Q2 = Q2.flatten()

    gXres = conf['gXres']
    #--compute d2 for all replicas
    SR=[]
    cnt=0
    n_replicas = len(replicas)
    for i in range(n_replicas):
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        func =  pidis.get_gX('g2',X,Q2,tar=tar,idx=None)

        func = func.reshape(shape)

        sr = np.sum(w*jac*func,axis=1)
        SR.append(sr)

    print()
    checkdir('%s/data'%wdir)
    filename='%s/data/pstf-BCSR-%s.dat'%(wdir,tar)

    save({'Q2':q2,'SR':SR},filename)
    print('Saving data to %s'%filename)

def gen_BCSR_trunc(wdir,tar='p',xmin=5e-3,W2=4,LT_only=False):

    print('\ngenerating BCSR from %s for %s from xmin = %3.5f at W2 = %3.5f (LT_only=%s)'%(wdir,tar,xmin,W2,LT_only))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = True
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = True
  
    #conf['pidis tmc'] = False
    if LT_only: conf['pidis gXres'] = False 
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    resman.setup_pidis()
    pidis = conf['pidis']
    parman=resman.parman

    parman.order=replicas[0]['order'][istep]

    ppdf=conf['ppdf']
    #--setup kinematics
    Q2 = np.linspace(1.27**2,10,20)
    xmax = Q2/(W2 + Q2 - 0.938**2)
    #--Gaussian quadrature
    npts = 99
    z,w = np.polynomial.legendre.leggauss(npts) 
    jac = 0.5
    X   = 0.5*np.einsum('z,x->xz',np.ones(len(z)),xmin+xmax) + 0.5*np.einsum('z,x->xz',z,(xmax-xmin))

    shape = X.shape
  
    gXres = conf['gXres']
    #--compute d2 for all replicas
    cnt=0
    n_replicas = len(replicas)
    SR=np.zeros((n_replicas,len(Q2)))
    for i in range(n_replicas):
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        sr = np.zeros(len(Q2))
        for j in range(len(Q2)):
            func =  pidis.get_gX('g2',X[j],Q2[j]*np.ones(len(z)),tar,idx=None)
            sr[j] = np.sum(w*jac*func,axis=0)

        SR[i] = sr
    print()
    checkdir('%s/data'%wdir)
    filename='%s/data/pstf-BCSR-trunc-%s-xmin=%3.5f-W2=%3.5f'%(wdir,tar,xmin,W2)
    if LT_only: filename += '-LT'
    filename += '.dat'

    save({'Q2':Q2,'SR':SR},filename)
    print('Saving data to %s'%filename)










