import sys, os
import numpy as np
import copy
import pandas as pd
import scipy as sp

## matplotlib
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter ## for minor ticks in x label
matplotlib.rc('text', usetex = True)
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.pyplot as py
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec

from scipy.stats import norm


#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from tools
from tools.tools import load,save,checkdir
from tools.config import conf, load_config

def plot_obs(wdir):

    for year in [2016,2025]:
        plot_COMPASS_pion_deuteron  (wdir,'pi+',year=year)
        plot_COMPASS_pion_deuteron  (wdir,'pi-',year=year)
        plot_COMPASS_kaon_deuteron  (wdir,'K+', year=year)
        plot_COMPASS_kaon_deuteron  (wdir,'K-', year=year)
        plot_COMPASS_hadron_deuteron(wdir,'h+', year=year)
        plot_COMPASS_hadron_deuteron(wdir,'h-', year=year)

        plot_COMPASS_pion_deuteron_ratio  (wdir,'pi+',year=year)
        plot_COMPASS_pion_deuteron_ratio  (wdir,'pi-',year=year)
        plot_COMPASS_kaon_deuteron_ratio  (wdir,'K+', year=year)
        plot_COMPASS_kaon_deuteron_ratio  (wdir,'K-', year=year)
        plot_COMPASS_hadron_deuteron_ratio(wdir,'h+', year=year)
        plot_COMPASS_hadron_deuteron_ratio(wdir,'h-', year=year)

        if year==2016: continue

        plot_COMPASS_pion_proton  (wdir,'pi+',year=year)
        plot_COMPASS_pion_proton  (wdir,'pi-',year=year)
        plot_COMPASS_kaon_proton  (wdir,'K+', year=year)
        plot_COMPASS_kaon_proton  (wdir,'K-', year=year)
        plot_COMPASS_hadron_proton(wdir,'h+', year=year)
        plot_COMPASS_hadron_proton(wdir,'h-', year=year)

        plot_COMPASS_pion_proton_ratio  (wdir,'pi+',year=year)
        plot_COMPASS_pion_proton_ratio  (wdir,'pi-',year=year)
        plot_COMPASS_kaon_proton_ratio  (wdir,'K+', year=year)
        plot_COMPASS_kaon_proton_ratio  (wdir,'K-', year=year)
        plot_COMPASS_hadron_proton_ratio(wdir,'h+', year=year)
        plot_COMPASS_hadron_proton_ratio(wdir,'h-', year=year)
    
    #plot_pion_shift(wdir,'pi+',year=COMPASS_year)
    #plot_pion_shift(wdir,'pi-',year=COMPASS_year)

#DEUTERON
def plot_COMPASS_pion_deuteron(wdir,hadron='pi+',year=2025):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    #labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    #cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if year==2016:
        if hadron=='pi+' and 1005 not in predictions['reactions']['sidis']: return 
        if hadron=='pi-' and 1006 not in predictions['reactions']['sidis']: return 
    if year==2025:
        if hadron=='pi+' and 1105 not in predictions['reactions']['sidis']: return 
        if hadron=='pi-' and 1106 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2016:
            if hadron=='pi+': D=pd.DataFrame(data[1005]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='pi-': D=pd.DataFrame(data[1006]).query('X>%f and X<%f'%(xmin,xmax))
        if year==2025:
            if hadron=='pi+': D=pd.DataFrame(data[1105]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='pi-': D=pd.DataFrame(data[1106]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:  color='r'
            elif ymin==0.15: color='darkorange'
            elif ymin==0.2: color='g'
            elif ymin==0.3: color='c'
            elif ymin==0.5: color='m'
            else: color='k'

            p=ax.errorbar(d.Z,d.value+f,d.alpha,fmt='.',color=color,zorder=1.1)

            thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.set_ylim(0.00,6.5)
        ax.set_xlim(0.15,0.85)
        #ax.semilogy()
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.yaxis.set_minor_locator(MultipleLocator(0.5)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='pi+': ax.set_ylabel(r'\boldmath$dM_D^{\pi^+}/dz_h + i$',size=30)
            if hadron=='pi-': ax.set_ylabel(r'\boldmath$dM_D^{\pi^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.33,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15,0.2,0.3,0.5]:
        if _ in H:
            handles.append((thy[_],H[_]))
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1

    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_COMPASS%d_deuteron_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d pion deuteron figure to %s'%(year,filename))

def plot_COMPASS_kaon_deuteron(wdir,hadron='K+',year=2025):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    #labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    #cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if year==2016:
        if hadron=='K+' and 2005 not in predictions['reactions']['sidis']: return 
        if hadron=='K-' and 2006 not in predictions['reactions']['sidis']: return 
    if year==2025:
        if hadron=='K+' and 2105 not in predictions['reactions']['sidis']: return 
        if hadron=='K-' and 2106 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2016:
            if hadron=='K+': D=pd.DataFrame(data[2005]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='K-': D=pd.DataFrame(data[2006]).query('X>%f and X<%f'%(xmin,xmax))
        if year==2025:
            if hadron=='K+': D=pd.DataFrame(data[2105]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='K-': D=pd.DataFrame(data[2106]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:  color='r'
            elif ymin==0.15: color='darkorange'
            elif ymin==0.2: color='g'
            elif ymin==0.3: color='c'
            elif ymin==0.5: color='m'
            else: color='k'

            p=ax.errorbar(d.Z,d.value+f,d.alpha,fmt='.',color=color,zorder=1.1)

            thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.set_ylim(0.00,4.90)
        ax.set_xlim(0.15,0.85)
        #ax.semilogy()
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.yaxis.set_minor_locator(MultipleLocator(0.5)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='K+': ax.set_ylabel(r'\boldmath$dM_D^{K^+}/dz_h + i$',size=30)
            if hadron=='K-': ax.set_ylabel(r'\boldmath$dM_D^{K^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.33,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15,0.2,0.3,0.5]:
        if _ in H:
            handles.append((thy[_],H[_]))
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1

    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_COMPASS%d_deuteron_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d kaon deuteron figure to %s'%(year,filename))

def plot_COMPASS_hadron_deuteron(wdir,hadron='h+',year=2025):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    #labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    #cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if year==2016:
        if hadron=='h+' and 3000 not in predictions['reactions']['sidis']: return 
        if hadron=='h-' and 3001 not in predictions['reactions']['sidis']: return 
    if year==2025:
        if hadron=='h+' and 3100 not in predictions['reactions']['sidis']: return 
        if hadron=='h-' and 3101 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2016:
            if hadron=='h+': D=pd.DataFrame(data[3000]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='h-': D=pd.DataFrame(data[3001]).query('X>%f and X<%f'%(xmin,xmax))
        if year==2025:
            if hadron=='h+': D=pd.DataFrame(data[3100]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='h-': D=pd.DataFrame(data[3101]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:  color='r'
            elif ymin==0.15: color='darkorange'
            elif ymin==0.2: color='g'
            elif ymin==0.3: color='c'
            elif ymin==0.5: color='m'
            else: color='k'

            p=ax.errorbar(d.Z,d.value+f,d.alpha,fmt='.',color=color,zorder=1.1)

            thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.set_ylim(0.00,6.90)
        ax.set_xlim(0.15,0.85)
        #ax.semilogy()
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.yaxis.set_minor_locator(MultipleLocator(0.5)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='h+': ax.set_ylabel(r'\boldmath$dM_D^{h^+}/dz_h + i$',size=30)
            if hadron=='h-': ax.set_ylabel(r'\boldmath$dM_D^{h^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.33,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15,0.2,0.3,0.5]:
        if _ in H:
            handles.append((thy[_],H[_]))
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1

    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_COMPASS%d_deuteron_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d hadron deuteron figure to %s'%(year,filename))

#PROTON
def plot_COMPASS_pion_proton(wdir,hadron='pi+',year=2025):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    #labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    #cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if year==2025:
        if hadron=='pi+' and 1115 not in predictions['reactions']['sidis']: return 
        if hadron=='pi-' and 1116 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2025:
            if hadron=='pi+': D=pd.DataFrame(data[1115]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='pi-': D=pd.DataFrame(data[1116]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:  color='r'
            elif ymin==0.15: color='darkorange'
            elif ymin==0.2: color='g'
            elif ymin==0.3: color='c'
            elif ymin==0.5: color='m'
            else: color='k'

            p=ax.errorbar(d.Z,d.value+f,d.alpha,fmt='.',color=color,zorder=1.1)

            thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.set_ylim(0.00,6.5)
        ax.set_xlim(0.15,0.85)
        #ax.semilogy()
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.yaxis.set_minor_locator(MultipleLocator(0.5)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='pi+': ax.set_ylabel(r'\boldmath$dM_p^{\pi^+}/dz_h + i$',size=30)
            if hadron=='pi-': ax.set_ylabel(r'\boldmath$dM_p^{\pi^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.33,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15,0.2,0.3,0.5]:
        if _ in H:
            handles.append((thy[_],H[_]))
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1

    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_COMPASS%d_proton_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d pion proton figure to %s'%(year,filename))

def plot_COMPASS_kaon_proton(wdir,hadron='K+',year=2025):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    #labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    #cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if year==2025:
        if hadron=='K+' and 2115 not in predictions['reactions']['sidis']: return 
        if hadron=='K-' and 2116 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2025:
            if hadron=='K+': D=pd.DataFrame(data[2115]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='K-': D=pd.DataFrame(data[2116]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:  color='r'
            elif ymin==0.15: color='darkorange'
            elif ymin==0.2: color='g'
            elif ymin==0.3: color='c'
            elif ymin==0.5: color='m'
            else: color='k'

            p=ax.errorbar(d.Z,d.value+f,d.alpha,fmt='.',color=color,zorder=1.1)

            thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.set_ylim(0.00,4.90)
        ax.set_xlim(0.15,0.85)
        #ax.semilogy()
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.yaxis.set_minor_locator(MultipleLocator(0.5)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='K+': ax.set_ylabel(r'\boldmath$dM_p^{K^+}/dz_h + i$',size=30)
            if hadron=='K-': ax.set_ylabel(r'\boldmath$dM_p^{K^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.33,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15,0.2,0.3,0.5]:
        if _ in H:
            handles.append((thy[_],H[_]))
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1

    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_COMPASS%d_proton_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d kaon proton figure to %s'%(year,filename))

def plot_COMPASS_hadron_proton(wdir,hadron='h+',year=2025):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    #labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    #cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if year==2025:
        if hadron=='h+' and 3110 not in predictions['reactions']['sidis']: return 
        if hadron=='h-' and 3111 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2025:
            if hadron=='h+': D=pd.DataFrame(data[3110]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='h-': D=pd.DataFrame(data[3111]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:  color='r'
            elif ymin==0.15: color='darkorange'
            elif ymin==0.2: color='g'
            elif ymin==0.3: color='c'
            elif ymin==0.5: color='m'
            else: color='k'

            p=ax.errorbar(d.Z,d.value+f,d.alpha,fmt='.',color=color,zorder=1.1)

            thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.set_ylim(0.00,6.90)
        ax.set_xlim(0.15,0.85)
        #ax.semilogy()
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.yaxis.set_minor_locator(MultipleLocator(0.5)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='h+': ax.set_ylabel(r'\boldmath$dM_p^{h^+}/dz_h + i$',size=30)
            if hadron=='h-': ax.set_ylabel(r'\boldmath$dM_p^{h^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.33,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15,0.2,0.3,0.5]:
        if _ in H:
            handles.append((thy[_],H[_]))
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1

    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_COMPASS%d_proton_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d hadron proton figure to %s'%(year,filename))


#DEUTERON (RATIO PLOTS)
def plot_COMPASS_pion_deuteron_ratio(wdir,hadron='pi+',year=2016):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sidis' not in predictions['reactions']: return
    if year==2016:
        if hadron=='pi+' and 1005 not in predictions['reactions']['sidis']: return 
        if hadron=='pi-' and 1006 not in predictions['reactions']['sidis']: return 
    if year==2025:
        if hadron=='pi+' and 1105 not in predictions['reactions']['sidis']: return 
        if hadron=='pi-' and 1106 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2016:
            if hadron=='pi+': D=pd.DataFrame(data[1005]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='pi-': D=pd.DataFrame(data[1006]).query('X>%f and X<%f'%(xmin,xmax))
        if year==2025:
            if hadron=='pi+': D=pd.DataFrame(data[1105]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='pi-': D=pd.DataFrame(data[1106]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:    dz,color=0.00,'r'
            elif ymin==0.15: dz,color=0.01,'darkorange'
            elif ymin==0.2:  dz,color=0.02,'g'
            elif ymin==0.3:  dz,color=0.03,'c'
            elif ymin==0.5:  dz,color=0.04,'m'
            else: color='k'

            p=ax.errorbar(d.Z+dz,d.value/d.thy,d.alpha/d.thy,fmt='.',color=color,zorder=1.1)

            #thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.axhline(1,0,1,color='black',alpha=0.9,ls='-')
        if cnt in [1,2,3,4,5,6,7,8]:
            ax.set_ylim(0.71,1.29)
            #ax.set_yticks([1,1.1,1.2,1.3,1.4])
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        #if cnt in [5,6,7,8]:
        #    ax.set_ylim(0.75,1.99)
        #    ax.set_yticks([0.8,1,1.2,1.4,1.6,1.8])
        #    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.set_xlim(0.15,0.85)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='pi+': ax.set_ylabel(r'\boldmath$dM_D^{\pi^+}/dz_h + i$',size=30)
            if hadron=='pi-': ax.set_ylabel(r'\boldmath$dM_D^{\pi^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.03,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:ax2=ax
        if cnt==3:ax3=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.2,0.3]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax2.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.5]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax3.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_ratio_COMPASS%d_deuteron_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d ratio deuteron pion figure to %s'%(year,filename))

def plot_COMPASS_kaon_deuteron_ratio(wdir,hadron='K+',year=2016):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sidis' not in predictions['reactions']: return
    if year==2016:
        if hadron=='K+' and 2005 not in predictions['reactions']['sidis']: return 
        if hadron=='K-' and 2006 not in predictions['reactions']['sidis']: return 
    if year==2025:
        if hadron=='K+' and 2105 not in predictions['reactions']['sidis']: return 
        if hadron=='K-' and 2106 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2016:
            if hadron=='K+': D=pd.DataFrame(data[2005]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='K-': D=pd.DataFrame(data[2006]).query('X>%f and X<%f'%(xmin,xmax))
        if year==2025:
            if hadron=='K+': D=pd.DataFrame(data[2105]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='K-': D=pd.DataFrame(data[2106]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:    dz,color=0.00,'r'
            elif ymin==0.15: dz,color=0.01,'darkorange'
            elif ymin==0.2:  dz,color=0.02,'g'
            elif ymin==0.3:  dz,color=0.03,'c'
            elif ymin==0.5:  dz,color=0.04,'m'
            else: color='k'

            p=ax.errorbar(d.Z+dz,d.value/d.thy,d.alpha/d.thy,fmt='.',color=color,zorder=1.1)

            #thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.axhline(1,0,1,color='black',alpha=0.9,ls='-')
        if cnt in [1,2,3,4,5,6,7,8]:
            ax.set_ylim(0.71,1.29)
            #ax.set_yticks([1,1.1,1.2,1.3,1.4])
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        #if cnt in [5,6,7,8]:
        #    ax.set_ylim(0.75,1.99)
        #    ax.set_yticks([0.8,1,1.2,1.4,1.6,1.8])
        #    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.set_xlim(0.15,0.85)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='K+': ax.set_ylabel(r'\boldmath$dM_D^{K^+}/dz_h + i$',size=30)
            if hadron=='K-': ax.set_ylabel(r'\boldmath$dM_D^{K^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.03,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:ax2=ax
        if cnt==3:ax3=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.2,0.3]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax2.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.5]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax3.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_ratio_COMPASS%d_deuteron_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d ratio deuteron kaon figure to %s'%(year,filename))

def plot_COMPASS_hadron_deuteron_ratio(wdir,hadron='h+',year=2016):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sidis' not in predictions['reactions']: return
    if year==2016:
        if hadron=='h+' and 3000 not in predictions['reactions']['sidis']: return 
        if hadron=='h-' and 3001 not in predictions['reactions']['sidis']: return 
    if year==2025:
        if hadron=='h+' and 3100 not in predictions['reactions']['sidis']: return 
        if hadron=='h-' and 3101 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return



    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2016:
            if hadron=='h+': D=pd.DataFrame(data[3000]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='h-': D=pd.DataFrame(data[3001]).query('X>%f and X<%f'%(xmin,xmax))
        if year==2025:
            if hadron=='h+': D=pd.DataFrame(data[3100]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='h-': D=pd.DataFrame(data[3101]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:    dz,color=0.00,'r'
            elif ymin==0.15: dz,color=0.01,'darkorange'
            elif ymin==0.2:  dz,color=0.02,'g'
            elif ymin==0.3:  dz,color=0.03,'c'
            elif ymin==0.5:  dz,color=0.04,'m'
            else: color='k'

            p=ax.errorbar(d.Z+dz,d.value/d.thy,d.alpha/d.thy,fmt='.',color=color,zorder=1.1)

            #thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.axhline(1,0,1,color='black',alpha=0.9,ls='-')
        if cnt in [1,2,3,4,5,6,7,8]:
            ax.set_ylim(0.71,1.29)
            #ax.set_yticks([1,1.1,1.2,1.3,1.4])
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        #if cnt in [5,6,7,8]:
        #    ax.set_ylim(0.75,1.99)
        #    ax.set_yticks([0.8,1,1.2,1.4,1.6,1.8])
        #    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.set_xlim(0.15,0.85)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='h+': ax.set_ylabel(r'\boldmath$dM_D^{h^+}/dz_h + i$',size=30)
            if hadron=='h-': ax.set_ylabel(r'\boldmath$dM_D^{h^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.03,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:ax2=ax
        if cnt==3:ax3=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.2,0.3]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax2.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.5]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax3.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_ratio_COMPASS%d_deuteron_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d ratio deuteron hadron figure to %s'%(year,filename))

#PROTON (RATIO PLOTS)
def plot_COMPASS_pion_proton_ratio(wdir,hadron='pi+',year=2016):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sidis' not in predictions['reactions']: return
    if year==2025:
        if hadron=='pi+' and 1115 not in predictions['reactions']['sidis']: return 
        if hadron=='pi-' and 1116 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2025:
            if hadron=='pi+': D=pd.DataFrame(data[1115]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='pi-': D=pd.DataFrame(data[1116]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:    dz,color=0.00,'r'
            elif ymin==0.15: dz,color=0.01,'darkorange'
            elif ymin==0.2:  dz,color=0.02,'g'
            elif ymin==0.3:  dz,color=0.03,'c'
            elif ymin==0.5:  dz,color=0.04,'m'
            else: color='k'

            p=ax.errorbar(d.Z+dz,d.value/d.thy,d.alpha/d.thy,fmt='.',color=color,zorder=1.1)

            #thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.axhline(1,0,1,color='black',alpha=0.9,ls='-')
        if cnt in [1,2,3,4,5,6,7,8]:
            ax.set_ylim(0.71,1.29)
            #ax.set_yticks([1,1.1,1.2,1.3,1.4])
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        #if cnt in [5,6,7,8]:
        #    ax.set_ylim(0.75,1.99)
        #    ax.set_yticks([0.8,1,1.2,1.4,1.6,1.8])
        #    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.set_xlim(0.15,0.85)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='pi+': ax.set_ylabel(r'\boldmath$dM_p^{\pi^+}/dz_h + i$',size=30)
            if hadron=='pi-': ax.set_ylabel(r'\boldmath$dM_p^{\pi^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.03,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:ax2=ax
        if cnt==3:ax3=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.2,0.3]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax2.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.5]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax3.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_ratio_COMPASS%d_proton_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d ratio proton pion figure to %s'%(year,filename))

def plot_COMPASS_kaon_proton_ratio(wdir,hadron='K+',year=2016):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sidis' not in predictions['reactions']: return
    if year==2025:
        if hadron=='K+' and 2115 not in predictions['reactions']['sidis']: return 
        if hadron=='K-' and 2116 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2025:
            if hadron=='K+': D=pd.DataFrame(data[2115]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='K-': D=pd.DataFrame(data[2116]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:    dz,color=0.00,'r'
            elif ymin==0.15: dz,color=0.01,'darkorange'
            elif ymin==0.2:  dz,color=0.02,'g'
            elif ymin==0.3:  dz,color=0.03,'c'
            elif ymin==0.5:  dz,color=0.04,'m'
            else: color='k'

            p=ax.errorbar(d.Z+dz,d.value/d.thy,d.alpha/d.thy,fmt='.',color=color,zorder=1.1)

            #thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.axhline(1,0,1,color='black',alpha=0.9,ls='-')
        if cnt in [1,2,3,4,5,6,7,8]:
            ax.set_ylim(0.71,1.29)
            #ax.set_yticks([1,1.1,1.2,1.3,1.4])
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        #if cnt in [5,6,7,8]:
        #    ax.set_ylim(0.75,1.99)
        #    ax.set_yticks([0.8,1,1.2,1.4,1.6,1.8])
        #    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.set_xlim(0.15,0.85)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='K+': ax.set_ylabel(r'\boldmath$dM_p^{K^+}/dz_h + i$',size=30)
            if hadron=='K-': ax.set_ylabel(r'\boldmath$dM_p^{K^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.03,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:ax2=ax
        if cnt==3:ax3=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.2,0.3]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax2.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.5]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax3.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_ratio_COMPASS%d_proton_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d ratio proton kaon figure to %s'%(year,filename))

def plot_COMPASS_hadron_proton_ratio(wdir,hadron='h+',year=2016):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sidis' not in predictions['reactions']: return
    if year==2025:
        if hadron=='h+' and 3110 not in predictions['reactions']['sidis']: return 
        if hadron=='h-' and 3111 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:
            if 'ymin' in data[_]: data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])
            else: return


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if year==2025:
            if hadron=='h+': D=pd.DataFrame(data[3110]).query('X>%f and X<%f'%(xmin,xmax))
            if hadron=='h-': D=pd.DataFrame(data[3111]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:    dz,color=0.00,'r'
            elif ymin==0.15: dz,color=0.01,'darkorange'
            elif ymin==0.2:  dz,color=0.02,'g'
            elif ymin==0.3:  dz,color=0.03,'c'
            elif ymin==0.5:  dz,color=0.04,'m'
            else: color='k'

            p=ax.errorbar(d.Z+dz,d.value/d.thy,d.alpha/d.thy,fmt='.',color=color,zorder=1.1)

            #thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.axhline(1,0,1,color='black',alpha=0.9,ls='-')
        if cnt in [1,2,3,4,5,6,7,8]:
            ax.set_ylim(0.71,1.29)
            #ax.set_yticks([1,1.1,1.2,1.3,1.4])
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        #if cnt in [5,6,7,8]:
        #    ax.set_ylim(0.75,1.99)
        #    ax.set_yticks([0.8,1,1.2,1.4,1.6,1.8])
        #    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.set_xlim(0.15,0.85)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            if hadron=='h+': ax.set_ylabel(r'\boldmath$dM_p^{h^+}/dz_h + i$',size=30)
            if hadron=='h-': ax.set_ylabel(r'\boldmath$dM_p^{h^-}/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.03,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:ax2=ax
        if cnt==3:ax3=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax1.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.2,0.3]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax2.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    handles, labels = [],[]
    for _ in [0.5]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax3.legend(handles,labels,loc=(0.00,0.00),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis_ratio_COMPASS%d_proton_%s.png'%(wdir,year,hadron)
    py.savefig(filename)
    print('Saving SIDIS COMPASS%d ratio proton hadron figure to %s'%(year,filename))


#old?
def plot_pion_shift(wdir,hadron='pi+'):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if hadron=='pi+' and 1005 not in predictions['reactions']['sidis']: return 
    if hadron=='pi-' and 1006 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        shift = data[_]['shift-rep']
        data[_]['rshift']  = np.mean(shift,axis=0)
        data[_]['drshift'] = np.std (shift,axis=0)
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        del data[_]['r-residuals']
        del data[_]['n-residuals']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:  data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    thy = {}
    for xrange in X:
        xmin,xmax=xrange
        if hadron=='pi+': D=pd.DataFrame(data[1005]).query('X>%f and X<%f'%(xmin,xmax))
        if hadron=='pi-': D=pd.DataFrame(data[1006]).query('X>%f and X<%f'%(xmin,xmax))

        if len(D.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
            if len(d)<=1: continue
            #f=1#2**alpha
            f=alpha
            #print(ymin)
            if ymin==0.1:    dz,color=0.00,'r'
            elif ymin==0.15: dz,color=0.01,'darkorange'
            elif ymin==0.2:  dz,color=0.02,'g'
            elif ymin==0.3:  dz,color=0.03,'c'
            elif ymin==0.5:  dz,color=0.04,'m'
            else: color='k'

            mean, std = d.rshift, d.drshift
            value = d.value
            p=ax.errorbar(d.Z+dz,mean/value,std/value,fmt='.',color=color,zorder=1.1)

            #thy[ymin] = ax.fill_between(d.Z,(d.thy-d.dthy)+f, (d.thy+d.dthy)+f,color=color,zorder=1,alpha=0.5)
            #if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy+f,color=color,ls='--')

            H[ymin]=p

        ax.axhline(1,0,1,color='black',alpha=0.9,ls='-')
        #if cnt in [1,2,3,4]:
        #    ax.set_ylim(0.91,1.49)
        #    ax.set_yticks([1,1.1,1.2,1.3,1.4])
        #    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        #if cnt in [5,6,7,8]:
        #    ax.set_ylim(0.75,1.99)
        #    ax.set_yticks([0.8,1,1.2,1.4,1.6,1.8])
        #    ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        ax.set_xlim(0.15,0.85)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])
        ax.tick_params(axis='both', which='both', labelsize=25,direction='in',top=True,right=True)
        ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            ax.set_ylabel(r'\boldmath$dM/dz_h + i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        #if cnt==1 or cnt==5:
        #    ax.set_yticks([0.1,1,10])
        #    ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.03,0.93,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        #if cnt==2:
        #    from matplotlib.lines import Line2D
        #    h = [Line2D([0], [0], color='k', lw=5,ls='--'),
        #        Line2D([0], [0], color='k', lw=5,ls=':')]
        #    l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
        #    ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)

    k = 0
    handles, labels = [],[]
    for _ in [0.1,0.15,0.2,0.3,0.5]:
        if _ in H:
            handles.append(H[_])
            labels.append(r'$%s~<y<%s~(i=%d)$'%(Y[k][0],Y[k][1],k))
            k+=1
    ax1.legend(handles,labels,loc=(0.00,0.35),fontsize=20,frameon=0,handletextpad=0.5,handlelength=1.0)

    if hadron=='pi+': ax1.text(0.03,0.75,r'\boldmath$\pi^+$',transform=ax1.transAxes,size=50)
    if hadron=='pi-': ax1.text(0.03,0.75,r'\boldmath$\pi^-$',transform=ax1.transAxes,size=50)

    py.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.01, hspace=0.01)
    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/sidis-shift-%s.png'%(wdir,hadron)
    py.savefig(filename)
    print('Saving SIDIS pion figure to %s'%filename)




