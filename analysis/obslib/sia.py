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

    plot_pion  (wdir)
    plot_kaon  (wdir)
    plot_hadron(wdir)

    plot_pion_ratio  (wdir)
    plot_kaon_ratio  (wdir)
    plot_hadron_ratio(wdir)

    plot_pion_difference(wdir)


#--absolute plots
def plot_pion(wdir):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()

    if 'sia version' in conf and conf['sia version']==0: return

    idxs = []
    idxs.append(1001) # TASSO
    idxs.append(1002) # TASSO
    idxs.append(1003) # TASSO
    idxs.append(1004) # TASSO
    idxs.append(1005) # TASSO
    idxs.append(1006) # TASSO
    idxs.append(1007) # TPC
    idxs.append(1008) # TPC
    idxs.append(1013) # TOPAZ
    idxs.append(1014) # SLD
    idxs.append(1018) # ALEPH
    idxs.append(1019) # OPAL
    idxs.append(1025) # DELPHI
    idxs.append(1028) # BABAR
    idxs.append(1029) # BELL
    idxs.append(1030) # ARGUS
    idxs.append(1015) # SLD (uds)
    idxs.append(1026) # DELPHI (uds)
    idxs.append(1010) # TPC(c)
    idxs.append(1011) # TPC(b)
    idxs.append(1016) # SLD(c)
    idxs.append(1017) # SLD(b)
    idxs.append(1023) # OPAL(c)
    idxs.append(1024) # OPAL(b)
    idxs.append(1027) # DELPHI(b)
    idxs.append(1032) # BELLE 2020

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sia' not in predictions['reactions']: return
  
    flag = True 
    data = predictions['reactions']['sia']
    for idx in data:
        if idx in idxs: flag = False
        predictions = copy.copy(data[idx]['prediction-rep'])
        shifts      = copy.copy(data[idx]['shift-rep'])
        #predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        del data[idx]['r-residuals']
        del data[idx]['n-residuals']
        if 'rres-rep' in data[idx]: del data[idx]['rres-rep']
        data[idx]['thy'] = np.mean(predictions, axis = 0)
        data[idx]['dthy'] = np.std(predictions, axis = 0)

    if flag: return

    nrows,ncols=2,4
    N = nrows*ncols - 1
    fig = py.figure(figsize=(ncols*7,nrows*5))

    axs={}
    for i in range(N+1):
        axs[i] = py.subplot(nrows,ncols,i+1)

    axs[7].axis("off")

    hand = {}
    thy_plot, thy_band = {}, {}
    for idx in idxs:

        if idx not in data: continue

        obs = data[idx]['obs'][0]
        had = data[idx]['hadron'][0]
        rs  = data[idx]['RS'][0]
        Z      = data[idx]['z']
        units_val = 1/data[idx]['units-val'][0]
        jac       = 1/data[idx]['jac']

        idn = data[idx]['is normalized?'][0]

        if idn: sigtot = np.ones(len(Z))
        else:   sigtot = data[idx]['sigtot']

        values = data[idx]['value']*jac*units_val/sigtot
        alpha  = data[idx]['alpha']*jac*units_val/sigtot

        #--TASSO
        if   idx==1001: ax,color,fmt,ms = axs[0],'firebrick','*',8
        elif idx==1002: ax,color,fmt,ms = axs[0],'darkgreen','^',6
        elif idx==1003: ax,color,fmt,ms = axs[0],'blue'     ,'o',6
        elif idx==1004: ax,color,fmt,ms = axs[0],'purple'   ,'v',6
        elif idx==1005: ax,color,fmt,ms = axs[0],'black'    ,'s',5
        elif idx==1006: ax,color,fmt,ms = axs[0],'orange'   ,'D',5
        #--Belle, BABAR, ARGUS (low RS)
        elif idx==1028: ax,color,fmt,ms = axs[1],'firebrick','*',8
        elif idx==1029: ax,color,fmt,ms = axs[1],'darkgreen','^',6
        elif idx==1030: ax,color,fmt,ms = axs[1],'blue'     ,'o',6
        elif idx==1032: ax,color,fmt,ms = axs[1],'purple'   ,'v',6
        #--TPC, TOPAZ (medium energy)
        elif idx==1007: ax,color,fmt,ms = axs[2],'firebrick','*',8
        elif idx==1008: ax,color,fmt,ms = axs[2],'darkgreen','^',6
        elif idx==1013: ax,color,fmt,ms = axs[2],'blue'     ,'o',6
        #--SLD, ALEPH, OPAL, DELPHI (high energy)
        elif idx==1014: ax,color,fmt,ms = axs[3],'firebrick','*',8
        elif idx==1018: ax,color,fmt,ms = axs[3],'darkgreen','^',6
        elif idx==1019: ax,color,fmt,ms = axs[3],'blue'     ,'o',6
        elif idx==1025: ax,color,fmt,ms = axs[3],'purple'   ,'v',6
        #--tagged data
        #--uds
        elif idx==1015: ax,color,fmt,ms = axs[4],'darkgreen','^',6
        elif idx==1026: ax,color,fmt,ms = axs[4],'purple'   ,'v',6
        #--charm
        elif idx==1010: ax,color,fmt,ms = axs[5],'firebrick','*',8
        elif idx==1016: ax,color,fmt,ms = axs[5],'darkgreen','^',6
        elif idx==1023: ax,color,fmt,ms = axs[5],'blue'     ,'o',6
        #--bottom
        elif idx==1011: ax,color,fmt,ms = axs[6],'firebrick','*',8
        elif idx==1017: ax,color,fmt,ms = axs[6],'darkgreen','^',6
        elif idx==1024: ax,color,fmt,ms = axs[6],'blue'     ,'o',6
        elif idx==1027: ax,color,fmt,ms = axs[6],'purple'   ,'v',6
        else:
            print('Skipping index %s'%idx)
            continue

        hand[idx] = ax.errorbar(Z,values,yerr=alpha,color=color,fmt=fmt,ms=ms,capsize=3.0)

        thy = data[idx]['thy'] *jac*units_val/sigtot
        std = data[idx]['dthy']*jac*units_val/sigtot

        down = thy - std
        up   = thy + std
        #thy_plot[idx] ,= ax.plot(Z,thy,color=color)
        thy_band[idx]  = ax.fill_between(Z,down,up,color=color,alpha=0.7)
        if len(Z)==1:
            z = [Z[0]*0.95, Z[0]*1.05]
            down, up, thy = [down[0],down[0]],[up[0],up[0]],[thy[0],thy[0]]
            thy_band[idx] = ax.fill_between(z,down,up,color=color,alpha=0.7)


    for i in range(N):
        axs[i].tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=25)
        axs[i].xaxis.set_tick_params(which='major',length=8)
        axs[i].yaxis.set_tick_params(which='major',length=8)
        axs[i].xaxis.set_tick_params(which='minor',length=4)
        axs[i].yaxis.set_tick_params(which='minor',length=4)
        axs[i].semilogx()
        axs[i].semilogy()
        axs[i].set_xlim(1.5e-2,0.99)
        axs[i].set_xticks([0.02,0.1,0.5])
        axs[i].set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])
        #axs[i].xaxis.set_minor_locator(MultipleLocator(0.10))
        #axs[i].xaxis.set_major_locator(MultipleLocator(0.20))
        if i in [0,1,2]:     axs[i].tick_params(labelbottom=False)
        if i in [1,2,3,5,6]: axs[i].tick_params(labelleft=False)
        if i in [3,4,5,6]: 
            axs[i].set_xlabel(r'\boldmath$z$',size=40)
            axs[i].xaxis.set_label_coords(0.95,-0.01)

    ymax = 500
    ymin1 = 8e-3
    ymin2 = 5e-4
    axs[0].set_ylim(ymin1,ymax)
    axs[1].set_ylim(ymin1,ymax)
    axs[2].set_ylim(ymin1,ymax)
    axs[3].set_ylim(ymin1,ymax)
    axs[4].set_ylim(ymin2,ymax)
    axs[5].set_ylim(ymin2,ymax)
    axs[6].set_ylim(ymin2,ymax)
       
    axs[0].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}$',    size=50) 
    axs[4].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}~(q)$',size=50) 

    axs[0].text(0.60, 0.85, r'\textrm{\textbf{TASSO}}',transform=axs[0].transAxes,size=40)
    axs[1].text(0.50, 0.85, r'\boldmath$\sqrt{s} \approx 10~{\rm GeV}$',transform=axs[1].transAxes,size=30)
    #axs[2].text(0.40, 0.85, r'\boldmath$\sqrt{s} = 29,58 ~{\rm GeV}$'  ,transform=axs[2].transAxes,size=30)
    axs[2].text(0.50, 0.85, r'\boldmath$\sqrt{s} = 58 ~{\rm GeV}$'  ,transform=axs[2].transAxes,size=30)
    axs[3].text(0.60, 0.85, r'\boldmath$\sqrt{s} = M_Z$'               ,transform=axs[3].transAxes,size=30)
    axs[4].text(0.75, 0.85, r'\boldmath$(uds)$',transform=axs[4].transAxes,size=40)
    axs[5].text(0.85, 0.85, r'\boldmath$(c)$'  ,transform=axs[5].transAxes,size=40)
    axs[6].text(0.85, 0.85, r'\boldmath$(b)$'  ,transform=axs[6].transAxes,size=40)

    axs[7].text(0.05, 0.20, r'\textrm{\textbf{SIA \boldmath$\pi^{\pm}$}}',transform=axs[7].transAxes,size=60)

    fs = 25
    handles,labels = [],[]
    if 1001 in hand: handles.append((hand[1001],thy_band[1001]))
    if 1002 in hand: handles.append((hand[1002],thy_band[1002]))
    if 1003 in hand: handles.append((hand[1003],thy_band[1003]))
    if 1004 in hand: handles.append((hand[1004],thy_band[1004]))
    if 1005 in hand: handles.append((hand[1005],thy_band[1005]))
    if 1006 in hand: handles.append((hand[1006],thy_band[1006]))
    if 1001 in hand: labels.append(r'\textbf{\textrm{12 GeV}}')
    if 1002 in hand: labels.append(r'\textbf{\textrm{14 GeV}}')
    if 1003 in hand: labels.append(r'\textbf{\textrm{22 GeV}}')
    if 1004 in hand: labels.append(r'\textbf{\textrm{30 GeV}}')
    if 1005 in hand: labels.append(r'\textbf{\textrm{34 GeV}}')
    if 1006 in hand: labels.append(r'\textbf{\textrm{44 GeV}}')
    axs[0].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1028 in hand: handles.append((hand[1028],thy_band[1028]))
    if 1029 in hand: handles.append((hand[1029],thy_band[1029]))
    if 1030 in hand: handles.append((hand[1030],thy_band[1030]))
    if 1032 in hand: handles.append((hand[1032],thy_band[1032]))
    if 1028 in hand: labels.append(r'\textbf{\textrm{BABAR}}')
    if 1029 in hand: labels.append(r'\textbf{\textrm{BELLE 2013}}')
    if 1030 in hand: labels.append(r'\textbf{\textrm{ARGUS}}')
    if 1032 in hand: labels.append(r'\textbf{\textrm{BELLE 2020}}')
    axs[1].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    py.tight_layout()

    handles,labels = [],[]
    if 1007 in hand: handles.append((hand[1007],thy_band[1007]))
    if 1008 in hand: handles.append((hand[1008],thy_band[1008]))
    if 1013 in hand: handles.append((hand[1013],thy_band[1013]))
    if 1007 in hand: labels.append(r'\textbf{\textrm{TPC (1984)}}')
    if 1008 in hand: labels.append(r'\textbf{\textrm{TPC (1988)}}')
    if 1013 in hand: labels.append(r'\textbf{\textrm{TOPAZ}}')
    axs[2].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1014 in hand: handles.append((hand[1014],thy_band[1014]))
    if 1018 in hand: handles.append((hand[1018],thy_band[1018]))
    if 1019 in hand: handles.append((hand[1019],thy_band[1019]))
    if 1025 in hand: handles.append((hand[1025],thy_band[1025]))
    if 1014 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1018 in hand: labels.append(r'\textbf{\textrm{ALEPH}}')
    if 1019 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 1025 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[3].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1015 in hand: handles.append((hand[1015],thy_band[1015]))
    if 1026 in hand: handles.append((hand[1026],thy_band[1026]))
    if 1015 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1026 in hand: labels.append(r'\textbf{\textrm{DELHPI}}')
    axs[4].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    
    handles,labels = [],[]
    if 1010 in hand: handles.append((hand[1010],thy_band[1010]))
    if 1016 in hand: handles.append((hand[1016],thy_band[1016]))
    if 1023 in hand: handles.append((hand[1023],thy_band[1023]))
    if 1010 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 1016 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1023 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    axs[5].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1011 in hand: handles.append((hand[1011],thy_band[1011]))
    if 1017 in hand: handles.append((hand[1017],thy_band[1017]))
    if 1024 in hand: handles.append((hand[1024],thy_band[1024]))
    if 1027 in hand: handles.append((hand[1027],thy_band[1027]))
    if 1011 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 1017 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1024 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 1027 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[6].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)


    py.tight_layout()

    py.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
    filename = '%s/gallery/sia-pion.png'%wdir
    py.savefig(filename)
    print('Saving SIA pion figure to %s'%filename)

def plot_kaon(wdir):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()

    if 'sia version' in conf and conf['sia version']==0: return

    idxs = []
    idxs.append(2001) # TASSO
    idxs.append(2002) # TASSO
    idxs.append(2003) # TASSO
    idxs.append(2004) # TASSO
    idxs.append(2005) # TASSO
    idxs.append(2006) # TASSO
    idxs.append(2007) # TPC
    idxs.append(2008) # TPC
    idxs.append(2013) # TOPAZ
    idxs.append(2014) # SLD
    idxs.append(2018) # ALEPH
    idxs.append(2019) # OPAL
    idxs.append(2025) # DELPHI
    idxs.append(2028) # BABAR
    idxs.append(2029) # BELL
    idxs.append(2030) # ARGUS
    idxs.append(2015) # SLD (uds)
    idxs.append(2026) # DELPHI (uds)
    idxs.append(2010) # TPC(c)
    idxs.append(2011) # TPC(b)
    idxs.append(2016) # SLD(c)
    idxs.append(2017) # SLD(b)
    idxs.append(2023) # OPAL(c)
    idxs.append(2024) # OPAL(b)
    idxs.append(2027) # DELPHI(b)
    idxs.append(2032) # BELLE 2020

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sia' not in predictions['reactions']: return
  
    flag = True 
    data = predictions['reactions']['sia']
    for idx in data:
        if idx in idxs: flag = False
        predictions = copy.copy(data[idx]['prediction-rep'])
        shifts      = copy.copy(data[idx]['shift-rep'])
        #predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        del data[idx]['r-residuals']
        del data[idx]['n-residuals']
        if 'rres-rep' in data[idx]: del data[idx]['rres-rep']
        data[idx]['thy'] = np.mean(predictions, axis = 0)
        data[idx]['dthy'] = np.std(predictions, axis = 0)

    if flag: return

    nrows,ncols=2,4
    N = nrows*ncols - 1
    fig = py.figure(figsize=(ncols*7,nrows*5))

    axs={}
    for i in range(N+1):
        axs[i] = py.subplot(nrows,ncols,i+1)

    axs[7].axis("off")

    hand = {}
    thy_plot, thy_band = {}, {}
    for idx in idxs:

        if idx not in data: continue

        obs = data[idx]['obs'][0]
        had = data[idx]['hadron'][0]
        rs  = data[idx]['RS'][0]
        Z      = data[idx]['z']
        units_val = 1/data[idx]['units-val'][0]
        jac       = 1/data[idx]['jac']

        idn = data[idx]['is normalized?'][0]

        if idn: sigtot = np.ones(len(Z))
        else:   sigtot = data[idx]['sigtot']

        values = data[idx]['value']*jac*units_val/sigtot
        alpha  = data[idx]['alpha']*jac*units_val/sigtot

        #--TASSO
        if   idx==2001: ax,color,fmt,ms = axs[0],'firebrick','*',8
        elif idx==2002: ax,color,fmt,ms = axs[0],'darkgreen','^',6
        elif idx==2003: ax,color,fmt,ms = axs[0],'blue'     ,'o',6
        elif idx==2004: ax,color,fmt,ms = axs[0],'purple'   ,'v',6
        elif idx==2005: ax,color,fmt,ms = axs[0],'black'    ,'s',5
        elif idx==2006: ax,color,fmt,ms = axs[0],'orange'   ,'D',5
        #--Belle, BABAR, ARGUS (low RS)
        elif idx==2028: ax,color,fmt,ms = axs[1],'firebrick','*',8
        elif idx==2029: ax,color,fmt,ms = axs[1],'darkgreen','^',6
        elif idx==2030: ax,color,fmt,ms = axs[1],'blue'     ,'o',6
        elif idx==2032: ax,color,fmt,ms = axs[1],'purple'   ,'v',6
        #--TPC, TOPAZ (medium energy)
        elif idx==2007: ax,color,fmt,ms = axs[2],'firebrick','*',8
        elif idx==2008: ax,color,fmt,ms = axs[2],'darkgreen','^',6
        elif idx==2013: ax,color,fmt,ms = axs[2],'blue'     ,'o',6
        #--SLD, ALEPH, OPAL, DELPHI (high energy)
        elif idx==2014: ax,color,fmt,ms = axs[3],'firebrick','*',8
        elif idx==2018: ax,color,fmt,ms = axs[3],'darkgreen','^',6
        elif idx==2019: ax,color,fmt,ms = axs[3],'blue'     ,'o',6
        elif idx==2025: ax,color,fmt,ms = axs[3],'purple'   ,'v',6
        #--tagged data
        #--uds
        elif idx==2015: ax,color,fmt,ms = axs[4],'darkgreen','^',6
        elif idx==2026: ax,color,fmt,ms = axs[4],'purple'   ,'v',6
        #--charm
        elif idx==2010: ax,color,fmt,ms = axs[5],'firebrick','*',8
        elif idx==2016: ax,color,fmt,ms = axs[5],'darkgreen','^',6
        elif idx==2023: ax,color,fmt,ms = axs[5],'blue'     ,'o',6
        #--bottom
        elif idx==2011: ax,color,fmt,ms = axs[6],'firebrick','*',8
        elif idx==2017: ax,color,fmt,ms = axs[6],'darkgreen','^',6
        elif idx==2024: ax,color,fmt,ms = axs[6],'blue'     ,'o',6
        elif idx==2027: ax,color,fmt,ms = axs[6],'purple'   ,'v',6
        else:
            print('Skipping index %s'%idx)
            continue

        hand[idx] = ax.errorbar(Z,values,yerr=alpha,color=color,fmt=fmt,ms=ms,capsize=3.0)

        thy = data[idx]['thy'] *jac*units_val/sigtot
        std = data[idx]['dthy']*jac*units_val/sigtot

        down = thy - std
        up   = thy + std
        #thy_plot[idx] ,= ax.plot(Z,thy,color=color)
        thy_band[idx]  = ax.fill_between(Z,down,up,color=color,alpha=0.7)
        if len(Z)==1:
            z = [Z[0]*0.95, Z[0]*1.05]
            down, up, thy = [down[0],down[0]],[up[0],up[0]],[thy[0],thy[0]]
            thy_band[idx] = ax.fill_between(z,down,up,color=color,alpha=0.7)


    for i in range(N):
        axs[i].tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=25)
        axs[i].xaxis.set_tick_params(which='major',length=8)
        axs[i].yaxis.set_tick_params(which='major',length=8)
        axs[i].xaxis.set_tick_params(which='minor',length=4)
        axs[i].yaxis.set_tick_params(which='minor',length=4)
        axs[i].semilogx()
        axs[i].semilogy()
        axs[i].set_xlim(1.5e-2,0.99)
        axs[i].set_xticks([0.02,0.1,0.5])
        axs[i].set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])
        #axs[i].xaxis.set_minor_locator(MultipleLocator(0.10))
        #axs[i].xaxis.set_major_locator(MultipleLocator(0.20))
        if i in [0,1,2]:     axs[i].tick_params(labelbottom=False)
        if i in [1,2,3,5,6]: axs[i].tick_params(labelleft=False)
        if i in [3,4,5,6]: 
            axs[i].set_xlabel(r'\boldmath$z$',size=40)
            axs[i].xaxis.set_label_coords(0.95,-0.01)

    ymax = 60
    ymin1 = 8e-3
    ymin2 = 5e-4
    axs[0].set_ylim(ymin1,ymax)
    axs[1].set_ylim(ymin1,ymax)
    axs[2].set_ylim(ymin1,ymax)
    axs[3].set_ylim(ymin1,ymax)
    axs[4].set_ylim(ymin2,ymax)
    axs[5].set_ylim(ymin2,ymax)
    axs[6].set_ylim(ymin2,ymax)
       
    axs[0].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}$',    size=50) 
    axs[4].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}~(q)$',size=50) 

    axs[0].text(0.60, 0.85, r'\textrm{\textbf{TASSO}}',transform=axs[0].transAxes,size=40)
    axs[1].text(0.50, 0.85, r'\boldmath$\sqrt{s} \approx 10~{\rm GeV}$',transform=axs[1].transAxes,size=30)
    axs[2].text(0.40, 0.85, r'\boldmath$\sqrt{s} = 29,58 ~{\rm GeV}$'  ,transform=axs[2].transAxes,size=30)
    axs[3].text(0.60, 0.85, r'\boldmath$\sqrt{s} = M_Z$'               ,transform=axs[3].transAxes,size=30)
    axs[4].text(0.75, 0.85, r'\boldmath$(uds)$',transform=axs[4].transAxes,size=40)
    axs[5].text(0.85, 0.85, r'\boldmath$(c)$'  ,transform=axs[5].transAxes,size=40)
    axs[6].text(0.85, 0.85, r'\boldmath$(b)$'  ,transform=axs[6].transAxes,size=40)

    axs[7].text(0.05, 0.20, r'\textrm{\textbf{SIA \boldmath$K^{\pm}$}}',transform=axs[7].transAxes,size=60)

    fs = 25
    handles,labels = [],[]
    if 2001 in hand: handles.append((hand[2001],thy_band[2001]))
    if 2002 in hand: handles.append((hand[2002],thy_band[2002]))
    if 2003 in hand: handles.append((hand[2003],thy_band[2003]))
    if 2004 in hand: handles.append((hand[2004],thy_band[2004]))
    if 2005 in hand: handles.append((hand[2005],thy_band[2005]))
    if 2006 in hand: handles.append((hand[2006],thy_band[2006]))
    if 2001 in hand: labels.append(r'\textbf{\textrm{12 GeV}}')
    if 2002 in hand: labels.append(r'\textbf{\textrm{14 GeV}}')
    if 2003 in hand: labels.append(r'\textbf{\textrm{22 GeV}}')
    if 2004 in hand: labels.append(r'\textbf{\textrm{30 GeV}}')
    if 2005 in hand: labels.append(r'\textbf{\textrm{34 GeV}}')
    if 2006 in hand: labels.append(r'\textbf{\textrm{44 GeV}}')
    axs[0].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 2028 in hand: handles.append((hand[2028],thy_band[2028]))
    if 2029 in hand: handles.append((hand[2029],thy_band[2029]))
    if 2030 in hand: handles.append((hand[2030],thy_band[2030]))
    if 2032 in hand: handles.append((hand[2032],thy_band[2032]))
    if 2028 in hand: labels.append(r'\textbf{\textrm{BABAR}}')
    if 2029 in hand: labels.append(r'\textbf{\textrm{BELLE 2013}}')
    if 2030 in hand: labels.append(r'\textbf{\textrm{ARGUS}}')
    if 2032 in hand: labels.append(r'\textbf{\textrm{BELLE 2020}}')
    axs[1].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    py.tight_layout()

    handles,labels = [],[]
    if 2007 in hand: handles.append((hand[2007],thy_band[2007]))
    if 2008 in hand: handles.append((hand[2008],thy_band[2008]))
    if 2013 in hand: handles.append((hand[2013],thy_band[2013]))
    if 2007 in hand: labels.append(r'\textbf{\textrm{TPC (1984)}}')
    if 2008 in hand: labels.append(r'\textbf{\textrm{TPC (1988)}}')
    if 2013 in hand: labels.append(r'\textbf{\textrm{TOPAZ}}')
    axs[2].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 2014 in hand: handles.append((hand[2014],thy_band[2014]))
    if 2018 in hand: handles.append((hand[2018],thy_band[2018]))
    if 2019 in hand: handles.append((hand[2019],thy_band[2019]))
    if 2025 in hand: handles.append((hand[2025],thy_band[2025]))
    if 2014 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 2018 in hand: labels.append(r'\textbf{\textrm{ALEPH}}')
    if 2019 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 2025 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[3].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 2015 in hand: handles.append((hand[2015],thy_band[2015]))
    if 2026 in hand: handles.append((hand[2026],thy_band[2026]))
    if 2015 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 2026 in hand: labels.append(r'\textbf{\textrm{DELHPI}}')
    axs[4].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    
    handles,labels = [],[]
    if 2010 in hand: handles.append((hand[2010],thy_band[2010]))
    if 2016 in hand: handles.append((hand[2016],thy_band[2016]))
    if 2023 in hand: handles.append((hand[2023],thy_band[2023]))
    if 2010 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 2016 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 2023 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    axs[5].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 2011 in hand: handles.append((hand[2011],thy_band[2011]))
    if 2017 in hand: handles.append((hand[2017],thy_band[2017]))
    if 2024 in hand: handles.append((hand[2024],thy_band[2024]))
    if 2027 in hand: handles.append((hand[2027],thy_band[2027]))
    if 2011 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 2017 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 2024 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 2027 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[6].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)


    py.tight_layout()

    py.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
    filename = '%s/gallery/sia-kaon.png'%wdir
    py.savefig(filename)
    print('Saving SIA kaon figure to %s'%filename)

def plot_hadron(wdir):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if 'sia version' in conf and conf['sia version']==0: return

    idxs = []
    idxs.append(4003) # TASSO
    idxs.append(4008) # TASSO
    idxs.append(4009) # TASSO
    idxs.append(4010) # TASSO
    idxs.append(4011) # TASSO
    idxs.append(4012) # TASSO
    idxs.append(4004) # TPC
    idxs.append(4002) # SLD
    idxs.append(4000) # ALEPH
    idxs.append(4007) # OPAL
    idxs.append(4001) # DELPHI
    idxs.append(4014) # SLD(c)
    idxs.append(4015) # SLD(b)
    idxs.append(4005) # OPAL(c)
    idxs.append(4006) # OPAL(b)
    idxs.append(4013) # DELPHI(b)
    idxs.append(4016) # SLD(uds)
    idxs.append(4017) # DELPHI(uds)

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sia' not in predictions['reactions']: return
  
    flag = True 
    data = predictions['reactions']['sia']
    for idx in data:
        if idx in idxs: flag = False
        predictions = copy.copy(data[idx]['prediction-rep'])
        shifts      = copy.copy(data[idx]['shift-rep'])
        #predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        del data[idx]['r-residuals']
        del data[idx]['n-residuals']
        if 'rres-rep' in data[idx]: del data[idx]['rres-rep']
        data[idx]['thy'] = np.mean(predictions, axis = 0)
        data[idx]['dthy'] = np.std(predictions, axis = 0)

    if flag: return

    nrows,ncols=2,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*7,nrows*5))

    axs={}
    for i in range(N):
        axs[i] = py.subplot(nrows,ncols,i+1)

    hand = {}
    thy_plot, thy_band = {}, {}
    for idx in idxs:

        if idx not in data: continue

        obs = data[idx]['obs'][0]
        had = data[idx]['hadron'][0]
        rs  = data[idx]['RS'][0]
        Z      = data[idx]['z']
        units_val = 1/data[idx]['units-val'][0]
        jac       = 1/data[idx]['jac']

        idn = data[idx]['is normalized?'][0]

        if idn: sigtot = np.ones(len(Z))
        else:   sigtot = data[idx]['sigtot']

        values = data[idx]['value']*jac*units_val/sigtot
        alpha  = data[idx]['alpha']*jac*units_val/sigtot

        #--TASSO
        if   idx==4003: ax,color,fmt,ms = axs[0],'firebrick','*',8
        elif idx==4008: ax,color,fmt,ms = axs[0],'darkgreen','^',6
        elif idx==4009: ax,color,fmt,ms = axs[0],'blue'     ,'o',6
        elif idx==4010: ax,color,fmt,ms = axs[0],'purple'   ,'v',6
        elif idx==4011: ax,color,fmt,ms = axs[0],'black'    ,'s',5
        elif idx==4012: ax,color,fmt,ms = axs[0],'orange'   ,'D',5
        #--TPC (medium energy)
        elif idx==4004: ax,color,fmt,ms = axs[1],'darkgreen','^',6
        #--SLD, ALEPH, OPAL, DELPHI (high energy)
        elif idx==4002: ax,color,fmt,ms = axs[2],'firebrick','*',8
        elif idx==4000: ax,color,fmt,ms = axs[2],'darkgreen','^',6
        elif idx==4007: ax,color,fmt,ms = axs[2],'blue'     ,'o',6
        elif idx==4001: ax,color,fmt,ms = axs[2],'purple'   ,'v',6
        #--tagged data
        #--uds
        elif idx==4016: ax,color,fmt,ms = axs[3],'darkgreen','^',6
        elif idx==4017: ax,color,fmt,ms = axs[3],'purple'   ,'v',6
        #--charm
        elif idx==4014: ax,color,fmt,ms = axs[4],'darkgreen','^',6
        elif idx==4006: ax,color,fmt,ms = axs[4],'blue'     ,'o',6
        #--bottom
        elif idx==4015: ax,color,fmt,ms = axs[5],'darkgreen','^',6
        elif idx==4005: ax,color,fmt,ms = axs[5],'blue'     ,'o',6
        elif idx==4013: ax,color,fmt,ms = axs[5],'purple'   ,'v',6
        else:
            print('Skipping index %s'%idx)
            continue

        hand[idx] = ax.errorbar(Z,values,yerr=alpha,color=color,fmt=fmt,ms=ms,capsize=3.0)

        thy = data[idx]['thy'] *jac*units_val/sigtot
        std = data[idx]['dthy']*jac*units_val/sigtot

        down = thy - std
        up   = thy + std
        #thy_plot[idx] ,= ax.plot(Z,thy,color=color)
        thy_band[idx]  = ax.fill_between(Z,down,up,color=color,alpha=0.7)
        if len(Z)==1:
            z = [Z[0]*0.95, Z[0]*1.05]
            down, up, thy = [down[0],down[0]],[up[0],up[0]],[thy[0],thy[0]]
            thy_band[idx] = ax.fill_between(z,down,up,color=color,alpha=0.7)


    for i in range(N):
        axs[i].tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        axs[i].xaxis.set_tick_params(which='major',length=8)
        axs[i].yaxis.set_tick_params(which='major',length=8)
        axs[i].xaxis.set_tick_params(which='minor',length=4)
        axs[i].yaxis.set_tick_params(which='minor',length=4)
        axs[i].semilogx()
        axs[i].semilogy()
        axs[i].set_xlim(1.5e-2,0.99)
        axs[i].set_xticks([0.02,0.1,0.5])
        axs[i].set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])
        if i in [0,1,3]: axs[i].tick_params(labelbottom=False)
        if i in [1,2,4,5]: axs[i].tick_params(labelleft=False)
        if i in [3,4,5]: 
            axs[i].set_xlabel(r'\boldmath$z$',size=40)
            axs[i].xaxis.set_label_coords(0.95,-0.01)

    ymax = 500
    ymin1 = 8e-3
    ymin2 = 2e-3
    axs[0].set_ylim(ymin1,ymax)
    axs[1].set_ylim(ymin1,ymax)
    axs[2].set_ylim(ymin1,ymax)
    axs[3].set_ylim(ymin2,ymax)
    axs[4].set_ylim(ymin2,ymax)
       
    axs[0].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}$',    size=50) 
    axs[3].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}~(q)$',size=50) 

    axs[0].text(0.60, 0.85, r'\textrm{\textbf{TASSO}}',transform=axs[0].transAxes,size=40)
    #axs[1].text(0.40, 0.85, r'\boldmath$\sqrt{s} = 29,58 ~{\rm GeV}$'  ,transform=axs[1].transAxes,size=30)
    axs[1].text(0.50, 0.85, r'\boldmath$\sqrt{s} = 29 ~{\rm GeV}$'  ,transform=axs[1].transAxes,size=30)
    axs[2].text(0.60, 0.85, r'\boldmath$\sqrt{s} = M_Z$'               ,transform=axs[2].transAxes,size=30)
    axs[3].text(0.75, 0.85, r'\boldmath$(uds)$',transform=axs[3].transAxes,size=40)
    axs[4].text(0.85, 0.85, r'\boldmath$(c)$',transform=axs[4].transAxes,size=40)
    axs[5].text(0.85, 0.85, r'\boldmath$(b)$',transform=axs[5].transAxes,size=40)

    axs[5].text(0.05, 0.25, r'\textrm{\textbf{SIA \boldmath$h^{\pm}$}}',transform=axs[5].transAxes,size=60)

    fs = 25
    handles,labels = [],[]
    if 4003 in hand: handles.append((hand[4003],thy_band[4003]))
    if 4008 in hand: handles.append((hand[4008],thy_band[4008]))
    if 4009 in hand: handles.append((hand[4009],thy_band[4009]))
    if 4010 in hand: handles.append((hand[4010],thy_band[4010]))
    if 4011 in hand: handles.append((hand[4011],thy_band[4011]))
    if 4012 in hand: handles.append((hand[4012],thy_band[4012]))
    if 4003 in hand: labels.append(r'\textbf{\textrm{12 GeV}}')
    if 4008 in hand: labels.append(r'\textbf{\textrm{14 GeV}}')
    if 4009 in hand: labels.append(r'\textbf{\textrm{22 GeV}}')
    if 4010 in hand: labels.append(r'\textbf{\textrm{30 GeV}}')
    if 4011 in hand: labels.append(r'\textbf{\textrm{35 GeV}}')
    if 4012 in hand: labels.append(r'\textbf{\textrm{44 GeV}}')
    axs[0].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 4004 in hand: handles.append((hand[4004],thy_band[4004]))
    if 4004 in hand: labels.append(r'\textbf{\textrm{TPC (1988)}}')
    axs[1].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 4002 in hand: handles.append((hand[4002],thy_band[4002]))
    if 4000 in hand: handles.append((hand[4000],thy_band[4000]))
    if 4007 in hand: handles.append((hand[4007],thy_band[4007]))
    if 4001 in hand: handles.append((hand[4001],thy_band[4001]))
    if 4002 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 4000 in hand: labels.append(r'\textbf{\textrm{ALEPH}}')
    if 4007 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 4001 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[2].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
   
    handles,labels = [],[]
    if 4016 in hand: handles.append((hand[4016],thy_band[4016]))
    if 4017 in hand: handles.append((hand[4017],thy_band[4017]))
    if 4016 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 4017 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[3].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
 
    handles,labels = [],[]
    if 4014 in hand: handles.append((hand[4014],thy_band[4014]))
    if 4006 in hand: handles.append((hand[4006],thy_band[4006]))
    if 4014 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 4006 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    axs[4].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 4015 in hand: handles.append((hand[4015],thy_band[4015]))
    if 4005 in hand: handles.append((hand[4005],thy_band[4005]))
    if 4013 in hand: handles.append((hand[4013],thy_band[4013]))
    if 4015 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 4005 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 4013 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[5].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)


    py.tight_layout()

    py.subplots_adjust(left=0.10, bottom=0.05, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
    filename = '%s/gallery/sia-hadron.png'%wdir
    py.savefig(filename)
    print('Saving SIA hadron figure to %s'%filename)


#--ratio plots
def plot_pion_ratio(wdir):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if 'sia version' in conf and conf['sia version']==0: return

    idxs = []
    idxs.append(1001) # TASSO
    idxs.append(1002) # TASSO
    idxs.append(1003) # TASSO
    idxs.append(1004) # TASSO
    idxs.append(1005) # TASSO
    idxs.append(1006) # TASSO
    idxs.append(1007) # TPC
    idxs.append(1008) # TPC
    idxs.append(1013) # TOPAZ
    idxs.append(1014) # SLD
    idxs.append(1018) # ALEPH
    idxs.append(1019) # OPAL
    idxs.append(1025) # DELPHI
    idxs.append(1028) # BABAR
    idxs.append(1029) # BELL
    idxs.append(1030) # ARGUS
    idxs.append(1015) # SLD (uds)
    idxs.append(1026) # DELPHI (uds)
    idxs.append(1010) # TPC(c)
    idxs.append(1011) # TPC(b)
    idxs.append(1016) # SLD(c)
    idxs.append(1017) # SLD(b)
    idxs.append(1023) # OPAL(c)
    idxs.append(1024) # OPAL(b)
    idxs.append(1027) # DELPHI(b)

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sia' not in predictions['reactions']: return
  
    flag = True 
    data = predictions['reactions']['sia']
    for idx in data:
        if idx in idxs: flag = False
        predictions = copy.copy(data[idx]['prediction-rep'])
        shifts      = copy.copy(data[idx]['shift-rep'])
        #predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        del data[idx]['r-residuals']
        del data[idx]['n-residuals']
        if 'rres-rep' in data[idx]: del data[idx]['rres-rep']
        data[idx]['thy'] = np.mean(predictions, axis = 0)
        data[idx]['dthy'] = np.std(predictions, axis = 0)

    if flag: return

    nrows,ncols=2,4
    N = nrows*ncols - 1
    fig = py.figure(figsize=(ncols*7,nrows*5))

    axs={}
    for i in range(N+1):
        axs[i] = py.subplot(nrows,ncols,i+1)

    axs[7].axis("off")

    hand = {}
    thy_plot, thy_band = {}, {}
    for idx in idxs:

        if idx not in data: continue

        obs = data[idx]['obs'][0]
        had = data[idx]['hadron'][0]
        rs  = data[idx]['RS'][0]
        Z      = data[idx]['z']
        xp     = data[idx]['xp']
        p      = data[idx]['p']
        units_val = 1/data[idx]['units-val'][0]
        jac       = 1/data[idx]['jac']

        idn = data[idx]['is normalized?'][0]

        if idn: sigtot = np.ones(len(Z))
        else:   sigtot = data[idx]['sigtot']

        values = data[idx]['value']*jac*units_val/sigtot
        alpha  = data[idx]['alpha']*jac*units_val/sigtot

        #--TASSO
        if   idx==1001: ax,color,fmt,ms,zorder = axs[0],'firebrick','*',8,1
        elif idx==1002: ax,color,fmt,ms,zorder = axs[0],'darkgreen','^',6,1
        elif idx==1003: ax,color,fmt,ms,zorder = axs[0],'blue'     ,'o',6,1
        elif idx==1004: ax,color,fmt,ms,zorder = axs[0],'purple'   ,'v',6,1
        elif idx==1005: ax,color,fmt,ms,zorder = axs[0],'black'    ,'s',5,1
        elif idx==1006: ax,color,fmt,ms,zorder = axs[0],'orange'   ,'D',5,1
        #--Belle, BABAR, ARGUS (low RS),zorder
        elif idx==1028: ax,color,fmt,ms,zorder = axs[1],'firebrick','*',8,2
        elif idx==1029: ax,color,fmt,ms,zorder = axs[1],'darkgreen','^',6,1
        elif idx==1030: ax,color,fmt,ms,zorder = axs[1],'blue'     ,'o',6,1
        #--TPC, TOPAZ (medium energy)
        elif idx==1007: ax,color,fmt,ms,zorder = axs[2],'firebrick','*',8,1
        elif idx==1008: ax,color,fmt,ms,zorder = axs[2],'darkgreen','^',6,1
        elif idx==1013: ax,color,fmt,ms,zorder = axs[2],'blue'     ,'o',6,1
        #--SLD, ALEPH, OPAL, DELPHI (hirgh energy)
        elif idx==1014: ax,color,fmt,ms,zorder = axs[3],'firebrick','*',8,1
        elif idx==1018: ax,color,fmt,ms,zorder = axs[3],'darkgreen','^',6,1
        elif idx==1019: ax,color,fmt,ms,zorder = axs[3],'blue'     ,'o',6,1
        elif idx==1025: ax,color,fmt,ms,zorder = axs[3],'purple'   ,'v',6,1
        #--tagged data
        #--uds
        elif idx==1015: ax,color,fmt,ms,zorder = axs[4],'darkgreen','^',6,1
        elif idx==1026: ax,color,fmt,ms,zorder = axs[4],'purple'   ,'v',6,1
        #--charm
        elif idx==1010: ax,color,fmt,ms,zorder = axs[5],'firebrick','*',8,1
        elif idx==1016: ax,color,fmt,ms,zorder = axs[5],'darkgreen','^',6,1
        elif idx==1023: ax,color,fmt,ms,zorder = axs[5],'blue'     ,'o',6,1
        #--bottom
        elif idx==1011: ax,color,fmt,ms,zorder = axs[6],'firebrick','*',8,1
        elif idx==1017: ax,color,fmt,ms,zorder = axs[6],'darkgreen','^',6,1
        elif idx==1024: ax,color,fmt,ms,zorder = axs[6],'blue'     ,'o',6,1
        elif idx==1027: ax,color,fmt,ms,zorder = axs[6],'purple'   ,'v',6,1
        else:
            print('Skipping index %s'%idx)
            continue

        thy = data[idx]['thy'] *jac*units_val/sigtot
        std = data[idx]['dthy']*jac*units_val/sigtot
        hand[idx] = ax.errorbar(Z,values/thy,yerr=alpha/np.abs(thy),color=color,fmt=fmt,ms=ms,capsize=3.0,zorder=zorder)

        down = (thy - std)/thy
        up   = (thy + std)/thy
        thy_band[idx]  = ax.fill_between(Z,down,up,color=color,alpha=0.7,zorder=zorder)
        if len(Z)==1:
            z = [Z[0]*0.95, Z[0]*1.05]
            down, up, thy = [down[0],down[0]],[up[0],up[0]],[thy[0],thy[0]]
            thy_band[idx] = ax.fill_between(z,down,up,color=color,alpha=0.7,zorder=zorder)


    for i in range(N):
        axs[i].axhline(1,0,1,color='black',ls='-',alpha=0.8,zorder=10)
        axs[i].tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        axs[i].xaxis.set_tick_params(which='major',length=8)
        axs[i].yaxis.set_tick_params(which='major',length=8)
        axs[i].xaxis.set_tick_params(which='minor',length=4)
        axs[i].yaxis.set_tick_params(which='minor',length=4)
        axs[i].semilogx()
        axs[i].set_xlim(1.5e-2,0.99)
        axs[i].set_xticks([0.02,0.1,0.5])
        axs[i].set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])
        #axs[i].xaxis.set_minor_locator(MultipleLocator(0.10))
        #axs[i].xaxis.set_major_locator(MultipleLocator(0.20))
        if i in [0,1,2]:     axs[i].tick_params(labelbottom=False)
        #if i in [1,2,3,5,6]: axs[i].tick_params(labelleft=False)
        if i in [2,3,4,5,6]: 
            axs[i].set_xlabel(r'\boldmath$z$',size=40)
            axs[i].xaxis.set_label_coords(0.95,-0.01)

    axs[0].set_ylim(0.20,1.80)
    axs[0].set_yticks([0.5,1,1.5])
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[1].set_ylim(0.91,1.09)
    axs[1].set_yticks([0.95,1,1.05])
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.01))
    
    axs[2].set_ylim(0.70,1.30)
    axs[2].set_yticks([0.8,1,1.2])
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[3].set_ylim(0.50,1.50)
    axs[3].set_yticks([0.6,1,1.4])
    axs[3].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[4].set_ylim(0.50,1.50)
    axs[4].set_yticks([0.6,1,1.4])
    axs[4].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[5].set_ylim(0.20,1.80)
    axs[5].set_yticks([0.5,1,1.5])
    axs[5].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[6].set_ylim(0.20,1.80)
    axs[6].set_yticks([0.5,1,1.5])
    axs[6].yaxis.set_minor_locator(MultipleLocator(0.10))


   
    axs[0].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}$',    size=50) 
    axs[4].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}~(q)$',size=50) 

    axs[0].text(0.05, 0.85, r'\textrm{\textbf{TASSO}}',transform=axs[0].transAxes,size=40)
    axs[1].text(0.05, 0.85, r'\boldmath$\sqrt{s} \approx 10~{\rm GeV}$',transform=axs[1].transAxes,size=30)
    #axs[2].text(0.05, 0.85, r'\boldmath$\sqrt{s} = 29,58 ~{\rm GeV}$'  ,transform=axs[2].transAxes,size=30)
    axs[2].text(0.05, 0.85, r'\boldmath$\sqrt{s} = 58 ~{\rm GeV}$'  ,transform=axs[2].transAxes,size=30)
    axs[3].text(0.05, 0.85, r'\boldmath$\sqrt{s} = M_Z$'               ,transform=axs[3].transAxes,size=30)
    axs[4].text(0.05, 0.85, r'\boldmath$(uds)$',transform=axs[4].transAxes,size=40)
    axs[5].text(0.05, 0.85, r'\boldmath$(c)$'  ,transform=axs[5].transAxes,size=40)
    axs[6].text(0.05, 0.85, r'\boldmath$(b)$'  ,transform=axs[6].transAxes,size=40)

    axs[7].text(0.05, 0.20, r'\textrm{\textbf{SIA \boldmath$\pi^{\pm}$}}',transform=axs[7].transAxes,size=60)

    fs = 25
    handles,labels = [],[]
    if 1001 in hand: handles.append((hand[1001],thy_band[1001]))
    if 1002 in hand: handles.append((hand[1002],thy_band[1002]))
    if 1003 in hand: handles.append((hand[1003],thy_band[1003]))
    if 1004 in hand: handles.append((hand[1004],thy_band[1004]))
    if 1005 in hand: handles.append((hand[1005],thy_band[1005]))
    if 1006 in hand: handles.append((hand[1006],thy_band[1006]))
    if 1001 in hand: labels.append(r'\textbf{\textrm{12 GeV}}')
    if 1002 in hand: labels.append(r'\textbf{\textrm{14 GeV}}')
    if 1003 in hand: labels.append(r'\textbf{\textrm{22 GeV}}')
    if 1004 in hand: labels.append(r'\textbf{\textrm{30 GeV}}')
    if 1005 in hand: labels.append(r'\textbf{\textrm{34 GeV}}')
    if 1006 in hand: labels.append(r'\textbf{\textrm{44 GeV}}')
    axs[0].legend(handles,labels,frameon=False,fontsize=20,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1028 in hand: handles.append((hand[1028],thy_band[1028]))
    if 1029 in hand: handles.append((hand[1029],thy_band[1029]))
    if 1030 in hand: handles.append((hand[1030],thy_band[1030]))
    if 1028 in hand: labels.append(r'\textbf{\textrm{BABAR}}')
    if 1029 in hand: labels.append(r'\textbf{\textrm{BELLE}}')
    if 1030 in hand: labels.append(r'\textbf{\textrm{ARGUS}}')
    axs[1].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    py.tight_layout()

    handles,labels = [],[]
    if 1007 in hand: handles.append((hand[1007],thy_band[1007]))
    if 1008 in hand: handles.append((hand[1008],thy_band[1008]))
    if 1013 in hand: handles.append((hand[1013],thy_band[1013]))
    if 1007 in hand: labels.append(r'\textbf{\textrm{TPC (1984)}}')
    if 1008 in hand: labels.append(r'\textbf{\textrm{TPC (1988)}}')
    if 1013 in hand: labels.append(r'\textbf{\textrm{TOPAZ}}')
    axs[2].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1014 in hand: handles.append((hand[1014],thy_band[1014]))
    if 1018 in hand: handles.append((hand[1018],thy_band[1018]))
    if 1019 in hand: handles.append((hand[1019],thy_band[1019]))
    if 1025 in hand: handles.append((hand[1025],thy_band[1025]))
    if 1014 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1018 in hand: labels.append(r'\textbf{\textrm{ALEPH}}')
    if 1019 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 1025 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[3].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    
    handles,labels = [],[]
    if 1015 in hand: handles.append((hand[1015],thy_band[1015]))
    if 1026 in hand: handles.append((hand[1026],thy_band[1026]))
    if 1015 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1026 in hand: labels.append(r'\textbf{\textrm{DELHPI}}')
    axs[4].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1010 in hand: handles.append((hand[1010],thy_band[1010]))
    if 1016 in hand: handles.append((hand[1016],thy_band[1016]))
    if 1023 in hand: handles.append((hand[1023],thy_band[1023]))
    if 1010 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 1016 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1023 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    axs[5].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1011 in hand: handles.append((hand[1011],thy_band[1011]))
    if 1017 in hand: handles.append((hand[1017],thy_band[1017]))
    if 1024 in hand: handles.append((hand[1024],thy_band[1024]))
    if 1027 in hand: handles.append((hand[1027],thy_band[1027]))
    if 1011 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 1017 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1024 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 1027 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[6].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)


    py.tight_layout()

    py.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=0.02)
    filename = '%s/gallery/sia-pion-ratio.png'%wdir
    py.savefig(filename)
    print('Saving SIA pion figure to %s'%filename)

def plot_kaon_ratio(wdir):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if 'sia version' in conf and conf['sia version']==0: return

    idxs = []
    idxs.append(2001) # TASSO
    idxs.append(2002) # TASSO
    idxs.append(2003) # TASSO
    idxs.append(2004) # TASSO
    idxs.append(2005) # TASSO
    idxs.append(2006) # TASSO
    idxs.append(2007) # TPC
    idxs.append(2008) # TPC
    idxs.append(2013) # TOPAZ
    idxs.append(2014) # SLD
    idxs.append(2018) # ALEPH
    idxs.append(2019) # OPAL
    idxs.append(2025) # DELPHI
    idxs.append(2028) # BABAR
    idxs.append(2029) # BELL
    idxs.append(2030) # ARGUS
    idxs.append(2015) # SLD (uds)
    idxs.append(2026) # DELPHI (uds)
    idxs.append(2010) # TPC(c)
    idxs.append(2011) # TPC(b)
    idxs.append(2016) # SLD(c)
    idxs.append(2017) # SLD(b)
    idxs.append(2023) # OPAL(c)
    idxs.append(2024) # OPAL(b)
    idxs.append(2027) # DELPHI(b)

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sia' not in predictions['reactions']: return
  
    flag = True 
    data = predictions['reactions']['sia']
    for idx in data:
        if idx in idxs: flag = False
        predictions = copy.copy(data[idx]['prediction-rep'])
        shifts      = copy.copy(data[idx]['shift-rep'])
        #predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        del data[idx]['r-residuals']
        del data[idx]['n-residuals']
        if 'rres-rep' in data[idx]: del data[idx]['rres-rep']
        data[idx]['thy'] = np.mean(predictions, axis = 0)
        data[idx]['dthy'] = np.std(predictions, axis = 0)

    if flag: return

    nrows,ncols=2,4
    N = nrows*ncols - 1
    fig = py.figure(figsize=(ncols*7,nrows*5))

    axs={}
    for i in range(N+1):
        axs[i] = py.subplot(nrows,ncols,i+1)

    axs[7].axis("off")

    hand = {}
    thy_plot, thy_band = {}, {}
    for idx in idxs:

        if idx not in data: continue

        obs = data[idx]['obs'][0]
        had = data[idx]['hadron'][0]
        rs  = data[idx]['RS'][0]
        Z      = data[idx]['z']
        xp     = data[idx]['xp']
        p      = data[idx]['p']
        units_val = 1/data[idx]['units-val'][0]
        jac       = 1/data[idx]['jac']

        idn = data[idx]['is normalized?'][0]

        if idn: sigtot = np.ones(len(Z))
        else:   sigtot = data[idx]['sigtot']

        values = data[idx]['value']*jac*units_val/sigtot
        alpha  = data[idx]['alpha']*jac*units_val/sigtot

        #--TASSO
        if   idx==2001: ax,color,fmt,ms,zorder = axs[0],'firebrick','*',8,1
        elif idx==2002: ax,color,fmt,ms,zorder = axs[0],'darkgreen','^',6,1
        elif idx==2003: ax,color,fmt,ms,zorder = axs[0],'blue'     ,'o',6,1
        elif idx==2004: ax,color,fmt,ms,zorder = axs[0],'purple'   ,'v',6,1
        elif idx==2005: ax,color,fmt,ms,zorder = axs[0],'black'    ,'s',5,1
        elif idx==2006: ax,color,fmt,ms,zorder = axs[0],'orange'   ,'D',5,1
        #--Belle, BABAR, ARGUS (low RS),zorder
        elif idx==2028: ax,color,fmt,ms,zorder = axs[1],'firebrick','*',8,2
        elif idx==2029: ax,color,fmt,ms,zorder = axs[1],'darkgreen','^',6,1
        elif idx==2030: ax,color,fmt,ms,zorder = axs[1],'blue'     ,'o',6,1
        #--TPC, TOPAZ (medium energy)
        elif idx==2007: ax,color,fmt,ms,zorder = axs[2],'firebrick','*',8,1
        elif idx==2008: ax,color,fmt,ms,zorder = axs[2],'darkgreen','^',6,1
        elif idx==2013: ax,color,fmt,ms,zorder = axs[2],'blue'     ,'o',6,1
        #--SLD, ALEPH, OPAL, DELPHI (hirgh energy)
        elif idx==2014: ax,color,fmt,ms,zorder = axs[3],'firebrick','*',8,1
        elif idx==2018: ax,color,fmt,ms,zorder = axs[3],'darkgreen','^',6,1
        elif idx==2019: ax,color,fmt,ms,zorder = axs[3],'blue'     ,'o',6,1
        elif idx==2025: ax,color,fmt,ms,zorder = axs[3],'purple'   ,'v',6,1
        #--tagged data
        #--uds
        elif idx==2015: ax,color,fmt,ms,zorder = axs[4],'darkgreen','^',6,1
        elif idx==2026: ax,color,fmt,ms,zorder = axs[4],'purple'   ,'v',6,1
        #--charm
        elif idx==2010: ax,color,fmt,ms,zorder = axs[5],'firebrick','*',8,1
        elif idx==2016: ax,color,fmt,ms,zorder = axs[5],'darkgreen','^',6,1
        elif idx==2023: ax,color,fmt,ms,zorder = axs[5],'blue'     ,'o',6,1
        #--bottom
        elif idx==2011: ax,color,fmt,ms,zorder = axs[6],'firebrick','*',8,1
        elif idx==2017: ax,color,fmt,ms,zorder = axs[6],'darkgreen','^',6,1
        elif idx==2024: ax,color,fmt,ms,zorder = axs[6],'blue'     ,'o',6,1
        elif idx==2027: ax,color,fmt,ms,zorder = axs[6],'purple'   ,'v',6,1
        else:
            print('Skipping index %s'%idx)
            continue

        thy = data[idx]['thy'] *jac*units_val/sigtot
        std = data[idx]['dthy']*jac*units_val/sigtot
        hand[idx] = ax.errorbar(Z,values/thy,yerr=alpha/np.abs(thy),color=color,fmt=fmt,ms=ms,capsize=3.0,zorder=zorder)

        down = (thy - std)/thy
        up   = (thy + std)/thy
        thy_band[idx]  = ax.fill_between(Z,down,up,color=color,alpha=0.7,zorder=zorder)
        if len(Z)==1:
            z = [Z[0]*0.95, Z[0]*1.05]
            down, up, thy = [down[0],down[0]],[up[0],up[0]],[thy[0],thy[0]]
            thy_band[idx] = ax.fill_between(z,down,up,color=color,alpha=0.7,zorder=zorder)


    for i in range(N):
        axs[i].axhline(1,0,1,color='black',ls='-',alpha=0.8,zorder=10)
        axs[i].tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        axs[i].xaxis.set_tick_params(which='major',length=8)
        axs[i].yaxis.set_tick_params(which='major',length=8)
        axs[i].xaxis.set_tick_params(which='minor',length=4)
        axs[i].yaxis.set_tick_params(which='minor',length=4)
        axs[i].semilogx()
        axs[i].set_xlim(1.5e-2,0.99)
        axs[i].set_xticks([0.02,0.1,0.5])
        axs[i].set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])
        #axs[i].xaxis.set_minor_locator(MultipleLocator(0.10))
        #axs[i].xaxis.set_major_locator(MultipleLocator(0.20))
        if i in [0,1,2]:     axs[i].tick_params(labelbottom=False)
        #if i in [1,2,3,5,6]: axs[i].tick_params(labelleft=False)
        if i in [2,3,4,5,6]: 
            axs[i].set_xlabel(r'\boldmath$z$',size=40)
            axs[i].xaxis.set_label_coords(0.95,-0.01)

    axs[0].set_ylim(0.20,1.80)
    axs[0].set_yticks([0.5,1,1.5])
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[1].set_ylim(0.92,1.08)
    axs[1].set_yticks([0.95,1,1.05])
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.01))
    
    axs[2].set_ylim(0.70,1.30)
    axs[2].set_yticks([0.8,1,1.2])
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[3].set_ylim(0.50,1.50)
    axs[3].set_yticks([0.6,1,1.4])
    axs[3].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[4].set_ylim(0.50,1.50)
    axs[4].set_yticks([0.6,1,1.4])
    axs[4].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[5].set_ylim(0.20,1.80)
    axs[5].set_yticks([0.5,1,1.5])
    axs[5].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[6].set_ylim(0.20,1.80)
    axs[6].set_yticks([0.5,1,1.5])
    axs[6].yaxis.set_minor_locator(MultipleLocator(0.10))


   
    axs[0].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}$',    size=50) 
    axs[4].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}~(q)$',size=50) 

    axs[0].text(0.05, 0.85, r'\textrm{\textbf{TASSO}}',transform=axs[0].transAxes,size=40)
    axs[1].text(0.05, 0.85, r'\boldmath$\sqrt{s} \approx 10~{\rm GeV}$',transform=axs[1].transAxes,size=30)
    axs[2].text(0.05, 0.85, r'\boldmath$\sqrt{s} = 29,58 ~{\rm GeV}$'  ,transform=axs[2].transAxes,size=30)
    axs[3].text(0.05, 0.85, r'\boldmath$\sqrt{s} = M_Z$'               ,transform=axs[3].transAxes,size=30)
    axs[4].text(0.05, 0.85, r'\boldmath$(uds)$',transform=axs[4].transAxes,size=40)
    axs[5].text(0.05, 0.85, r'\boldmath$(c)$'  ,transform=axs[5].transAxes,size=40)
    axs[6].text(0.05, 0.85, r'\boldmath$(b)$'  ,transform=axs[6].transAxes,size=40)

    axs[7].text(0.05, 0.20, r'\textrm{\textbf{SIA \boldmath$K^{\pm}$}}',transform=axs[7].transAxes,size=60)

    fs = 25
    handles,labels = [],[]
    if 2001 in hand: handles.append((hand[2001],thy_band[2001]))
    if 2002 in hand: handles.append((hand[2002],thy_band[2002]))
    if 2003 in hand: handles.append((hand[2003],thy_band[2003]))
    if 2004 in hand: handles.append((hand[2004],thy_band[2004]))
    if 2005 in hand: handles.append((hand[2005],thy_band[2005]))
    if 2006 in hand: handles.append((hand[2006],thy_band[2006]))
    if 2001 in hand: labels.append(r'\textbf{\textrm{12 GeV}}')
    if 2002 in hand: labels.append(r'\textbf{\textrm{14 GeV}}')
    if 2003 in hand: labels.append(r'\textbf{\textrm{22 GeV}}')
    if 2004 in hand: labels.append(r'\textbf{\textrm{30 GeV}}')
    if 2005 in hand: labels.append(r'\textbf{\textrm{34 GeV}}')
    if 2006 in hand: labels.append(r'\textbf{\textrm{44 GeV}}')
    axs[0].legend(handles,labels,frameon=False,fontsize=20,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 2028 in hand: handles.append((hand[2028],thy_band[2028]))
    if 2029 in hand: handles.append((hand[2029],thy_band[2029]))
    if 2030 in hand: handles.append((hand[2030],thy_band[2030]))
    if 2028 in hand: labels.append(r'\textbf{\textrm{BABAR}}')
    if 2029 in hand: labels.append(r'\textbf{\textrm{BELLE}}')
    if 2030 in hand: labels.append(r'\textbf{\textrm{ARGUS}}')
    axs[1].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    py.tight_layout()

    handles,labels = [],[]
    if 2007 in hand: handles.append((hand[2007],thy_band[2007]))
    if 2008 in hand: handles.append((hand[2008],thy_band[2008]))
    if 2013 in hand: handles.append((hand[2013],thy_band[2013]))
    if 2007 in hand: labels.append(r'\textbf{\textrm{TPC (1984)}}')
    if 2008 in hand: labels.append(r'\textbf{\textrm{TPC (1988)}}')
    if 2013 in hand: labels.append(r'\textbf{\textrm{TOPAZ}}')
    axs[2].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 2014 in hand: handles.append((hand[2014],thy_band[2014]))
    if 2018 in hand: handles.append((hand[2018],thy_band[2018]))
    if 2019 in hand: handles.append((hand[2019],thy_band[2019]))
    if 2025 in hand: handles.append((hand[2025],thy_band[2025]))
    if 2014 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 2018 in hand: labels.append(r'\textbf{\textrm{ALEPH}}')
    if 2019 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 2025 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[3].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    
    handles,labels = [],[]
    if 2015 in hand: handles.append((hand[2015],thy_band[2015]))
    if 2026 in hand: handles.append((hand[2026],thy_band[2026]))
    if 2015 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 2026 in hand: labels.append(r'\textbf{\textrm{DELHPI}}')
    axs[4].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 2010 in hand: handles.append((hand[2010],thy_band[2010]))
    if 2016 in hand: handles.append((hand[2016],thy_band[2016]))
    if 2023 in hand: handles.append((hand[2023],thy_band[2023]))
    if 2010 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 2016 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 2023 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    axs[5].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 2011 in hand: handles.append((hand[2011],thy_band[2011]))
    if 2017 in hand: handles.append((hand[2017],thy_band[2017]))
    if 2024 in hand: handles.append((hand[2024],thy_band[2024]))
    if 2027 in hand: handles.append((hand[2027],thy_band[2027]))
    if 2011 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 2017 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 2024 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 2027 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[6].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)


    py.tight_layout()

    py.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=0.02)
    filename = '%s/gallery/sia-kaon-ratio.png'%wdir
    py.savefig(filename)
    print('Saving SIA kaon figure to %s'%filename)

def plot_hadron_ratio(wdir):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if 'sia version' in conf and conf['sia version']==0: return

    idxs = []
    idxs.append(4003) # TASSO
    idxs.append(4008) # TASSO
    idxs.append(4009) # TASSO
    idxs.append(4010) # TASSO
    idxs.append(4011) # TASSO
    idxs.append(4012) # TASSO
    idxs.append(4004) # TPC
    idxs.append(4002) # SLD
    idxs.append(4000) # ALEPH
    idxs.append(4007) # OPAL
    idxs.append(4001) # DELPHI
    idxs.append(4014) # SLD(c)
    idxs.append(4015) # SLD(b)
    idxs.append(4005) # OPAL(c)
    idxs.append(4006) # OPAL(b)
    idxs.append(4013) # DELPHI(b)
    idxs.append(4016) # SLD(uds)
    idxs.append(4017) # DELPHI(uds)

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sia' not in predictions['reactions']: return
   
    flag = True
    data = predictions['reactions']['sia']
    for idx in data:
        if idx in idxs: flag = False
        predictions = copy.copy(data[idx]['prediction-rep'])
        shifts      = copy.copy(data[idx]['shift-rep'])
        #predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        del data[idx]['r-residuals']
        del data[idx]['n-residuals']
        if 'rres-rep' in data[idx]: del data[idx]['rres-rep']
        data[idx]['thy'] = np.mean(predictions, axis = 0)
        data[idx]['dthy'] = np.std(predictions, axis = 0)

    if flag: return

    nrows,ncols=2,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*7,nrows*5))

    axs={}
    for i in range(N):
        axs[i] = py.subplot(nrows,ncols,i+1)

    hand = {}
    thy_plot, thy_band = {}, {}
    for idx in idxs:

        if idx not in data: continue

        obs = data[idx]['obs'][0]
        had = data[idx]['hadron'][0]
        rs  = data[idx]['RS'][0]
        Z      = data[idx]['z']
        xp     = data[idx]['xp']
        p      = data[idx]['p']
        units_val = 1/data[idx]['units-val'][0]
        jac       = 1/data[idx]['jac']

        idn = data[idx]['is normalized?'][0]

        if idn: sigtot = np.ones(len(Z))
        else:   sigtot = data[idx]['sigtot']

        values = data[idx]['value']*jac*units_val/sigtot
        alpha  = data[idx]['alpha']*jac*units_val/sigtot

        #--TASSO
        if   idx==4003: ax,color,fmt,ms = axs[0],'firebrick','*',8
        elif idx==4008: ax,color,fmt,ms = axs[0],'darkgreen','^',6
        elif idx==4009: ax,color,fmt,ms = axs[0],'blue'     ,'o',6
        elif idx==4010: ax,color,fmt,ms = axs[0],'purple'   ,'v',6
        elif idx==4011: ax,color,fmt,ms = axs[0],'black'    ,'s',5
        elif idx==4012: ax,color,fmt,ms = axs[0],'orange'   ,'D',5
        #--TPC (medium energy)
        elif idx==4004: ax,color,fmt,ms = axs[1],'darkgreen','^',6
        #--SLD, ALEPH, OPAL, DELPHI (high energy)
        elif idx==4002: ax,color,fmt,ms = axs[2],'firebrick','*',8
        elif idx==4000: ax,color,fmt,ms = axs[2],'darkgreen','^',6
        elif idx==4007: ax,color,fmt,ms = axs[2],'blue'     ,'o',6
        elif idx==4001: ax,color,fmt,ms = axs[2],'purple'   ,'v',6
        #--tagged data
        #--uds
        elif idx==4016: ax,color,fmt,ms = axs[3],'darkgreen','^',6
        elif idx==4017: ax,color,fmt,ms = axs[3],'purple'   ,'v',6
        #--charm
        elif idx==4014: ax,color,fmt,ms = axs[4],'darkgreen','^',6
        elif idx==4006: ax,color,fmt,ms = axs[4],'blue'     ,'o',6
        #--bottom
        elif idx==4015: ax,color,fmt,ms = axs[5],'darkgreen','^',6
        elif idx==4005: ax,color,fmt,ms = axs[5],'blue'     ,'o',6
        elif idx==4013: ax,color,fmt,ms = axs[5],'purple'   ,'v',6
        else:
            print('Skipping index %s'%idx)
            continue

        thy = data[idx]['thy'] *jac*units_val/sigtot
        std = data[idx]['dthy']*jac*units_val/sigtot
        hand[idx] = ax.errorbar(Z,values/thy,yerr=alpha/np.abs(thy),color=color,fmt=fmt,ms=ms,capsize=3.0)

        down = (thy - std)/thy
        up   = (thy + std)/thy
        thy_band[idx]  = ax.fill_between(Z,down,up,color=color,alpha=0.7)
        if len(Z)==1:
            z = [Z[0]*0.95, Z[0]*1.05]
            down, up, thy = [down[0],down[0]],[up[0],up[0]],[thy[0],thy[0]]
            thy_band[idx] = ax.fill_between(z,down,up,color=color,alpha=0.7)


    for i in range(N):
        axs[i].axhline(1,0,1,color='black',ls='-',alpha=0.8,zorder=10)
        axs[i].tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        axs[i].xaxis.set_tick_params(which='major',length=8)
        axs[i].yaxis.set_tick_params(which='major',length=8)
        axs[i].xaxis.set_tick_params(which='minor',length=4)
        axs[i].yaxis.set_tick_params(which='minor',length=4)
        axs[i].semilogx()
        axs[i].set_xlim(1.5e-2,0.99)
        axs[i].set_xticks([0.02,0.1,0.5])
        axs[i].set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])
        if i in [0,1,2]:     axs[i].tick_params(labelbottom=False)
        if i in [3,4,5]: 
            axs[i].set_xlabel(r'\boldmath$z$',size=40)
            axs[i].xaxis.set_label_coords(0.95,-0.01)



    axs[0].set_ylim(0.41,1.59)
    axs[0].set_yticks([0.5,1,1.5])
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[1].set_ylim(0.81,1.19)
    axs[1].set_yticks([0.9,1,1.1])
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[2].set_ylim(0.81,1.19)
    axs[2].set_yticks([0.9,1,1.1])
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[3].set_ylim(0.61,1.39)
    axs[3].set_yticks([0.8,1,1.2])
    axs[3].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[4].set_ylim(0.61,1.39)
    axs[4].set_yticks([0.8,1,1.2])
    axs[4].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[5].set_ylim(0.61,1.39)
    axs[5].set_yticks([0.8,1,1.2])
    axs[5].yaxis.set_minor_locator(MultipleLocator(0.10))


    axs[0].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}$',    size=50) 
    axs[3].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}~(q)$',size=50) 

    axs[0].text(0.05, 0.85, r'\textrm{\textbf{TASSO}}',transform=axs[0].transAxes,size=40)
    #axs[1].text(0.50, 0.85, r'\boldmath$\sqrt{s} \approx 10~{\rm GeV}$',transform=axs[1].transAxes,size=30)
    axs[1].text(0.05, 0.85, r'\boldmath$\sqrt{s} = 29,58 ~{\rm GeV}$'  ,transform=axs[1].transAxes,size=30)
    axs[2].text(0.05, 0.85, r'\boldmath$\sqrt{s} = M_Z$'               ,transform=axs[2].transAxes,size=30)
    axs[3].text(0.05, 0.85, r'\boldmath$(uds)$',transform=axs[3].transAxes,size=40)
    axs[4].text(0.05, 0.85, r'\boldmath$(c)$',transform=axs[4].transAxes,size=40)
    axs[5].text(0.05, 0.85, r'\boldmath$(b)$',transform=axs[5].transAxes,size=40)

    axs[5].text(0.03, 0.25, r'\textrm{\textbf{SIA \boldmath$h^{\pm}$}}',transform=axs[5].transAxes,size=60)

    fs = 25
    handles,labels = [],[]
    if 4003 in hand: handles.append((hand[4003],thy_band[4003]))
    if 4008 in hand: handles.append((hand[4008],thy_band[4008]))
    if 4009 in hand: handles.append((hand[4009],thy_band[4009]))
    if 4010 in hand: handles.append((hand[4010],thy_band[4010]))
    if 4011 in hand: handles.append((hand[4011],thy_band[4011]))
    if 4012 in hand: handles.append((hand[4012],thy_band[4012]))
    if 4003 in hand: labels.append(r'\textbf{\textrm{12 GeV}}')
    if 4008 in hand: labels.append(r'\textbf{\textrm{14 GeV}}')
    if 4009 in hand: labels.append(r'\textbf{\textrm{22 GeV}}')
    if 4010 in hand: labels.append(r'\textbf{\textrm{30 GeV}}')
    if 4011 in hand: labels.append(r'\textbf{\textrm{35 GeV}}')
    if 4012 in hand: labels.append(r'\textbf{\textrm{44 GeV}}')
    axs[0].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 2, columnspacing = 1.0)

    handles,labels = [],[]
    if 4004 in hand: handles.append((hand[4004],thy_band[4004]))
    if 4004 in hand: labels.append(r'\textbf{\textrm{TPC (1988)}}')
    axs[1].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 4002 in hand: handles.append((hand[4002],thy_band[4002]))
    if 4000 in hand: handles.append((hand[4000],thy_band[4000]))
    if 4007 in hand: handles.append((hand[4007],thy_band[4007]))
    if 4001 in hand: handles.append((hand[4001],thy_band[4001]))
    if 4002 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 4000 in hand: labels.append(r'\textbf{\textrm{ALEPH}}')
    if 4007 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 4001 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[2].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 2, columnspacing = 1.0)
   
    handles,labels = [],[]
    if 4016 in hand: handles.append((hand[4016],thy_band[4016]))
    if 4017 in hand: handles.append((hand[4017],thy_band[4017]))
    if 4016 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 4017 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[3].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
 
    handles,labels = [],[]
    if 4014 in hand: handles.append((hand[4014],thy_band[4014]))
    if 4006 in hand: handles.append((hand[4006],thy_band[4006]))
    if 4014 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 4006 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    axs[4].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 4015 in hand: handles.append((hand[4015],thy_band[4015]))
    if 4005 in hand: handles.append((hand[4005],thy_band[4005]))
    if 4013 in hand: handles.append((hand[4013],thy_band[4013]))
    if 4015 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 4005 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 4013 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[5].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)


    py.tight_layout()

    py.subplots_adjust(left=0.09, bottom=0.05, right=0.99, top=0.99, wspace=0.12, hspace=0.02)
    filename = '%s/gallery/sia-hadron-ratio.png'%wdir
    py.savefig(filename)
    print('Saving SIA hadron figure to %s'%filename)


#--difference plots
def plot_pion_difference(wdir):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if 'sia version' in conf and conf['sia version']==0: return

    idxs = []
    idxs.append(1001) # TASSO
    idxs.append(1002) # TASSO
    idxs.append(1003) # TASSO
    idxs.append(1004) # TASSO
    idxs.append(1005) # TASSO
    idxs.append(1006) # TASSO
    idxs.append(1007) # TPC
    idxs.append(1008) # TPC
    idxs.append(1013) # TOPAZ
    idxs.append(1014) # SLD
    idxs.append(1018) # ALEPH
    idxs.append(1019) # OPAL
    idxs.append(1025) # DELPHI
    idxs.append(1028) # BABAR
    idxs.append(1029) # BELL
    idxs.append(1030) # ARGUS
    idxs.append(1015) # SLD (uds)
    idxs.append(1026) # DELPHI (uds)
    idxs.append(1010) # TPC(c)
    idxs.append(1011) # TPC(b)
    idxs.append(1016) # SLD(c)
    idxs.append(1017) # SLD(b)
    idxs.append(1023) # OPAL(c)
    idxs.append(1024) # OPAL(b)
    idxs.append(1027) # DELPHI(b)

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'sia' not in predictions['reactions']: return
  
    flag = True 
    data = predictions['reactions']['sia']
    for idx in data:
        if idx in idxs: flag = False
        predictions = copy.copy(data[idx]['prediction-rep'])
        shifts      = copy.copy(data[idx]['shift-rep'])
        #predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        del data[idx]['r-residuals']
        del data[idx]['n-residuals']
        if 'rres-rep' in data[idx]: del data[idx]['rres-rep']
        data[idx]['thy'] = np.mean(predictions, axis = 0)
        data[idx]['dthy'] = np.std(predictions, axis = 0)

    if flag: return

    nrows,ncols=2,4
    N = nrows*ncols - 1
    fig = py.figure(figsize=(ncols*7,nrows*5))

    axs={}
    for i in range(N+1):
        axs[i] = py.subplot(nrows,ncols,i+1)

    axs[7].axis("off")

    hand = {}
    thy_plot, thy_band = {}, {}
    for idx in idxs:

        if idx not in data: continue

        obs = data[idx]['obs'][0]
        had = data[idx]['hadron'][0]
        rs  = data[idx]['RS'][0]
        Z      = data[idx]['z']
        xp     = data[idx]['xp']
        p      = data[idx]['p']
        units_val = 1/data[idx]['units-val'][0]
        jac       = 1/data[idx]['jac']

        idn = data[idx]['is normalized?'][0]

        if idn: sigtot = np.ones(len(Z))
        else:   sigtot = data[idx]['sigtot']

        values = data[idx]['value']*jac*units_val/sigtot
        alpha  = data[idx]['alpha']*jac*units_val/sigtot

        #--TASSO
        if   idx==1001: ax,color,fmt,ms,zorder = axs[0],'firebrick','*',8,1
        elif idx==1002: ax,color,fmt,ms,zorder = axs[0],'darkgreen','^',6,1
        elif idx==1003: ax,color,fmt,ms,zorder = axs[0],'blue'     ,'o',6,1
        elif idx==1004: ax,color,fmt,ms,zorder = axs[0],'purple'   ,'v',6,1
        elif idx==1005: ax,color,fmt,ms,zorder = axs[0],'black'    ,'s',5,1
        elif idx==1006: ax,color,fmt,ms,zorder = axs[0],'orange'   ,'D',5,1
        #--Belle, BABAR, ARGUS (low RS),zorder
        elif idx==1028: ax,color,fmt,ms,zorder = axs[1],'firebrick','*',8,2
        elif idx==1029: ax,color,fmt,ms,zorder = axs[1],'darkgreen','^',6,1
        elif idx==1030: ax,color,fmt,ms,zorder = axs[1],'blue'     ,'o',6,1
        #--TPC, TOPAZ (medium energy)
        elif idx==1007: ax,color,fmt,ms,zorder = axs[2],'firebrick','*',8,1
        elif idx==1008: ax,color,fmt,ms,zorder = axs[2],'darkgreen','^',6,1
        elif idx==1013: ax,color,fmt,ms,zorder = axs[2],'blue'     ,'o',6,1
        #--SLD, ALEPH, OPAL, DELPHI (hirgh energy)
        elif idx==1014: ax,color,fmt,ms,zorder = axs[3],'firebrick','*',8,1
        elif idx==1018: ax,color,fmt,ms,zorder = axs[3],'darkgreen','^',6,1
        elif idx==1019: ax,color,fmt,ms,zorder = axs[3],'blue'     ,'o',6,1
        elif idx==1025: ax,color,fmt,ms,zorder = axs[3],'purple'   ,'v',6,1
        #--tagged data
        #--uds
        elif idx==1015: ax,color,fmt,ms,zorder = axs[4],'darkgreen','^',6,1
        elif idx==1026: ax,color,fmt,ms,zorder = axs[4],'purple'   ,'v',6,1
        #--charm
        elif idx==1010: ax,color,fmt,ms,zorder = axs[5],'firebrick','*',8,1
        elif idx==1016: ax,color,fmt,ms,zorder = axs[5],'darkgreen','^',6,1
        elif idx==1023: ax,color,fmt,ms,zorder = axs[5],'blue'     ,'o',6,1
        #--bottom
        elif idx==1011: ax,color,fmt,ms,zorder = axs[6],'firebrick','*',8,1
        elif idx==1017: ax,color,fmt,ms,zorder = axs[6],'darkgreen','^',6,1
        elif idx==1024: ax,color,fmt,ms,zorder = axs[6],'blue'     ,'o',6,1
        elif idx==1027: ax,color,fmt,ms,zorder = axs[6],'purple'   ,'v',6,1
        else:
            print('Skipping index %s'%idx)
            continue

        thy = data[idx]['thy'] *jac*units_val/sigtot
        std = data[idx]['dthy']*jac*units_val/sigtot
        hand[idx] = ax.errorbar(Z,(values-thy)/thy,yerr=alpha/np.abs(thy),color=color,fmt=fmt,ms=ms,capsize=3.0,zorder=zorder)

        down = -std/thy
        up   =  std/thy
        thy_band[idx]  = ax.fill_between(Z,down,up,color=color,alpha=0.7,zorder=zorder)
        if len(Z)==1:
            z = [Z[0]*0.95, Z[0]*1.05]
            down, up, thy = [down[0],down[0]],[up[0],up[0]],[thy[0],thy[0]]
            thy_band[idx] = ax.fill_between(z,down,up,color=color,alpha=0.7,zorder=zorder)


    for i in range(N):
        axs[i].axhline(0,0,1,color='black',ls='-',alpha=0.8,zorder=10)
        axs[i].tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        axs[i].xaxis.set_tick_params(which='major',length=8)
        axs[i].yaxis.set_tick_params(which='major',length=8)
        axs[i].xaxis.set_tick_params(which='minor',length=4)
        axs[i].yaxis.set_tick_params(which='minor',length=4)
        #axs[i].semilogx()
        axs[i].set_xlim(1.5e-2,0.99)
        axs[i].set_xticks([0.02,0.1,0.5])
        axs[i].set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])
        #axs[i].xaxis.set_minor_locator(MultipleLocator(0.10))
        #axs[i].xaxis.set_major_locator(MultipleLocator(0.20))
        if i in [0,1,2]:     axs[i].tick_params(labelbottom=False)
        #if i in [1,2,3,5,6]: axs[i].tick_params(labelleft=False)
        if i in [2,3,4,5,6]: 
            axs[i].set_xlabel(r'\boldmath$z$',size=40)
            axs[i].xaxis.set_label_coords(0.95,-0.01)

    #axs[0].set_ylim(0.20,1.80)
    #axs[0].set_yticks([0.5,1,1.5])
    #axs[0].yaxis.set_minor_locator(MultipleLocator(0.10))

    axs[1].set_ylim(-0.5,0.5)
    #axs[1].set_yticks([0.95,1,1.05])
    #axs[1].yaxis.set_minor_locator(MultipleLocator(0.01))
    #
    #axs[2].set_ylim(0.70,1.30)
    #axs[2].set_yticks([0.8,1,1.2])
    #axs[2].yaxis.set_minor_locator(MultipleLocator(0.10))

    #axs[3].set_ylim(0.50,1.50)
    #axs[3].set_yticks([0.6,1,1.4])
    #axs[3].yaxis.set_minor_locator(MultipleLocator(0.10))

    #axs[4].set_ylim(0.50,1.50)
    #axs[4].set_yticks([0.6,1,1.4])
    #axs[4].yaxis.set_minor_locator(MultipleLocator(0.10))

    #axs[5].set_ylim(0.20,1.80)
    #axs[5].set_yticks([0.5,1,1.5])
    #axs[5].yaxis.set_minor_locator(MultipleLocator(0.10))

    #axs[6].set_ylim(0.20,1.80)
    #axs[6].set_yticks([0.5,1,1.5])
    #axs[6].yaxis.set_minor_locator(MultipleLocator(0.10))


   
    axs[0].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}$',    size=50) 
    axs[4].set_ylabel(r'\boldmath$\frac{1}{\sigma} \frac{d\sigma}{dz}~(q)$',size=50) 

    axs[0].text(0.05, 0.85, r'\textrm{\textbf{TASSO}}',transform=axs[0].transAxes,size=40)
    axs[1].text(0.05, 0.85, r'\boldmath$\sqrt{s} \approx 10~{\rm GeV}$',transform=axs[1].transAxes,size=30)
    axs[2].text(0.05, 0.85, r'\boldmath$\sqrt{s} = 29,58 ~{\rm GeV}$'  ,transform=axs[2].transAxes,size=30)
    axs[3].text(0.05, 0.85, r'\boldmath$\sqrt{s} = M_Z$'               ,transform=axs[3].transAxes,size=30)
    axs[4].text(0.05, 0.85, r'\boldmath$(uds)$',transform=axs[4].transAxes,size=40)
    axs[5].text(0.05, 0.85, r'\boldmath$(c)$'  ,transform=axs[5].transAxes,size=40)
    axs[6].text(0.05, 0.85, r'\boldmath$(b)$'  ,transform=axs[6].transAxes,size=40)

    axs[7].text(0.05, 0.20, r'\textrm{\textbf{SIA \boldmath$\pi^{\pm}$}}',transform=axs[7].transAxes,size=60)

    fs = 25
    handles,labels = [],[]
    if 1001 in hand: handles.append((hand[1001],thy_band[1001]))
    if 1002 in hand: handles.append((hand[1002],thy_band[1002]))
    if 1003 in hand: handles.append((hand[1003],thy_band[1003]))
    if 1004 in hand: handles.append((hand[1004],thy_band[1004]))
    if 1005 in hand: handles.append((hand[1005],thy_band[1005]))
    if 1006 in hand: handles.append((hand[1006],thy_band[1006]))
    if 1001 in hand: labels.append(r'\textbf{\textrm{12 GeV}}')
    if 1002 in hand: labels.append(r'\textbf{\textrm{14 GeV}}')
    if 1003 in hand: labels.append(r'\textbf{\textrm{22 GeV}}')
    if 1004 in hand: labels.append(r'\textbf{\textrm{30 GeV}}')
    if 1005 in hand: labels.append(r'\textbf{\textrm{34 GeV}}')
    if 1006 in hand: labels.append(r'\textbf{\textrm{44 GeV}}')
    axs[0].legend(handles,labels,frameon=False,fontsize=20,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1028 in hand: handles.append((hand[1028],thy_band[1028]))
    if 1029 in hand: handles.append((hand[1029],thy_band[1029]))
    if 1030 in hand: handles.append((hand[1030],thy_band[1030]))
    if 1028 in hand: labels.append(r'\textbf{\textrm{BABAR}}')
    if 1029 in hand: labels.append(r'\textbf{\textrm{BELLE}}')
    if 1030 in hand: labels.append(r'\textbf{\textrm{ARGUS}}')
    axs[1].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    py.tight_layout()

    handles,labels = [],[]
    if 1007 in hand: handles.append((hand[1007],thy_band[1007]))
    if 1008 in hand: handles.append((hand[1008],thy_band[1008]))
    if 1013 in hand: handles.append((hand[1013],thy_band[1013]))
    if 1007 in hand: labels.append(r'\textbf{\textrm{TPC (1984)}}')
    if 1008 in hand: labels.append(r'\textbf{\textrm{TPC (1988)}}')
    if 1013 in hand: labels.append(r'\textbf{\textrm{TOPAZ}}')
    axs[2].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1014 in hand: handles.append((hand[1014],thy_band[1014]))
    if 1018 in hand: handles.append((hand[1018],thy_band[1018]))
    if 1019 in hand: handles.append((hand[1019],thy_band[1019]))
    if 1025 in hand: handles.append((hand[1025],thy_band[1025]))
    if 1014 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1018 in hand: labels.append(r'\textbf{\textrm{ALEPH}}')
    if 1019 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 1025 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[3].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)
    
    handles,labels = [],[]
    if 1015 in hand: handles.append((hand[1015],thy_band[1015]))
    if 1026 in hand: handles.append((hand[1026],thy_band[1026]))
    if 1015 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1026 in hand: labels.append(r'\textbf{\textrm{DELHPI}}')
    axs[4].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1010 in hand: handles.append((hand[1010],thy_band[1010]))
    if 1016 in hand: handles.append((hand[1016],thy_band[1016]))
    if 1023 in hand: handles.append((hand[1023],thy_band[1023]))
    if 1010 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 1016 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1023 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    axs[5].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)

    handles,labels = [],[]
    if 1011 in hand: handles.append((hand[1011],thy_band[1011]))
    if 1017 in hand: handles.append((hand[1017],thy_band[1017]))
    if 1024 in hand: handles.append((hand[1024],thy_band[1024]))
    if 1027 in hand: handles.append((hand[1027],thy_band[1027]))
    if 1011 in hand: labels.append(r'\textbf{\textrm{TPC}}')
    if 1017 in hand: labels.append(r'\textbf{\textrm{SLD}}')
    if 1024 in hand: labels.append(r'\textbf{\textrm{OPAL}}')
    if 1027 in hand: labels.append(r'\textbf{\textrm{DELPHI}}')
    axs[6].legend(handles,labels,frameon=False,fontsize=fs,loc=(0.00,0.00),handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 1.0)


    py.tight_layout()

    py.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=0.02)
    filename = '%s/gallery/sia-pion-difference.png'%wdir
    py.savefig(filename)
    print('Saving SIA pion figure to %s'%filename)
















