#!/usr/bin/env python
import sys, os
import numpy as np
import copy
import pandas as pd
import scipy as sp
from scipy.interpolate import griddata

## matplotlib
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.latex.preview']=True
import pylab as py
from matplotlib.ticker import MultipleLocator

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack fitlib
from fitlib.resman import RESMAN

## from fitpack analysis
from analysis.corelib import core
from analysis.corelib import classifier

def get_theory(PLOT,nbins,loop=True,funcQ2=True,funcX=False,functheta=False):

    #--interpolate theoretical values across Q2 or X

    theory = {}

    if funcQ2:     svar = 'Q2'
    if funcX:      svar = 'X'
    if functheta:  svar = 'theta'

    theory = {_:{} for _ in [svar,'value','std']}
    for key in range(nbins):
        var = []
        thy = []
        std = []
        
        #--if plotting for multiple experiments, loop over and combine
        if loop:
            for exp in PLOT:
                var. extend(PLOT[exp][svar][key])
                thy.extend(PLOT[exp]['theory'][key])
                std.extend(PLOT[exp]['std'][key])
        else:
            var.extend(PLOT[svar][key])
            thy.extend(PLOT['theory'][key])
            std.extend(PLOT['std'][key])

        #--if nothing in bin, skip
        if len(var) == 0: continue

        vmin = np.min(var)
        vmax = np.max(var)
        theory[svar][key]  = np.geomspace(vmin,vmax,100)

        #--if more than one value, interpolate between them
        if len(var) > 1:
            theory['value'][key] = griddata(np.array(var),np.array(thy),theory[svar][key],method='linear')
            theory['std'][key]   = griddata(np.array(var),np.array(std),theory[svar][key],method='linear')
        else:
            theory['value'][key] = np.ones(100)*thy 
            theory['std'][key]   = np.ones(100)*std


    return theory


def get_details(exp):

    #--get details for plotting

    if exp=='NMC' :          color, marker, ms = 'firebrick', '*', 6
    if exp=='SLAC' :         color, marker, ms = 'darkgreen', '^', 6
    if exp=='BCDMS':         color, marker, ms = 'blue'     , 'o', 6

    if exp=='HERA 10026':    color, marker, ms = 'black'    , '.', 8 
    if exp=='HERA 10030':    color, marker, ms = 'firebrick', 'D', 8 
    if exp=='HERA 10031':    color, marker, ms = 'blue'     , '^', 6 
    if exp=='HERA 10032':    color, marker, ms = 'darkgreen', 's', 6 

    if exp=='BCDMS p':         color, marker, ms = 'firebrick', '*', 6
    if exp=='BCDMS d':         color, marker, ms = 'darkgreen', '^', 6

    if exp=='SLAC p':         color, marker, ms = 'firebrick', '*', 6
    if exp=='SLAC d':         color, marker, ms = 'darkgreen', '^', 6
    
    if exp=='JLab p':         color, marker, ms = 'firebrick', '*', 6
    if exp=='JLab d':         color, marker, ms = 'darkgreen', '^', 6

    if exp=='E140 d':          color, marker, ms = 'darkgreen', '^', 6
    if exp=='E140x p':         color, marker, ms = 'firebrick', '*', 6
    if exp=='E140x d':         color, marker, ms = 'darkgreen', '^', 6
    return color,marker,ms




def plot_CLAS12(wdir, data):

    if 10090 not in data: return

    nrows, ncols = 3, 3
    py.figure(figsize = (ncols * 6, nrows * 7))
    ax01 = py.subplot(nrows, ncols, 1)
    ax02 = py.subplot(nrows, ncols, 2)
    ax03 = py.subplot(nrows, ncols, 3)
    ax11 = py.subplot(nrows, ncols, 4)
    ax12 = py.subplot(nrows, ncols, 5)
    ax13 = py.subplot(nrows, ncols, 6)
    ax21 = py.subplot(nrows, ncols, 7)
    ax22 = py.subplot(nrows, ncols, 8)
    ax23 = py.subplot(nrows, ncols, 9)

    #################################
    #--Plot F2d/F2h from JLab Hall C
    #################################

    nbins = 9

    p   = data[10090] ## CLAS12

    DATA = {}
    DATA['CLAS12']  = pd.DataFrame(p)

    #PLOT = {}
    #theory = {}
    #for exp in DATA:
    #    query = get_Q2bins(DATA[exp],'CLAS12')
    #    PLOT[exp] = get_plot(query)
    #    theory[exp] = get_theory(PLOT[exp],nbins,loop=False,funcX=True)

    SAVE = {}
    SAVE['X']  = DATA['CLAS12']['X']
    SAVE['Q2'] = DATA['CLAS12']['Q2']

    SAVE['F2 mean'] = []
    SAVE['F2 std']  = []

    W2cut = 3.5
    hand = {}
    #--plot data points
    for exp in DATA:
        bins  = DATA[exp]['bin']
        for key in range(nbins):
            if key==0: ax = ax01
            if key==1: ax = ax02
            if key==2: ax = ax03
            if key==3: ax = ax11
            if key==4: ax = ax12
            if key==5: ax = ax13
            if key==6: ax = ax21
            if key==7: ax = ax22
            if key==8: ax = ax23
            X     = DATA[exp]['X'][bins==key+1]
            val   = DATA[exp]['value'][bins==key+1]
            alpha = DATA[exp]['alpha'][bins==key+1]
            color,marker,ms = get_details(exp)
            Q2 = DATA[exp]['Q2'][bins==key+1]
            W2 = 0.938**2 + Q2/X - Q2
            hand0 = ax.errorbar(X[W2> W2cut],val[W2> W2cut],alpha[W2> W2cut],marker=marker,color='firebrick',ms=ms,capsize=2.5,linestyle='none')
            hand1 = ax.errorbar(X[W2<=W2cut],val[W2<=W2cut],alpha[W2<=W2cut],marker=marker,color='darkgreen',ms=ms,capsize=2.5,linestyle='none')

            mean = DATA[exp]['thy-0'] [bins==key+1]
            std  = DATA[exp]['dthy-0'][bins==key+1]
            down = mean - std
            up   = mean + std
            SAVE['F2 mean'].extend(mean)
            SAVE['F2 std'] .extend(std)
            thy_plot0 ,= ax.plot(X[W2> W2cut],mean[W2> W2cut],linestyle='solid',    color='firebrick')
            thy_band0  = ax.fill_between(X[W2> W2cut],down[W2> W2cut],up[W2> W2cut],color='firebrick',alpha=0.5)
            thy_plot1 ,= ax.plot(X[W2<=W2cut],mean[W2<=W2cut],linestyle='solid',    color='darkgreen')
            thy_band1  = ax.fill_between(X[W2<=W2cut],down[W2<=W2cut],up[W2<=W2cut],color='darkgreen',alpha=0.5)

    SAVE['F2 mean'] = np.array(SAVE['F2 mean'])
    SAVE['F2 std']  = np.array(SAVE['F2 std'])

    #SAVE = pd.DataFrame(SAVE) 
    #SAVE.to_csv('shujie.csv')

    for ax in [ax01,ax02,ax03,ax11,ax12,ax13,ax21,ax22,ax23]:
        ax.set_xlim(0.21, 0.99)
        ax.semilogy()

        ax.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 30)

        ax.tick_params(axis='both', which = 'major', length = 8)
        ax.tick_params(axis='both', which = 'minor', length = 4)

        ax.set_xticks([0.4,0.6,0.8])
        ax.xaxis.set_minor_locator(MultipleLocator(0.10))

    for ax in [ax01,ax02,ax03]:
        ax.set_ylim(0.05, 4.99)
        #ax.set_yticks([1,2,3,4])
        #ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    for ax in [ax11,ax12,ax13]:
        ax.set_ylim(0.004, 0.99)
    #    ax.set_yticks([0.7,0.8,0.9])
    #    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    for ax in [ax21,ax22,ax23]:
        ax.set_ylim(0.0002, 0.19)
    #    ax.set_yticks([0.7,0.8,0.9])
    #    ax.yaxis.set_minor_locator(MultipleLocator(0.05))


    ax02.tick_params(labelleft=False)
    ax03.tick_params(labelleft=False)
    ax12.tick_params(labelleft=False)
    ax13.tick_params(labelleft=False)
    ax22.tick_params(labelleft=False)
    ax23.tick_params(labelleft=False)

    ax01.tick_params(labelbottom=False)
    ax02.tick_params(labelbottom=False)
    ax03.tick_params(labelbottom=False)
    ax11.tick_params(labelbottom=False)
    ax12.tick_params(labelbottom=False)
    ax13.tick_params(labelbottom=False)

    ax21.set_xlabel(r'\boldmath$x_{\rm bj}$', size=40)
    #ax21.xaxis.set_label_coords(0.95,0.00)
    ax22.set_xlabel(r'\boldmath$x_{\rm bj}$', size=40)
    #ax22.xaxis.set_label_coords(0.95,0.00)
    ax23.set_xlabel(r'\boldmath$x_{\rm bj}$', size=40)

    ax01.text(0.05, 0.40, r'\textrm{\textbf{CLAS12}}', transform = ax01.transAxes, size = 50)
    ax02.text(0.05, 0.20, r'\textrm{\textbf{proton}}', transform = ax02.transAxes, size = 50)
    #ax01.text(0.05, 0.08, r'\boldmath$F_2^{D}/F_2^{p}$', transform = ax01.transAxes, size = 60)
    ax01.text(0.05, 0.22, r'\boldmath$\frac{{\rm d} \sigma}{{\rm d} W {\rm d} Q^2}~[\frac{\rm nb}{{\rm GeV}^3}]$', transform = ax01.transAxes, size = 50)

    ax01.text(0.05, 0.05, r'$Q^2 = 2.8 ~ {\rm GeV}^2$', transform = ax01.transAxes, size = 30)
    ax02.text(0.05, 0.05, r'$Q^2 = 3.2 ~ {\rm GeV}^2$', transform = ax02.transAxes, size = 30)
    ax03.text(0.05, 0.05, r'$Q^2 = 3.8 ~ {\rm GeV}^2$', transform = ax03.transAxes, size = 30)
    ax11.text(0.05, 0.05, r'$Q^2 = 4.4 ~ {\rm GeV}^2$', transform = ax11.transAxes, size = 30)
    ax12.text(0.05, 0.05, r'$Q^2 = 5.2 ~ {\rm GeV}^2$', transform = ax12.transAxes, size = 30)
    ax13.text(0.05, 0.05, r'$Q^2 = 6.1 ~ {\rm GeV}^2$', transform = ax13.transAxes, size = 30)
    ax21.text(0.05, 0.05, r'$Q^2 = 7.1 ~ {\rm GeV}^2$', transform = ax21.transAxes, size = 30)
    ax22.text(0.05, 0.05, r'$Q^2 = 8.3 ~ {\rm GeV}^2$', transform = ax22.transAxes, size = 30)
    ax23.text(0.05, 0.05, r'$Q^2 = 9.7 ~ {\rm GeV}^2$', transform = ax23.transAxes, size = 30)


    handles, labels = [],[]
    handles.append((hand0,thy_band0,thy_plot0))
    handles.append((hand1,thy_band1,thy_plot1))
    labels.append(r'\boldmath$W^2 > 3.5 ~ {\rm GeV}^2$')
    labels.append(r'\boldmath$W^2 < 3.5 ~ {\rm GeV}^2$')
    ax11.legend(handles,labels,loc=(0.00,0.20), fontsize = 30, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    py.tight_layout()
    py.subplots_adjust(hspace=0.01,wspace=0.01,left=0.06,right=0.99,top=0.99)
    filename = '%s/gallery/dis-CLAS12.png'%wdir
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()

def plot_BCDMS(wdir, data):

    if 10069 not in data: return
    if 10070 not in data: return

    nrows, ncols = 1, 1
    py.figure(figsize = (ncols * 8, nrows * 7))
    ax01 = py.subplot(nrows, ncols, 1)

    p   = data[10069] ## BCDMS p
    d   = data[10070] ## BCDMS p

    DATA = {}
    DATA['BCDMS p']  = pd.DataFrame(p)
    DATA['BCDMS d']  = pd.DataFrame(d)

    hand,thy_plot,thy_band = {},{},{}
    #--plot data points
    ax = ax01
    for exp in DATA:
        if exp == 'BCDMS p': shift = 0
        if exp == 'BCDMS d': shift = 0.01
        X     = DATA[exp]['X'] + shift
        val   = DATA[exp]['value']
        alpha = DATA[exp]['alpha']
        color,marker,ms = get_details(exp)
        Q2 = DATA[exp]['Q2']
        hand[exp] = ax.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=2.5,linestyle='none')

        mean = DATA[exp]['thy-0'] 
        std  = DATA[exp]['dthy-0']
        down = mean - std
        up   = mean + std
        thy_plot[exp] ,= ax.plot(X,mean,linestyle='solid',    color=color)
        thy_band[exp]  = ax.fill_between(X,down,up,color=color,alpha=0.5)


    for ax in [ax01]:

        ax.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 30)

        ax.tick_params(axis='both', which = 'major', length = 8)
        ax.tick_params(axis='both', which = 'minor', length = 4)

        ax.set_xlim(0.05, 0.69)
        ax.set_xticks([0.2,0.4,0.6])
        ax.xaxis.set_minor_locator(MultipleLocator(0.10))
        ax.axhline(0,0,1,color='black',ls=':',alpha=0.5)

    for ax in [ax01]:
        ax.set_ylim(-0.19, 0.39)
        ax.set_yticks([-0.1,0,0.1,0.2,0.3])
        ax.set_yticklabels([r'-$0.1$',r'$0$',r'$0.1$',r'$0.2$',r'$0.3$'])
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    ax01.set_xlabel(r'\boldmath$x_{\rm bj}$', size=40)

    ax01.text(0.05, 0.88, r'\textrm{\textbf{BCDMS \boldmath$R$}}', transform = ax01.transAxes, size = 50)

    handles, labels = [],[]
    handles.append((hand['BCDMS p'],thy_band['BCDMS p'],thy_plot['BCDMS p']))
    handles.append((hand['BCDMS d'],thy_band['BCDMS d'],thy_plot['BCDMS d']))
    labels.append(r'\boldmath$p$')
    labels.append(r'\boldmath$D$')
    ax01.legend(handles,labels,loc=(0.00,0.00), fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    py.tight_layout()
    py.subplots_adjust(hspace=0.01,wspace=0.01,left=0.15,right=0.99,top=0.99)
    filename = '%s/gallery/dis-R-BCDMS.png'%wdir
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()

def plot_SLAC(wdir, data):

    if 10064 not in data: return
    if 10065 not in data: return
    if 10066 not in data: return
    if 10067 not in data: return
    if 10068 not in data: return

    nrows, ncols = 2, 2
    py.figure(figsize = (ncols * 6, nrows * 7))
    ax01 = py.subplot(nrows, ncols, 1)
    ax02 = py.subplot(nrows, ncols, 2)
    ax11 = py.subplot(nrows, ncols, 3)
    ax12 = py.subplot(nrows, ncols, 4)

    #################################
    #--Plot F2d/F2h from JLab Hall C
    #################################

    SLAC_p   = data[10064] ## 
    SLAC_d   = data[10065] ## 
    E140_d   = data[10066] ## 
    E140x_p  = data[10067] ## 
    E140x_d  = data[10068] ## 

    DATA = {}
    DATA['SLAC p']  = pd.DataFrame(SLAC_p)
    DATA['SLAC d']  = pd.DataFrame(SLAC_d)
    DATA['E140 d']  = pd.DataFrame(E140_d)
    DATA['E140x p'] = pd.DataFrame(E140x_p)
    DATA['E140x d'] = pd.DataFrame(E140x_d)

    thy_band,thy_plot, hand = {},{},{}
    #--plot data points
    for exp in ['SLAC p', 'SLAC d', 'E140 d']:
        if exp=='SLAC p': ax,shift = ax01, 2
        if exp=='SLAC d': ax,shift = ax02, 2
        if exp=='E140 d': ax,shift = ax11, 1
        bins  = DATA[exp]['bin']
        j = 0
        for key in np.unique(bins):
            X     = DATA[exp]['X'][bins==key]
            val   = DATA[exp]['value'][bins==key] + shift*j
            alpha = DATA[exp]['alpha'][bins==key]
            color,marker,ms = get_details(exp)
            Q2 = DATA[exp]['Q2'][bins==key]
            hand[exp] = ax.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=2.5,linestyle='none')

            mean = DATA[exp]['thy-0'] [bins==key] + shift*j
            std  = DATA[exp]['dthy-0'][bins==key]
            down = mean - std
            up   = mean + std
            thy_plot[exp] ,= ax.plot(Q2,mean,linestyle='solid',    color=color)
            thy_band[exp]  = ax.fill_between(Q2,down,up,color=color,alpha=0.5)

            ax.axhline(shift*j,0,1,alpha=0.5,ls=':',color='black')
            j+=1

    for exp in ['E140x p','E140x d']:
        if exp=='E140x p': ax,shift = ax12,0.0
        if exp=='E140x d': ax,shift = ax12,0.1
        X     = DATA[exp]['X']
        val   = DATA[exp]['value']
        alpha = DATA[exp]['alpha']
        color,marker,ms = get_details(exp)
        Q2 = DATA[exp]['Q2'] + shift
        hand[exp] = ax.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=2.5,linestyle='none')

        mean = DATA[exp]['thy-0']
        std  = DATA[exp]['dthy-0']
        down = mean - std
        up   = mean + std
        thy_plot[exp] ,= ax.plot(Q2,mean,linestyle='solid',    color=color)
        thy_band[exp]  = ax.fill_between(Q2,down,up,color=color,alpha=0.5)

    for ax in [ax01,ax02,ax11,ax12]:
    #    ax.set_xlim(0.21, 0.99)
    #    ax.semilogy()

        ax.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 25)

        ax.tick_params(axis='both', which = 'major', length = 8)
        ax.tick_params(axis='both', which = 'minor', length = 4)

        #ax.set_xticks([0.4,0.6,0.8])
        #ax.xaxis.set_minor_locator(MultipleLocator(0.10))
        ax.set_xlabel(r'\boldmath$Q^2~[{\rm GeV}^2]$', size=25)

    #for ax in [ax01,ax02,ax03]:

    #for ax in [ax11,ax12,ax13]:
    #    ax.set_ylim(0.004, 0.99)
    ##    ax.set_yticks([0.7,0.8,0.9])
    ##    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    #for ax in [ax21,ax22,ax23]:
    #    ax.set_ylim(0.0002, 0.19)
    ##    ax.set_yticks([0.7,0.8,0.9])
    ##    ax.yaxis.set_minor_locator(MultipleLocator(0.05))


    ##ax21.xaxis.set_label_coords(0.95,0.00)
    #ax22.set_xlabel(r'\boldmath$x_{\rm bj}$', size=40)
    ##ax22.xaxis.set_label_coords(0.95,0.00)
    #ax23.set_xlabel(r'\boldmath$x_{\rm bj}$', size=40)

    #ax01.text(0.05, 0.40, r'\textrm{\textbf{CLAS12}}', transform = ax01.transAxes, size = 50)
    #ax02.text(0.05, 0.20, r'\textrm{\textbf{proton}}', transform = ax02.transAxes, size = 50)
    ##ax01.text(0.05, 0.08, r'\boldmath$F_2^{D}/F_2^{p}$', transform = ax01.transAxes, size = 60)
    #ax01.text(0.05, 0.22, r'\boldmath$\frac{{\rm d} \sigma}{{\rm d} W {\rm d} Q^2}~[\frac{\rm nb}{{\rm GeV}^3}]$', transform = ax01.transAxes, size = 50)

    for ax in [ax01,ax02]:
        ax.set_xlim(1, 21)
        ax.set_ylim(-1.0, 21.0)
        #ax.set_yticks([1,2,3,4])
        #ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.text(0.15, 0.05, r'$x=0.1$',   transform = ax.transAxes, size = 20)
        ax.text(0.30, 0.14, r'$x=0.175$', transform = ax.transAxes, size = 20)
        ax.text(0.40, 0.23, r'$x=0.25$',  transform = ax.transAxes, size = 20)
        ax.text(0.50, 0.32, r'$x=0.325$', transform = ax.transAxes, size = 20)
        ax.text(0.60, 0.41, r'$x=0.4$',   transform = ax.transAxes, size = 20)
        ax.text(0.60, 0.50, r'$x=0.475$', transform = ax.transAxes, size = 20)
        ax.text(0.75, 0.59, r'$x=0.55$',  transform = ax.transAxes, size = 20)
        ax.text(0.75, 0.68, r'$x=0.625$', transform = ax.transAxes, size = 20)
        ax.text(0.10, 0.77, r'$x=0.7$',   transform = ax.transAxes, size = 20)
        ax.text(0.20, 0.86, r'$x=0.775$', transform = ax.transAxes, size = 20)
        ax.text(0.70, 0.95, r'$x=0.86$',  transform = ax.transAxes, size = 20)

    ax11.text(0.40, 0.10, r'$x=0.2$',  transform = ax11.transAxes, size = 20)
    ax11.text(0.40, 0.50, r'$x=0.35$', transform = ax11.transAxes, size = 20)
    ax11.text(0.15, 0.90, r'$x=0.5$',  transform = ax11.transAxes, size = 20)

    ax01.text(0.05, 0.90, r'\boldmath$R$',   transform = ax01.transAxes, size = 50)

    ax01.text(0.55, 0.05, r'\textrm{\textbf{SLAC p}}',   transform = ax01.transAxes, size = 40, color = 'firebrick')
    ax02.text(0.55, 0.05, r'\textrm{\textbf{SLAC D}}',   transform = ax02.transAxes, size = 40, color = 'darkgreen')
    ax11.text(0.05, 0.70, r'\textrm{\textbf{E140 D}}',   transform = ax11.transAxes, size = 40, color = 'darkgreen')
    ax12.text(0.30, 0.90, r'\textrm{\textbf{E140x p}}',  transform = ax12.transAxes, size = 40, color = 'firebrick')
    ax12.text(0.30, 0.80, r'\textrm{\textbf{E140x D}}',  transform = ax12.transAxes, size = 40, color = 'darkgreen')

    #handles, labels = [],[]
    #handles.append((hand0,thy_band0,thy_plot0))
    #handles.append((hand1,thy_band1,thy_plot1))
    #labels.append(r'\boldmath$W^2 > 3.5 ~ {\rm GeV}^2$')
    #labels.append(r'\boldmath$W^2 < 3.5 ~ {\rm GeV}^2$')
    #ax11.legend(handles,labels,loc=(0.00,0.20), fontsize = 30, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    py.tight_layout()
    py.subplots_adjust(hspace=0.20,wspace=0.20,left=0.06,right=0.99,top=0.99)
    filename = '%s/gallery/dis-R-SLAC.png'%wdir
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()

def plot_JLab(wdir, data):

    if 10071 not in data: return
    if 10074 not in data: return

    nrows, ncols = 1, 2
    py.figure(figsize = (ncols * 6, nrows * 7))
    ax01 = py.subplot(nrows, ncols, 1)
    ax02 = py.subplot(nrows, ncols, 2)

    #################################
    #--Plot F2d/F2h from JLab Hall C
    #################################

    d   = data[10071] ## 
    p   = data[10074] ## 

    DATA = {}
    DATA['JLab d']  = pd.DataFrame(d)
    DATA['JLab p']  = pd.DataFrame(p)

    thy_band,thy_plot, hand = {},{},{}
    #--plot data points
    for exp in ['JLab d','JLab p']:
        if exp=='JLab d': ax,shift = ax01, 1
        if exp=='JLab p': ax,shift = ax02, 1
        bins  = DATA[exp]['bin']
        j = 0
        for key in np.unique(bins):
            X     = DATA[exp]['X'][bins==key]
            val   = DATA[exp]['value'][bins==key] + shift*j
            alpha = DATA[exp]['alpha'][bins==key]
            color,marker,ms = get_details(exp)
            Q2 = DATA[exp]['Q2'][bins==key]
            hand[exp] = ax.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=2.5,linestyle='none')

            mean = DATA[exp]['thy-0'] [bins==key] + shift*j
            std  = DATA[exp]['dthy-0'][bins==key]
            down = mean - std
            up   = mean + std
            thy_plot[exp] ,= ax.plot(X,mean,linestyle='solid',    color=color)
            thy_band[exp]  = ax.fill_between(X,down,up,color=color,alpha=0.5)

            ax.axhline(shift*j,0,1,alpha=0.5,ls=':',color='black')
            j+=1


    for ax in [ax01,ax02]:
        ax.set_xlim(0.32, 0.65)

        ax.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 25)

        ax.tick_params(axis='both', which = 'major', length = 8)
        ax.tick_params(axis='both', which = 'minor', length = 4)

        ax.set_xticks([0.4,0.5,0.6])
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xlabel(r'\boldmath$x_{\rm bj}$', size=25)

        ax.set_ylim(0.01, 2.50)
        ax.set_yticks([1,2])
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    #for ax in [ax01,ax02,ax03]:

    #for ax in [ax11,ax12,ax13]:

    #for ax in [ax21,ax22,ax23]:
    #    ax.set_ylim(0.0002, 0.19)
    ##    ax.set_yticks([0.7,0.8,0.9])
    ##    ax.yaxis.set_minor_locator(MultipleLocator(0.05))


    ##ax22.xaxis.set_label_coords(0.95,0.00)
    #ax23.set_xlabel(r'\boldmath$x_{\rm bj}$', size=40)

    #ax01.text(0.05, 0.40, r'\textrm{\textbf{CLAS12}}', transform = ax01.transAxes, size = 50)
    #ax02.text(0.05, 0.20, r'\textrm{\textbf{proton}}', transform = ax02.transAxes, size = 50)
    ##ax01.text(0.05, 0.08, r'\boldmath$F_2^{D}/F_2^{p}$', transform = ax01.transAxes, size = 60)
    #ax01.text(0.05, 0.22, r'\boldmath$\frac{{\rm d} \sigma}{{\rm d} W {\rm d} Q^2}~[\frac{\rm nb}{{\rm GeV}^3}]$', transform = ax01.transAxes, size = 50)

    ax01.text(0.40, 0.08, r'$Q^2=2$', transform = ax01.transAxes, size = 20)
    ax01.text(0.70, 0.48, r'$Q^2=3$', transform = ax01.transAxes, size = 20)
    ax01.text(0.60, 0.88, r'$Q^2=4$', transform = ax01.transAxes, size = 20)

    ax02.text(0.28, 0.08, r'$Q^2=1.75$', transform = ax02.transAxes, size = 20)
    ax02.text(0.56, 0.45, r'$Q^2=2.5$',  transform = ax02.transAxes, size = 20)
    ax02.text(0.45, 0.85, r'$Q^2=3.75$', transform = ax02.transAxes, size = 20)

    ax01.text(0.05, 0.90, r'\textrm{\textbf{JLab D}}',   transform = ax01.transAxes, size = 40, color = 'darkgreen')
    ax02.text(0.05, 0.90, r'\textrm{\textbf{JLab p}}',   transform = ax02.transAxes, size = 40, color = 'firebrick')
    #ax11.text(0.05, 0.70, r'\textrm{\textbf{E140 D}}',   transform = ax11.transAxes, size = 40, color = 'darkgreen')
    #ax12.text(0.30, 0.90, r'\textrm{\textbf{E140x p}}',  transform = ax12.transAxes, size = 40, color = 'firebrick')
    #ax12.text(0.30, 0.80, r'\textrm{\textbf{E140x D}}',  transform = ax12.transAxes, size = 40, color = 'darkgreen')

    ax01.text(0.05, 0.50, r'\boldmath$R$',   transform = ax01.transAxes, size = 50)

    ##handles, labels = [],[]
    ##handles.append((hand0,thy_band0,thy_plot0))
    ##handles.append((hand1,thy_band1,thy_plot1))
    ##labels.append(r'\boldmath$W^2 > 3.5 ~ {\rm GeV}^2$')
    ##labels.append(r'\boldmath$W^2 < 3.5 ~ {\rm GeV}^2$')
    ##ax11.legend(handles,labels,loc=(0.00,0.20), fontsize = 30, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    py.tight_layout()
    py.subplots_adjust(hspace=0.20,wspace=0.20,left=0.06,right=0.99,top=0.99)
    filename = '%s/gallery/dis-R-JLab.png'%wdir
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()

def plot_obs(wdir, kc):

    print('\nplotting dis data from %s' % (wdir))

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

    predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
    if 'idis' not in predictions['reactions']:
        print('inclusive DIS is not in data file')
        return

    data = predictions['reactions']['idis']

    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        #predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        if 'r-residuals' in data[idx]: del data[idx]['r-residuals']
        if 'n-residuals' in data[idx]: del data[idx]['n-residuals']
        if 'rres-rep' in data[idx]: del data[idx]['rres-rep']
        for ic in range(kc.nc[istep]):
            predictions_ic = [predictions[i] for i in range(len(predictions))]
            data[idx]['thy-%d' % ic] = np.mean(predictions_ic, axis = 0)
            data[idx]['dthy-%d' % ic] = np.std(predictions_ic, axis = 0)
            if 'X' in data[idx]: data[idx]['x'] = data[idx]['X']
            data[idx]['rQ2'] = np.around(data[idx]['Q2'], decimals = 0)
            data[idx]['rx'] = np.around(data[idx]['x'], decimals = 2)


    plot_JLab (wdir,data)
    plot_BCDMS(wdir,data)
    plot_SLAC (wdir,data)

    return





