#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text',usetex=True)
from matplotlib import cm
import pylab as py

#--for biclustering
from sklearn.datasets import make_biclusters
#from sklearn.datasets import samples_generator as sg
#from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

#--for cluster finding
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn import metrics
#from sklearn.datasets.samples_generator import make_blobs


#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator

#--from local  
from analysis.corelib import core

def _gen_labels_by_chi2(wdir,istep):

    print('\t by chi2 ...')

    #--load data where residuals are computed
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))
    res=np.array(data['res'])
    #--compute chi2/npts
    chi2dof=[]
    for row in res:
        chi2dof.append(np.sum(row**2)/len(row))
    chi2dof=np.array(chi2dof)
    return chi2dof

def _gen_labels_by_echi2(wdir,istep):

    print('\t by echi2 ...')

    #--load data where residuals are computed
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))
    res=np.array(data['res'])
    #--compute chi2/npts
    echi2=[]
    for row in res:
        echi2.append(0)#np.sum(row**2)/len(row))
    echi2=np.array(echi2)

    nexp=0
    for reaction in data['reactions']:
        for idx in data['reactions'][reaction]:
            nexp+=1
            res=np.array(data['reactions'][reaction][idx]['residuals-rep'])
            for i in range(len(res)):
                echi2[i]+=(np.sum(res[i]**2)/len(res[i]))
    return echi2/nexp

def hook(params,order):
    sample=[]
    for i in range(len(order)):
        if order[i][0]!=1: continue
        #if order[i][1]!='pdf': continue
        #if 'g1'  in order[i][2]: continue 
        #if 'uv1' in order[i][2]: continue 
        #if 'dv1' in order[i][2]: continue 
        #if 'db1' in order[i][2]: continue 
        #if 'ub1' in order[i][2]: continue 
        #if 's1' in order[i][2]: continue 
        #if 'sb1' in order[i][2]: continue 
        sample.append(params[i])
    return sample

def _gen_labels_by_cluster(wdir,istep,nc,hook=None):

    print('\t by kmeans ...')

    replicas=core.get_replicas(wdir)
    samples=[]
    for replica in replicas:
        params = replica['params'][istep]
        order  = replica['order'][istep]

        if hook==None:
            samples.append(params)
        else:
            samples.append(hook(params,order))

    #--affinity propagation
    #af = AffinityPropagation(preference=-50).fit(samples)
    #cluster_centers_indices = af.cluster_centers_indices_
    #return af.labels_

    #--agglomerative
    #clustering = AgglomerativeClustering(nc).fit(samples)
    #return clustering.labels_

    #brc = Birch(branching_factor=3, n_clusters=None, threshold=0.5,compute_labels=True)
    #brc.fit(samples) 
    #return brc.labels_

    #--kmean   
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(samples)
    return kmeans.labels_
   
def gen_labels(wdir,kc):

    print('\ngenerating labels for %s\n'%wdir)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if istep not in kc.nc:
        print("istep not in kmeanconf.py")
        sys.exit()

    nc=kc.nc[istep]
    hook=kc.hooks[istep]

    istep=core.get_istep()
    labels={}
    labels['chi2dof']  = _gen_labels_by_chi2(wdir,istep)
    labels['echi2dof'] = _gen_labels_by_echi2(wdir,istep)
    labels['cluster']  = _gen_labels_by_cluster(wdir,istep,nc,hook)
    checkdir('%s/data'%wdir)
    save(labels,'%s/data/labels-%d.dat'%(wdir,istep))

def get_clusters(wdir,istep,kc): 

    nc=kc.nc[istep]
    hook=kc.hooks[istep]

    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster  = labels['cluster']
    chi2dof  = labels['chi2dof']
    echi2dof = labels['echi2dof']

    #--get clusters idx ordering
    echi2_means = [np.mean([echi2dof[i] for i in range(len(echi2dof))
           if cluster[i]==j ])  for j in range(nc)]

    order=np.argsort(echi2_means)

    clist=['r','c','g','y']
    colors={order[i]:clist[i] for i in range(nc)}
    return cluster,colors,nc,order

#--plots

def plot_chi2_dist(wdir,istep,kc):

    #--needs revisions NS (09/02/19)

    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    chi2     = labels['chi2dof']
    cluster,colors,nc,order = get_clusters(wdir,istep,kc) 

    nrows=1
    ncols=1
    #--plot labeled residuals
    ax=py.subplot(nrows,ncols,1)

    chi2=[_ for _ in chi2 if _<2]
    Mean=np.average(chi2)
    dchi=np.std(chi2)
    chi2min=Mean-2*dchi
    chi2max=Mean+2*dchi
    R=(chi2min,chi2max)
    ax.hist(chi2,bins=60,range=R,histtype='step',label='size=%d'%len(chi2))
    for j in range(nc):
        chi2_=[chi2[i] for i in range(len(chi2)) if cluster[i]==j]
        c=colors[j]
        print (c,np.mean(chi2_),np.mean(chi2))
        label='cluster id=%d  size=%d'%(j,len(chi2_))
        ax.hist(chi2_,bins=30,range=R,histtype='step',label=label,color=c)
        mean=np.average(chi2_)
        ax.axvline(mean,color=c,ls=':')
        

    ax.set_xlabel('chi2/npts')
    ax.set_ylabel('yiled')
    ax.legend()
    ax.text(0.1,0.8,'step %d'%istep,transform=ax.transAxes)
    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/chi2-dist-%d.jpg'%(wdir,istep))
    py.close()

def plot_chi2_dist_per_exp_backup(wdir,kc,reaction,idx):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster,colors,nc,order = get_clusters(wdir,istep,kc) 
    chi2     = labels['chi2dof']
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))

    if reaction not in data['reactions']: 
        print('%s not in step'%reaction)
        return
    if idx not in data['reactions'][reaction]: 
        print('%s %s not in step'%(reaction,idx))
        return

    res=np.array(data['reactions'][reaction][idx]['residuals-rep'])
    chi2=np.array([np.sum(r**2)/len(r) for r in res])
    #chi2=np.array([_ for _ in chi2 if _<5])

    Mean=np.average(chi2)
    
    dchi=np.std(chi2)
    chi2min=Mean-2*dchi
    chi2max=Mean+2*dchi
    R=(chi2min,chi2max)
    nrows=1
    ncols=1
    #--plot labeled residuals
    fig = py.figure(figsize=(nrows*7,ncols*4))
    ax=py.subplot(nrows,ncols,1)

    ax.hist(chi2,bins=30,range=R,histtype='step',label='size=%d'%len(chi2))
    for j in range(nc):
        _chi2=[chi2[i] for i in range(len(chi2)) if cluster[i]==j]
        c=colors[j]
        print (c, np.mean(_chi2),reaction,idx)
        label='cluster id=%d  size=%d'%(j,len(_chi2))
        ax.hist(_chi2,bins=30,range=R,histtype='step',label=label,color=c)
    ax.set_xlabel(r'\textrm{\textbf{chi2/npts}}',size=25)
    ax.set_ylabel(r'\textrm{\textbf{yield}}',size=25)
    ax.legend()
    ax.text(0.1,0.8,'step %d'%istep,transform=ax.transAxes)
    ax.tick_params(axis='both',which='both',direction='in',right=True,top=True,labelsize=15)
    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/chi2-dist-%d-%s-%d.png'%(wdir,istep,reaction,idx))
    py.close()

def get_norm(wdir):
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    tab={}
    j = 0
    for replica in replicas: 
        j+=1
        order=replica['order'][istep]
        params=replica['params'][istep]
        for i in range(len(order)):
            if order[i][0]==2:
                reaction=order[i][1]
                idx=order[i][2]
                if reaction not in tab: tab[reaction]={}
                if idx not in tab[reaction]: tab[reaction][idx]=[]
                tab[reaction][idx].append(params[i])

        #for k in conf['datasets']:
        #    for kk in conf['datasets'][k]['norm']:
        #        if conf['datasets'][k]['norm'][kk]['fixed'] == True:  continue
        #        if conf['datasets'][k]['norm'][kk]['fixed'] == False: continue
        #        reference_norm = conf['datasets'][k]['norm'][kk]['fixed']
        #        if k  not in tab: tab[k]={}
        #        if kk not in tab[k]: tab[k][kk]=[]
        #        tab[k][kk].append(tab[k][reference_norm])
                       
    for reaction in tab:
        if reaction not in conf['datasets']: continue
        for idx in tab[reaction]:
            norm=tab[reaction][idx][:]
            if idx not in conf['datasets'][reaction]['norm']: continue
            tab[reaction][idx]={}
            tab[reaction][idx]['norm'] = np.array(norm)
            tab[reaction][idx]['mean']=np.mean(norm)
            tab[reaction][idx]['std']=np.std(norm)
            tab[reaction][idx]['fixed'] = conf['datasets'][reaction]['norm'][idx]['fixed']
    return tab

def plot_chi2_dist_per_exp(wdir):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))
    norm_tab = get_norm(wdir)

    for reaction in data['reactions']:
        for idx in data['reactions'][reaction]:
            temp = data['reactions'][reaction][idx]
            #print(list(temp))
            #--compute chi2/npts
            res  = np.array(data['reactions'][reaction][idx]['residuals-rep'])
            rres = np.array(data['reactions'][reaction][idx]['rres-rep'])
            #nres = np.array(data['reactions'][reaction][idx]['n-residuals'])
            npts = len(data['reactions'][reaction][idx]['value'])
            chi2 = np.sum(res**2,axis=1)/npts

            mean = np.mean(chi2)
            std  = np.std (chi2)
            
            chi2min=mean-std
            chi2max=mean+std
            R=(chi2min,chi2max)
            nrows=1
            ncols=1
            #--plot labeled residuals
            fig = py.figure(figsize=(nrows*7,ncols*4))
            ax11=py.subplot(nrows,ncols,1)

            c='red'
            ax11.hist(chi2,bins=30,histtype='step',color=c)
            #ax11.axvspan(mean-std,mean+std,alpha=0.2,color=c)
            ax11.set_xlabel(r'\boldmath$\chi^2_{\rm red}$',size=25)
            ax11.set_ylabel(r'\textrm{\textbf{yield}}',size=25)
            ax11.text(0.05,0.85,r'\textrm{\textbf{%s %s}'%(reaction,idx),transform=ax11.transAxes,size=30)
            ax11.tick_params(axis='both',which='both',direction='in',right=True,top=True,labelsize=15)

            py.tight_layout()
            checkdir('%s/gallery/chi2'%wdir)
            filename = '%s/gallery/chi2/chi2-dist-%s-%s.png'%(wdir,reaction,idx)
            py.savefig(filename)
            print('Saving chi2 figure to %s'%filename)
            py.close()

            #--plot rchi2 as well
            if rres.shape[1]==0: pass
            else:
                rchi2 = rres**2
                rchi2_tot = np.sum(rchi2,axis=1)
                nerr = rres.shape[1]
                fig = py.figure(figsize=(nrows*18,ncols*20))
                ncols = 4
                nrows = int(np.ceil(nerr/ncols)) + 1
                #ncols=nerr
                for i in range(nerr):
                    ax=py.subplot(nrows,ncols,i+1)

                    c='red'
                    ax.hist(rchi2[:,i],bins=30,histtype='step',color=c)
                    #ax11.axvspan(mean-std,mean+std,alpha=0.2,color=c)
                    ax.set_xlabel(r'\boldmath$\chi^2_{\rm corr}$',size=25)
                    if i==0: ax.set_ylabel(r'\textrm{\textbf{yield}}',size=25)
                    if i==0: ax.text(0.05,0.85,r'\textrm{\textbf{%s %s}'%(reaction,idx),transform=ax.transAxes,size=30)
                    ax.text(0.85,0.85,r'\textrm{\textbf{%d}}'%i,transform=ax.transAxes,size=30)
                    #ax.set_ylabel(r'\textrm{\textbf{yield}}',size=25)
                    ax.tick_params(axis='both',which='both',direction='in',right=True,top=True,labelsize=15)

                #--plot total
                ax = py.subplot(nrows,ncols,nerr+1)
                ax.hist(rchi2_tot,bins=30,histtype='step',color=c)
                #ax11.axvspan(mean-std,mean+std,alpha=0.2,color=c)
                ax.set_xlabel(r'\boldmath$\chi^2_{\rm corr}$',size=25)
                ax.text(0.75,0.85,r'\textrm{\textbf{total}}',transform=ax.transAxes,size=30)
                

                py.tight_layout()
                py.subplots_adjust(left=0.02,right=0.99,top=0.99,hspace=0.12,wspace=0.03)
                checkdir('%s/gallery/chi2'%wdir)
                filename = '%s/gallery/chi2/rchi2-dist-%s-%s.png'%(wdir,reaction,idx)
                py.savefig(filename)
                print('Saving rchi2 figure to %s'%filename)
                py.close()

            #--plot nchi2 as well
            value = data['reactions'][reaction][idx]['value']
            _norm  = [_ for _ in data['reactions'][reaction][idx] if '_c' in _ and 'norm' in _ and '%' not in _]


            if   len(_norm)==0: pass
            else:
                for i in range(len(value)):
                    if value[i]!=0:
                        dN=data['reactions'][reaction][idx][_norm[0]][i]/value[i]
                        break
                if idx not in norm_tab[reaction]:
                    pass
                else:
                    norm = norm_tab[reaction][idx]['norm']
                    nres = (1-norm)/dN
                    nchi2 = nres**2
                nrows=1
                ncols=1
                #--plot labeled residuals
                fig = py.figure(figsize=(nrows*7,ncols*4))
                ax11=py.subplot(nrows,ncols,1)

                c='red'
                ax11.hist(nchi2,bins=30,histtype='step',color=c)
                #ax11.axvspan(mean-std,mean+std,alpha=0.2,color=c)
                ax11.set_xlabel(r'\boldmath$\chi^2_{\rm norm}$',size=25)
                ax11.set_ylabel(r'\textrm{\textbf{yield}}',size=25)
                ax11.text(0.05,0.85,r'\textrm{\textbf{%s %s}'%(reaction,idx),transform=ax11.transAxes,size=30)
                ax11.tick_params(axis='both',which='both',direction='in',right=True,top=True,labelsize=15)

                py.tight_layout()
                checkdir('%s/gallery/chi2'%wdir)
                filename = '%s/gallery/chi2/nchi2-dist-%s-%s.png'%(wdir,reaction,idx)
                py.savefig(filename)
                print('Saving nchi2 figure to %s'%filename)
                py.close()


def print_chi2_per_exp(wdir,kc):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster,colors,nc,order = get_clusters(wdir,istep,kc) 
    chi2     = labels['chi2dof']
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))

    npts     = 0
    res2tot  = {colors[i]:0 for i in range(len(colors))}
    chi2tot  = {colors[i]:0 for i in range(len(colors))}
    echi2tot = {colors[i]:0 for i in range(len(colors))}
    for reaction in data['reactions']:
        print
        print ('reaction: %s'%reaction)
        for idx in data['reactions'][reaction]:
            print('idx: %s'%idx)
                       
            res=np.array(data['reactions'][reaction][idx]['residuals-rep'])
            chi2=np.array([np.sum(r**2)/len(r) for r in res])

            res2=np.array([np.sum(r**2) for r in res])

            npts += len(res[1])

            for j in range(nc):
                _chi2=[chi2[i] for i in range(len(chi2)) if cluster[i]==j]
                _res2=[res2[i] for i in range(len(res2)) if cluster[i]==j]
                c=colors[j]
                print('color: %s, chi2: %3.2f'%(c,np.mean(_chi2)))
                echi2tot[c] += np.mean(_chi2)
                res2tot[c]  += np.mean(_res2)

    for j in range(nc):
        chi2tot[colors[j]] = res2tot[colors[j]]/npts

    print('Total chi2:')
    print(chi2tot)
    print('Total echi2:')
    print(echi2tot)

#--get scale for color coding replicas
def get_scale(wdir,reaction='all',idx=[]):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas = core.get_replicas(wdir)
    nrep = len(replicas)
    ones = np.ones(nrep)*0.4

    predictions = load('%s/data/predictions-%s.dat'%(wdir,istep))

    rchi2 = []
    if reaction=='all':
       npts,chi2 = 0,0
       for reaction in predictions['reactions']:
           data = predictions['reactions'][reaction]
           for _ in data:
               thy   = np.mean(data[_]['prediction-rep'],axis=0)
               res = (data[_]['value']-thy)/data[_]['alpha']
               npts += res.size
               chi2 += np.sum(res**2)
       rchi2.append(chi2/npts)

    else:

        if reaction not in predictions['reactions']: return ones
        data = predictions['reactions'][reaction]

        for _ in idx:
            if _ not in list(data): return ones

        for i in range(len(data[_]['prediction-rep'])):
            npts,chi2 = 0,0
            for _ in idx:
                if _ not in data: continue
                thy = data[_]['prediction-rep'][i]
                res = (data[_]['value']-thy)/data[_]['alpha']
                npts += res.size
                chi2 += np.sum(res**2)
            rchi2.append(chi2/npts)

    amin = np.amin(rchi2)
    amax = np.amax(rchi2)
    scale = np.interp(rchi2, (amin,amax), (0,1))

    return scale







