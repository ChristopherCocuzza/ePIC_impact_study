import sys,os
import numpy as np
import scipy.stats
import copy
from subprocess import Popen, PIPE, STDOUT
import pandas as pd

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pylab as py

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.corelib import core
from analysis.corelib import classifier



def get_lqcd_itd_chi2(wdir):

    replicas=core.get_replicas(wdir)
    load_config(wdir + '/input.py')
    istep=core.get_istep()
    core.mod_conf(istep,replicas[0])
    conf['predict'] = True
    conf['bootstrap']=False
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    order=parman.order
  
    residuals = resman.lqcd_itd_res

    value = residuals.data['value']
    cov   = residuals.cov
    dim   = residuals.dim

    inv_cov=np.linalg.inv(cov)
    eigenvalues, eigenvectors = np.linalg.eig(inv_cov)
    sqrt_eigenvalues=np.sqrt(eigenvalues)
    II=np.argsort(sqrt_eigenvalues)
    diag=np.diag(eigenvalues)

    nrep = len(replicas) 
    cnt =0
    R=[]
    res = []
    for replica in replicas:
        #lprint('Processing: [%s/%s]'%(cnt+1,nrep))
        parman.set_new_params(replica['params'][istep],initial=True)
        res_lattice,exp,thy = residuals.get_residuals()
        res.append(res_lattice)
        r=value[0]-thy
        r=np.einsum('ij,j',eigenvectors.T,r)
        r*=sqrt_eigenvalues
        R.append(r)
        cnt +=1

    res = np.array(res)

    chi2 = np.mean(np.sum(res**2,axis=1))

    R=np.abs(np.mean(R,axis=0)[II])

    return chi2, dim

def get_z_score(chi2_red,dof):
    x = chi2_red
    cdf = scipy.stats.chi2.cdf(x,dof)
    p_val = 1-cdf
    #return np.abs(scipy.stats.norm.ppf(p_val))
    return -scipy.stats.norm.ppf(p_val)

def get_norm(wdir):
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    tab={}
    for replica in replicas: 
        order=replica['order'][istep]
        params=replica['params'][istep]
        for i in range(len(order)):
            if order[i][0]==2:
                reaction=order[i][1]
                idx=order[i][2]
                if reaction not in tab: tab[reaction]={}
                if idx not in tab[reaction]: tab[reaction][idx]=[] 
                tab[reaction][idx].append(params[i])

        for k in conf['datasets']:
            for kk in conf['datasets'][k]['norm']:
                if conf['datasets'][k]['norm'][kk]['fixed'] == True:  continue
                if conf['datasets'][k]['norm'][kk]['fixed'] == False: continue
                reference_norm = conf['datasets'][k]['norm'][kk]['fixed']
                if k  not in tab: tab[k]={}
                if kk not in tab[k]: tab[k][kk]=[] 
                tab[k][kk].append(tab[k][reference_norm])
                       
    for reaction in tab:
        if reaction not in conf['datasets']: continue
        for idx in tab[reaction]:
            norm=tab[reaction][idx][:]
            if idx not in conf['datasets'][reaction]['norm']: continue
            tab[reaction][idx]={}
            tab[reaction][idx]['mean']=np.mean(norm)
            tab[reaction][idx]['std']=np.std(norm)
            tab[reaction][idx]['fixed'] = conf['datasets'][reaction]['norm'][idx]['fixed']
    return tab

def get_chi2(wdir,norm_tab,REACTION=None,COL=None,OBS=None,TAR=None,HAD=None):

    istep=core.get_istep()

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    data=predictions['reactions']
    tab={}
    for reaction in data:
        if REACTION!=None and reaction!=REACTION: continue
        if len(list(data[reaction])) == 0: continue
        if reaction not in tab: 
            tab[reaction]={}
            tab[reaction]['chi2']  = 0
            tab[reaction]['rchi2'] = 0
            tab[reaction]['nchi2'] = 0
            tab[reaction]['npts']  = 0
            tab[reaction]['pull']  = 0
        for idx in data[reaction]:

            if 'resampling' in data[reaction][idx]: resampling = data[reaction][idx]['resampling'][0]
            else:                                   resampling = 0

            if idx not in tab[reaction]: tab[reaction][idx]={}
            value=data[reaction][idx]['value']
            alpha=data[reaction][idx]['alpha']
            if 'rres-rep' in data[reaction][idx].keys():
                rres=np.mean(data[reaction][idx]['rres-rep'],axis=0)
            else:
                rres=0.
            if np.isnan(rres).any(): rres=0.0


            thy=np.mean(data[reaction][idx]['prediction-rep'],axis=0)
            tab['nrep'] = len(data[reaction][idx]['prediction-rep'])
            col=data[reaction][idx]['col'][0]
            if col=='compass': col='COMPASS'
            if col=='hermes':  col='HERMES'
            if col=='jlab':    col='JLab'
            if col=='belle':   col='Belle'
            if 'target' in            data[reaction][idx]: tar=data[reaction][idx]['target'][0]
            elif 'tar' in             data[reaction][idx]: tar=data[reaction][idx]['tar'][0]
            elif 'particles-in' in    data[reaction][idx]: tar=data[reaction][idx]['particles-in'][0]
            elif 'reaction' in        data[reaction][idx]: tar=data[reaction][idx]['reaction'][0]
            else:                                          tar='-' 
            if tar=='proton':   tar='p'
            if tar=='neutron':  tar='n'
            if tar=='deuteron' or tar=='d': tar='D'
            if   'hadron'  in            data[reaction][idx]: had=data[reaction][idx]['hadron'][0]
            elif 'hadron1' in            data[reaction][idx]: had=data[reaction][idx]['hadron1'][0] + ',' + data[reaction][idx]['hadron2'][0]
            else:                                          had='-'
            obs=data[reaction][idx]['obs'][0]

            norm  = [_ for _ in data[reaction][idx] if '_c' in _ and 'norm' in _ and '%' not in _]

            if   len(norm)==0:
                nres = 0.0
            elif len(norm)==1:
                for i in range(len(value)):
                    if value[i]!=0:
                        dN=data[reaction][idx][norm[0]][i]/value[i]
                        break
                if idx not in norm_tab[reaction]:
                    nres = 0.0
                else:
                    norm = norm_tab[reaction][idx]['mean']
                    nres = (1-norm)/dN

            
            #--check if norm is fixed to another dataset
            fixed_norm=False
            if idx in conf['datasets'][reaction]['norm']:
                fixed = conf['datasets'][reaction]['norm'][idx]['fixed']
                if fixed not in [True,False]:
                    fixed_norm = fixed

            tab[reaction][idx]['col']       = col
            tab[reaction][idx]['tar']       = tar
            tab[reaction][idx]['had']       = had
            tab[reaction][idx]['obs']       = obs

            if COL!=None and col!=COL: continue
            if OBS!=None and obs!=OBS: continue
            if TAR!=None and tar!=TAR: continue
            if HAD!=None and had!=HAD: continue

            if resampling==0:
                res=(value-thy)/alpha
            if resampling==1:
                res=np.mean(data[reaction][idx]['residuals-rep'],axis=0)

            chi2=np.sum(res**2)# + np.sum(rres**2)
            npts=res.size
            chi2_npts=chi2/npts

            #--get chi2 per replica
            thy_rep=data[reaction][idx]['prediction-rep']
            nrep = len(thy_rep)
            chi2_rep = np.zeros(nrep)
            for i in range(nrep):
                if resampling==0:
                    res=(value-thy_rep[i])/alpha
                if resampling==1:
                    res=data[reaction][idx]['residuals'][i] #--this may be wrong
                
                npts=res.size
                chi2_rep[i]=np.sum(res**2)/npts

            #--take 68% CI of chi2_rep
            p = 16
            chi2_rep_min = np.percentile(chi2_rep,p    ,axis=0)
            chi2_rep_max = np.percentile(chi2_rep,100-p,axis=0)
            chi2_rep_mid = 0.5*(chi2_rep_max + chi2_rep_min) 
            chi2_rep_err = 0.5*(chi2_rep_max - chi2_rep_min) 

            try:
                if len(rres)==0:
                    chi2_corr = '-'
                else:
                    chi2_corr = np.sum(rres**2)
            except:
                if rres==0.0:
                    chi2_corr='-'
                else:
                    chi2_corr = np.sum(rres**2)

            if nres==0:
                chi2_norm = '-'
            elif fixed_norm!=False:
                chi2_norm = '-'
            else:
                chi2_norm = np.sum(nres**2)

            zscore = get_z_score(chi2,npts)

            dthy=np.std(data[reaction][idx]['prediction-rep'],axis=0)
            pull=np.abs(value-thy)/np.sqrt(alpha**2 + dthy**2)
            pull=np.sum(pull)
            pull_npts = pull/npts

            tab[reaction][idx]['chi2']      = chi2
            tab[reaction][idx]['chi2_rep_mid']      = chi2_rep_mid
            tab[reaction][idx]['chi2_rep_err']      = chi2_rep_err
            tab[reaction][idx]['pull']      = pull
            tab[reaction][idx]['npts']      = npts
            tab[reaction][idx]['chi2_npts'] = chi2_npts
            tab[reaction][idx]['chi2_corr'] = chi2_corr
            tab[reaction][idx]['chi2_norm'] = chi2_norm
            tab[reaction][idx]['pull_npts'] = pull_npts
            tab[reaction][idx]['zscore']    = zscore
            tab[reaction]['chi2']          += chi2
            if chi2_corr != '-': tab[reaction]['rchi2'] += chi2_corr
            if chi2_norm != '-' and fixed_norm==False: tab[reaction]['nchi2'] += chi2_norm
            tab[reaction]['npts']          += npts
            tab[reaction]['pull']          += pull
        tab[reaction]['chi2_npts'] = tab[reaction]['chi2']/tab[reaction]['npts']
        tab[reaction]['pull_npts'] = tab[reaction]['pull']/tab[reaction]['npts']
        tab[reaction]['zscore']    = get_z_score(tab[reaction]['chi2'],tab[reaction]['npts'])

    return tab

def print_summary_backup(wdir,REACTION=None,COL=None,OBS=None,TAR=None,HAD=None,legend=True):


    load_config('%s/input.py'%wdir)
    norm_tab=get_norm(wdir)
    chi2_tab=get_chi2(wdir,norm_tab,REACTION=REACTION,COL=COL,OBS=OBS,TAR=TAR,HAD=None)

    L = len(os.listdir('%s/msr-inspected'%wdir))

    print('\nsummary of  %s [%d/%d replicas]\n'%(wdir,chi2_tab['nrep'],L))

    #--adjust width of col and obs columns
    lobs,lcol = [],[]
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        if REACTION!=None and reaction!=REACTION: continue
        for idx in chi2_tab[reaction]:
            if idx in ['chi2','nchi2','rchi2','npts','chi2_npts','zscore','pull_npts','pull']: continue
            lcol.append(len(chi2_tab[reaction][idx]['col']))
            lobs.append(len(chi2_tab[reaction][idx]['obs']))

    lcol = np.max(lcol)
    lobs = np.max(lobs)

    msg1 ='\033[1m%10s '
    msg1+=' '*(lcol-2)
    msg1+='%s '
    msg1+='%5s '
    msg1+='%8s '
    msg1+=' '*(lobs-2)
    msg1+='%s '
    msg1+='%5s '
    msg1+='%10s '
    msg1+='%10s '
    msg1+='%12s '
    msg1+='%10s '
    msg1+='%10s '
    msg1+='%10s \033[0m'
    msg1+='%10s '
    msg1=msg1%('idx','col','tar','had(s)','obs','npts','chi2','chi2red','chi2corr','norm','chi2norm','tot chi2','zscore')
    #print(msg1)
    #print("-"*len(msg1))
    chi2_tot = 0
    rchi2_tot= 0
    nchi2_tot= 0
    pull_tot = 0
    npts_tot = 0
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        print("-"*len(msg1))
        print('\033[1mREACTION: %s\033[0m'%reaction)
        print(msg1)
        print("-"*len(msg1))
        if REACTION!=None and reaction!=REACTION: continue
        l = list(chi2_tab[reaction])
        for _ in ['chi2','rchi2','nchi2','npts','chi2_npts','zscore','pull_npts','pull']: l.remove(_)
        for idx in sorted(l):
            col=chi2_tab[reaction][idx]['col']
            tar=chi2_tab[reaction][idx]['tar']
            had=chi2_tab[reaction][idx]['had']
            obs=chi2_tab[reaction][idx]['obs']
            if COL!=None and col!=COL: continue
            if OBS!=None and obs!=OBS: continue
            if TAR!=None and tar!=TAR: continue
            if HAD!=None and had!=HAD: continue
            npts=chi2_tab[reaction][idx]['npts']
            chi2=chi2_tab[reaction][idx]['chi2']
            zscore=chi2_tab[reaction][idx]['zscore']
            pull=chi2_tab[reaction][idx]['pull']
            chi2_npts=chi2_tab[reaction][idx]['chi2_npts']
            pull_npts=chi2_tab[reaction][idx]['pull_npts']
            chi2_corr=chi2_tab[reaction][idx]['chi2_corr']
            chi2_norm=chi2_tab[reaction][idx]['chi2_norm']

            chi2_tot_idx = chi2
            if chi2_corr != '-': 
                rchi2_tot += chi2_corr
                chi2_tot_idx += chi2_corr
            if reaction in norm_tab:
                if idx in norm_tab[reaction]:
                    fixed = norm_tab[reaction][idx]['fixed']
                    if fixed==False:
                        mean=norm_tab[reaction][idx]['mean']
                        std =norm_tab[reaction][idx]['std']
                        std =str(int(np.round(std*1000)))
                        norm = '%4.3f'%mean
                        if len(std)==1: norm+=' '
                        norm += '('+std+')'
                        nchi2_tot += chi2_norm
                        chi2_tot_idx += chi2_norm
                    else:
                        norm = '[%s]'%fixed
                else:
                    norm='-'
            else: norm = '-'

            pull_tot+=pull
            chi2_tot+=chi2
            npts_tot+=npts

            if   chi2_npts < 1: color = '32' #--green
            elif chi2_npts < 2: color = '97' #--white
            elif chi2_npts < 3: color = '33' #--yellow/orange
            elif chi2_npts < 5: color = '91' #--bright red
            else:               color = '31' #--red

            if   np.abs(zscore) < 1: zcolor = '32' #--green
            elif np.abs(zscore) < 2: zcolor = '97' #--white
            elif np.abs(zscore) < 3: zcolor = '33' #--yellow/orange
            elif np.abs(zscore) < 5: zcolor = '91' #--bright red
            else:                    zcolor = '31' #--red

            if chi2_norm=='-':
                ncolor = ''
            else:
                if   chi2_norm < 1:  ncolor = '32' #--green
                elif chi2_norm < 4:  ncolor = '97' #--white
                elif chi2_norm < 9:  ncolor = '33' #--yellow/orange
                elif chi2_norm < 25: ncolor = '91' #--bright red
                else:                ncolor = '31' #--red


            msg2 ='%10d '
            msg2+=' '*(lcol-len(col)+1)
            msg2+='%s '
            msg2+='%5s '
            msg2+='%8s '
            msg2+=' '*(lobs-len(obs)+1)
            msg2+='%s '
            msg2+='%5d '
            msg2+='%11.2f '
            msg2+='\033[%sm%10.2f\033[0m'
            if chi2_corr=='-': msg2+='%12s '
            else:              msg2+='%12.2f '
            if norm=='-': msg2 += '%10s ' 
            else:         msg2 += '%10s ' 
            if chi2_norm=='-': msg2+='%s%9s '
            else:              msg2+='\033[%sm%10.2f\033[0m'
            msg2+='%11.2f '
            msg2+='\033[%sm%10.2f\033[0m'
            print(msg2%(idx,col,tar,had,obs,npts,chi2,color,chi2_npts,chi2_corr,norm,ncolor,chi2_norm,chi2_tot_idx,zcolor,zscore))

    chi2_npts_tot=chi2_tot/npts_tot
    pull_npts_tot=pull_tot/npts_tot

    zscore_tot = get_z_score(chi2_tot,npts_tot)

    print("-"*len(msg1))
    print('\033[1mPROCESS TOTALS:\033[0m')

    msg3='\033[1m%14s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s \033[0m'
    msg3=msg3%('reaction','npts','chi2','chi2red','chi2corr','chi2norm','tot chi2','zscore')
    print(msg3)
    print("-"*len(msg1))


    #--print chi2 per experiment
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        if REACTION!=None and reaction!=REACTION: continue
        npts      = chi2_tab[reaction]['npts']
        chi2      = chi2_tab[reaction]['chi2']
        rchi2     = chi2_tab[reaction]['rchi2']
        nchi2     = chi2_tab[reaction]['nchi2']
        zscore    = chi2_tab[reaction]['zscore']
        chi2_npts = chi2_tab[reaction]['chi2_npts']
        pull_npts = chi2_tab[reaction]['pull_npts']
        if   chi2_npts < 1: color = '32' #--green
        elif chi2_npts < 2: color = '97' #--white
        elif chi2_npts < 3: color = '33' #--yellow/orange
        elif chi2_npts < 5: color = '91' #--bright red
        else:               color = '31' #--red

        if   np.abs(zscore) < 1: zcolor = '32' #--green
        elif np.abs(zscore) < 2: zcolor = '97' #--white
        elif np.abs(zscore) < 3: zcolor = '33' #--yellow/orange
        elif np.abs(zscore) < 5: zcolor = '91' #--bright red
        else:                    zcolor = '31' #--red

        chi2_summed = chi2 + rchi2 + nchi2
        msg4='%14s '
        msg4+='%10s '
        msg4+='%10.2f '
        msg4+='\033[%sm%10.2f\033[0m'
        msg4+='%11.2f '
        msg4+='%10.2f '
        msg4+='%10.2f '
        msg4+='\033[%sm%10.2f\033[0m'
        print(msg4%(reaction,npts,chi2,color,chi2_npts,rchi2,nchi2,chi2_summed,zcolor,zscore))

    if 'lqcd_itd' in conf and conf['lqcd_itd']:
        chi2,npts = get_lqcd_itd_chi2(wdir)
        reaction = 'lqcd_itd'
        chi2_tot += chi2
        npts_tot += npts
        chi2_npts = chi2/npts
        zscore = get_z_score(chi2,npts)
        msg ='%14s '
        msg+='%10s '
        msg+='%10.2f '
        msg+='\033[%sm%10.2f\033[0m'
        msg+='\033[%sm%10.2f\033[0m'
        if   chi2_npts < 1: color = '32' #--green
        elif chi2_npts < 2: color = '97' #--white
        elif chi2_npts < 3: color = '33' #--yellow/orange
        elif chi2_npts < 5: color = '91' #--bright red
        else:               color = '31' #--red

        if   np.abs(zscore) < 1: zcolor = '32' #--green
        elif np.abs(zscore) < 2: zcolor = '97' #--white
        elif np.abs(zscore) < 3: zcolor = '33' #--yellow/orange
        elif np.abs(zscore) < 5: zcolor = '91' #--bright red
        else:                    zcolor = '31' #--red
        print(msg%(reaction,npts,chi2,color,chi2_npts,zcolor,zscore))

    if   chi2_npts_tot < 1: color = '32' #--green
    elif chi2_npts_tot < 2: color = '97' #--white
    elif chi2_npts_tot < 3: color = '33' #--yellow/orange
    elif chi2_npts_tot < 5: color = '91' #--bright red
    else:                   color = '31' #--red
    
    if   np.abs(zscore_tot) < 1: zcolor = '32' #--green
    elif np.abs(zscore_tot) < 2: zcolor = '97' #--white
    elif np.abs(zscore_tot) < 3: zcolor = '33' #--yellow/orange
    elif np.abs(zscore_tot) < 5: zcolor = '91' #--bright red
    else:                    zcolor = '31' #--red

    chi2_summed = chi2_tot + rchi2_tot + nchi2_tot

    print("-"*len(msg1))
    msg5='\033[1m%14s \033[0m'
    msg5+='%10s '
    msg5+='%10.2f '
    msg5+='\033[%sm%10.2f\033[0m'
    msg5+='%11.2f '
    msg5+='%10.2f '
    msg5+='%10.2f '
    msg5+='\033[%sm%10.2f\033[0m'
    print(msg5%('TOTAL',npts_tot,chi2_tot,color,chi2_npts_tot,rchi2_tot,nchi2_tot,chi2_summed,zcolor,zscore_tot))


    if legend:
        print()
        print("-"*len(msg1))
        print('\033[1mLEGEND\033[0m')
        print("-"*len(msg1))
        print('\033[32mchi2/npts < 1\033[0m, \033[97m1 < chi2/npts < 2\033[0m, \033[33m2 < chi2/npts < 3\033[0m, \033[91m3 < chi2/npts < 5\033[0m, \033[31mchi2/npts > 5\033[0m')


        print('\033[32mchi2norm < 1\033[0m, \033[97m1 < chi2norm < 4\033[0m, \033[33m4 < chi2norm < 9\033[0m, \033[91m9 < chi2norm < 25\033[0m, \033[31m chi2norm > 25\033[0m')


        print('\033[32m|zscore| < 1\033[0m, \033[97m1 < |zscore| < 2\033[0m, \033[33m2 < |zscore| < 3\033[0m, \033[91m3 < |zscore| < 5\033[0m, \033[31m |zscore| > 5\033[0m')


    print("-"*len(msg1))

def print_summary(wdir,REACTION=None,COL=None,OBS=None,TAR=None,HAD=None,legend=True):


    load_config('%s/input.py'%wdir)
    norm_tab=get_norm(wdir)
    chi2_tab=get_chi2(wdir,norm_tab,REACTION=REACTION,COL=COL,OBS=OBS,TAR=TAR,HAD=None)

    L = len(os.listdir('%s/msr-inspected'%wdir))

    print('\nsummary of  %s [%d/%d replicas]\n'%(wdir,chi2_tab['nrep'],L))

    #--adjust width of col and obs columns
    lobs,lcol = [],[]
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        if REACTION!=None and reaction!=REACTION: continue
        for idx in chi2_tab[reaction]:
            if idx in ['chi2','nchi2','rchi2','npts','chi2_npts','zscore','pull_npts','pull']: continue
            lcol.append(len(chi2_tab[reaction][idx]['col']))
            lobs.append(len(chi2_tab[reaction][idx]['obs']))

    lcol = np.max(lcol)
    lobs = np.max(lobs)

    msg1 ='\033[1m%10s '
    msg1+=' '*(lcol-2)
    msg1+='%s '
    msg1+='%5s '
    msg1+='%8s '
    msg1+=' '*(lobs-2)
    msg1+='%s '
    msg1+='%5s '
    msg1+='%10s '
    msg1+='%14s '
    msg1+='%10s '
    msg1+='%12s '
    msg1+='%10s '
    msg1+='%10s '
    msg1+='%10s \033[0m'
    msg1+='%10s '
    msg1=msg1%('idx','col','tar','had(s)','obs','npts','chi2','chi2rep','chi2red','chi2corr','norm','chi2norm','tot chi2','zscore')
    #print(msg1)
    #print("-"*len(msg1))
    chi2_tot = 0
    rchi2_tot= 0
    nchi2_tot= 0
    pull_tot = 0
    npts_tot = 0
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        print("-"*len(msg1))
        print('\033[1mREACTION: %s\033[0m'%reaction)
        print(msg1)
        print("-"*len(msg1))
        if REACTION!=None and reaction!=REACTION: continue
        l = list(chi2_tab[reaction])
        for _ in ['chi2','rchi2','nchi2','npts','chi2_npts','zscore','pull_npts','pull']: l.remove(_)
        for idx in sorted(l):
            col=chi2_tab[reaction][idx]['col']
            tar=chi2_tab[reaction][idx]['tar']
            had=chi2_tab[reaction][idx]['had']
            obs=chi2_tab[reaction][idx]['obs']
            if COL!=None and col!=COL: continue
            if OBS!=None and obs!=OBS: continue
            if TAR!=None and tar!=TAR: continue
            if HAD!=None and had!=HAD: continue
            npts=chi2_tab[reaction][idx]['npts']
            chi2=chi2_tab[reaction][idx]['chi2']
            chi2_rep_mid=chi2_tab[reaction][idx]['chi2_rep_mid']
            chi2_rep_err=chi2_tab[reaction][idx]['chi2_rep_err']
            zscore=chi2_tab[reaction][idx]['zscore']
            pull=chi2_tab[reaction][idx]['pull']
            chi2_npts=chi2_tab[reaction][idx]['chi2_npts']
            pull_npts=chi2_tab[reaction][idx]['pull_npts']
            chi2_corr=chi2_tab[reaction][idx]['chi2_corr']
            chi2_norm=chi2_tab[reaction][idx]['chi2_norm']

            chi2_tot_idx = chi2
            if chi2_corr != '-': 
                rchi2_tot += chi2_corr
                chi2_tot_idx += chi2_corr
            if reaction in norm_tab:
                if idx in norm_tab[reaction]:
                    fixed = norm_tab[reaction][idx]['fixed']
                    if fixed==False:
                        mean=norm_tab[reaction][idx]['mean']
                        std =norm_tab[reaction][idx]['std']
                        std =str(int(np.round(std*1000)))
                        norm = '%4.3f'%mean
                        if len(std)==1: norm+=' '
                        norm += '('+std+')'
                        nchi2_tot += chi2_norm
                        chi2_tot_idx += chi2_norm
                    else:
                        norm = '[%s]'%fixed
                else:
                    norm='-'
            else: norm = '-'

            pull_tot+=pull
            chi2_tot+=chi2
            npts_tot+=npts

            err =str(int(np.round(chi2_rep_err*1000)))
            chi2_rep = '%4.3f'%chi2_rep_mid
            if len(err)==1: chi2_rep+='  '
            if len(err)==2: chi2_rep+=' '
            chi2_rep += '('+err+')'

            if   chi2_npts < 1: color = '32' #--green
            elif chi2_npts < 2: color = '97' #--white
            elif chi2_npts < 3: color = '33' #--yellow/orange
            elif chi2_npts < 5: color = '91' #--bright red
            else:               color = '31' #--red

            if   np.abs(zscore) < 1: zcolor = '32' #--green
            elif np.abs(zscore) < 2: zcolor = '97' #--white
            elif np.abs(zscore) < 3: zcolor = '33' #--yellow/orange
            elif np.abs(zscore) < 5: zcolor = '91' #--bright red
            else:                    zcolor = '31' #--red

            if chi2_norm=='-':
                ncolor = ''
            else:
                if   chi2_norm < 1:  ncolor = '32' #--green
                elif chi2_norm < 4:  ncolor = '97' #--white
                elif chi2_norm < 9:  ncolor = '33' #--yellow/orange
                elif chi2_norm < 25: ncolor = '91' #--bright red
                else:                ncolor = '31' #--red


            msg2 ='%10d '
            msg2+=' '*(lcol-len(col)+1)
            msg2+='%s '
            msg2+='%5s '
            msg2+='%8s '
            msg2+=' '*(lobs-len(obs)+1)
            msg2+='%s '
            msg2+='%5d '
            msg2+='%11.2f '
            msg2+='%14s'
            msg2+='\033[%sm%10.2f\033[0m'
            if chi2_corr=='-': msg2+='%12s '
            else:              msg2+='%12.2f '
            if norm=='-': msg2 += '%10s ' 
            else:         msg2 += '%10s ' 
            if chi2_norm=='-': msg2+='%s%9s '
            else:              msg2+='\033[%sm%10.2f\033[0m'
            msg2+='%11.2f '
            msg2+='\033[%sm%10.2f\033[0m'
            print(msg2%(idx,col,tar,had,obs,npts,chi2,chi2_rep,color,chi2_npts,chi2_corr,norm,ncolor,chi2_norm,chi2_tot_idx,zcolor,zscore))

    chi2_npts_tot=chi2_tot/npts_tot
    pull_npts_tot=pull_tot/npts_tot

    zscore_tot = get_z_score(chi2_tot,npts_tot)

    print("-"*len(msg1))
    print('\033[1mPROCESS TOTALS:\033[0m')

    msg3='\033[1m%14s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s '
    msg3+='%10s \033[0m'
    msg3=msg3%('reaction','npts','chi2','chi2red','chi2corr','chi2norm','tot chi2','zscore')
    print(msg3)
    print("-"*len(msg1))


    #--print chi2 per experiment
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        if REACTION!=None and reaction!=REACTION: continue
        npts      = chi2_tab[reaction]['npts']
        chi2      = chi2_tab[reaction]['chi2']
        rchi2     = chi2_tab[reaction]['rchi2']
        nchi2     = chi2_tab[reaction]['nchi2']
        zscore    = chi2_tab[reaction]['zscore']
        chi2_npts = chi2_tab[reaction]['chi2_npts']
        pull_npts = chi2_tab[reaction]['pull_npts']
        if   chi2_npts < 1: color = '32' #--green
        elif chi2_npts < 2: color = '97' #--white
        elif chi2_npts < 3: color = '33' #--yellow/orange
        elif chi2_npts < 5: color = '91' #--bright red
        else:               color = '31' #--red

        if   np.abs(zscore) < 1: zcolor = '32' #--green
        elif np.abs(zscore) < 2: zcolor = '97' #--white
        elif np.abs(zscore) < 3: zcolor = '33' #--yellow/orange
        elif np.abs(zscore) < 5: zcolor = '91' #--bright red
        else:                    zcolor = '31' #--red

        chi2_summed = chi2 + rchi2 + nchi2
        msg4='%14s '
        msg4+='%10s '
        msg4+='%10.2f '
        msg4+='\033[%sm%10.2f\033[0m'
        msg4+='%11.2f '
        msg4+='%10.2f '
        msg4+='%10.2f '
        msg4+='\033[%sm%10.2f\033[0m'
        print(msg4%(reaction,npts,chi2,color,chi2_npts,rchi2,nchi2,chi2_summed,zcolor,zscore))

    if 'lqcd_itd' in conf and conf['lqcd_itd']:
        chi2,npts = get_lqcd_itd_chi2(wdir)
        reaction = 'lqcd_itd'
        chi2_tot += chi2
        npts_tot += npts
        chi2_npts = chi2/npts
        zscore = get_z_score(chi2,npts)
        msg ='%14s '
        msg+='%10s '
        msg+='%10.2f '
        msg+='\033[%sm%10.2f\033[0m'
        msg+='\033[%sm%10.2f\033[0m'
        if   chi2_npts < 1: color = '32' #--green
        elif chi2_npts < 2: color = '97' #--white
        elif chi2_npts < 3: color = '33' #--yellow/orange
        elif chi2_npts < 5: color = '91' #--bright red
        else:               color = '31' #--red

        if   np.abs(zscore) < 1: zcolor = '32' #--green
        elif np.abs(zscore) < 2: zcolor = '97' #--white
        elif np.abs(zscore) < 3: zcolor = '33' #--yellow/orange
        elif np.abs(zscore) < 5: zcolor = '91' #--bright red
        else:                    zcolor = '31' #--red
        print(msg%(reaction,npts,chi2,color,chi2_npts,zcolor,zscore))

    if   chi2_npts_tot < 1: color = '32' #--green
    elif chi2_npts_tot < 2: color = '97' #--white
    elif chi2_npts_tot < 3: color = '33' #--yellow/orange
    elif chi2_npts_tot < 5: color = '91' #--bright red
    else:                   color = '31' #--red
    
    if   np.abs(zscore_tot) < 1: zcolor = '32' #--green
    elif np.abs(zscore_tot) < 2: zcolor = '97' #--white
    elif np.abs(zscore_tot) < 3: zcolor = '33' #--yellow/orange
    elif np.abs(zscore_tot) < 5: zcolor = '91' #--bright red
    else:                    zcolor = '31' #--red

    chi2_summed = chi2_tot + rchi2_tot + nchi2_tot

    print("-"*len(msg1))
    msg5='\033[1m%14s \033[0m'
    msg5+='%10s '
    msg5+='%10.2f '
    msg5+='\033[%sm%10.2f\033[0m'
    msg5+='%11.2f '
    msg5+='%10.2f '
    msg5+='%10.2f '
    msg5+='\033[%sm%10.2f\033[0m'
    print(msg5%('TOTAL',npts_tot,chi2_tot,color,chi2_npts_tot,rchi2_tot,nchi2_tot,chi2_summed,zcolor,zscore_tot))


    if legend:
        print()
        print("-"*len(msg1))
        print('\033[1mLEGEND\033[0m')
        print("-"*len(msg1))
        print('\033[32mchi2/npts < 1\033[0m, \033[97m1 < chi2/npts < 2\033[0m, \033[33m2 < chi2/npts < 3\033[0m, \033[91m3 < chi2/npts < 5\033[0m, \033[31mchi2/npts > 5\033[0m')


        print('\033[32mchi2norm < 1\033[0m, \033[97m1 < chi2norm < 4\033[0m, \033[33m4 < chi2norm < 9\033[0m, \033[91m9 < chi2norm < 25\033[0m, \033[31m chi2norm > 25\033[0m')


        print('\033[32m|zscore| < 1\033[0m, \033[97m1 < |zscore| < 2\033[0m, \033[33m2 < |zscore| < 3\033[0m, \033[91m3 < |zscore| < 5\033[0m, \033[31m |zscore| > 5\033[0m')


    print("-"*len(msg1))


