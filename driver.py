#!/usr/bin/env python
import os,sys
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import kmeanconf as kc

#--from corelib
from analysis.corelib import core, inspect, predict, classifier, optpriors, jar, summary

#--from qpdlib
from analysis.qpdlib import pdf, offpdf, nucpdf, ppdf, ff, tolhapdf, QCFxlsx

#--from obslib
from analysis.obslib import stf, pstf, off, ht, idis, pidis, sia, sidis, psidis, dy, wasym, zrap, wzrv, wpol, SU23, lattice, idis
from analysis.obslib import discalc, xlsx, reweight, zscore

#--from parlib
from analysis.parlib  import params, corr

#--primary working directory
try: wdir=sys.argv[1]
except: wdir = None

Q2 = 10.0
mc2 = 1.28**2 
mb2 = 4.18**2 


#xiaochao.gen_xlsx(wdir)
#classifier.plot_chi2_dist_per_exp(wdir)
#ht.gen_ht_W2fixed (wdir,W2=3.5)
#ht.plot_ht_W2fixed(wdir,W2=3.5,stf='F2',mode=1)
#ht.plot_ht_W2fixed(wdir,W2=3.5,stf='F2',mode=1)
#ht.plot_ht_W2fixed(wdir,W2=3.5,stf='FL',mode=0)
#ht.plot_ht_W2fixed(wdir,W2=3.5,stf='FL',mode=0)
#ff.gen_xf(wdir,'pion'  ,Q2=10)
#ppdf.plot_xf(wdir,Q2=10,mode=1)
#ppdf.plot_moments_trunc(wdir,Q2=100,mode=1,mom=1)
#summary.print_summary(wdir)
#sys.exit()

######################
##--Initial Processing
######################

inspect.get_msr_inspected(wdir,limit=100.0)
predict.get_predictions(wdir,force=False)
classifier.gen_labels(wdir,kc)
jar.gen_jar_file(wdir,kc)
summary.print_summary(wdir)

classifier.plot_chi2_dist_per_exp(wdir)

###################
##--Optimize priors
###################

#optpriors.gen_priors(wdir,kc,10)

##---------------------------------------------------------------
##--Polarized
##---------------------------------------------------------------

####################
##--Observable plots
####################

sia.plot_obs(wdir)
sidis. plot_obs(wdir)

idis. plot_obs (wdir,kc)
pidis. plot_obs (wdir,kc)

psidis. plot_obs (wdir,kc,mode=0)
psidis. plot_obs (wdir,kc,mode=1)

wzrv. plot_obs (wdir,kc,mode=0)
wzrv. plot_obs (wdir,kc,mode=1)
wzrv. plot_star(wdir,kc,mode=0)
wzrv. plot_star(wdir,kc,mode=1)

#pjet. plot_obs (wdir,kc,mode=0)
#pjet. plot_obs (wdir,kc,mode=1)

########################
#--Polarized proton pdfs
########################

ppdf.gen_xf(wdir,Q2=Q2)         
ppdf.plot_xf(wdir,Q2=10,mode=0)
ppdf.plot_xf(wdir,Q2=10,mode=1)


########################
#--polarized structure functions and related quantities
########################
#pstf.gen_g2res(wdir,tar='p',Q2=10)
#pstf.gen_g2res(wdir,tar='n',Q2=10)
#pstf.plot_g2res(wdir,Q2=10,mode=0)
#pstf.plot_g2res(wdir,Q2=10,mode=1)

#--unpolarized
pdf.gen_xf(wdir,Q2=Q2)         
pdf.plot_xf(wdir,Q2=10,mode=0)
pdf.plot_xf(wdir,Q2=10,mode=1)

##---------------------------------------------------------------
##--heavy quark PDFs
##---------------------------------------------------------------

pdf.gen_hq(wdir,Q2=mc2)
pdf.gen_hq(wdir,Q2=mb2)
pdf.gen_hq(wdir,Q2=100)
pdf.plot_hq(wdir,Q2=100,mode=0)
pdf.plot_hq(wdir,Q2=100,mode=1)



##---------------------------------------------------------------
##--Fragmentation functions
##---------------------------------------------------------------

ff.gen_xf(wdir,'pion'  ,Q2=100)
ff.gen_xf(wdir,'kaon'  ,Q2=100)
ff.gen_xf(wdir,'hadron',Q2=100)
ff.plot_xf_pion  (wdir ,Q2=100,mode=0)
ff.plot_xf_pion  (wdir ,Q2=100,mode=1)
ff.plot_xf_pion  (wdir ,Q2=100,mode=2)
ff.plot_xf_pion  (wdir ,Q2=100,mode=0,logx=True,logy=True)
ff.plot_xf_pion  (wdir ,Q2=100,mode=1,logx=True,logy=True)
ff.plot_xf_kaon  (wdir ,Q2=100,mode=0)
ff.plot_xf_kaon  (wdir ,Q2=100,mode=1)
ff.plot_xf_kaon  (wdir ,Q2=100,mode=0,logx=True,logy=True)
ff.plot_xf_kaon  (wdir ,Q2=100,mode=1,logx=True,logy=True)
ff.plot_xf_hadron(wdir ,Q2=100,mode=0)
ff.plot_xf_hadron(wdir ,Q2=100,mode=1)
ff.plot_xf_hadron(wdir ,Q2=100,mode=0,logx=True,logy=True)
ff.plot_xf_hadron(wdir ,Q2=100,mode=1,logx=True,logy=True)


##---------------------------------------------------------------
##--Parameter distributions
##---------------------------------------------------------------
hist=False

params.plot_params(wdir,'gXres',hist)
params.plot_params(wdir,'ffpion',hist)
params.plot_params(wdir,'ffkaon',hist)
params.plot_params(wdir,'ffhadron',hist)
params.plot_params(wdir,'ppdf',hist)
params.plot_params(wdir,'pdf',hist)













