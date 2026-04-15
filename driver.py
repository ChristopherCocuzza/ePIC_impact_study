#!/usr/bin/env python
import os,sys
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/ceph24/JAM/ccocuzza/lhapdf/python%s/sets'%version
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import kmeanconf as kc

#--from corelib
from analysis.corelib import core, inspect, predict, classifier, optpriors, jar, summary

#--from qpdlib
from analysis.qpdlib import pdf, offpdf, nucpdf, ppdf, ff, tolhapdf, QCFxlsx

#--from obslib
from analysis.obslib import stf, pstf, off, ht, idis, idis_R, pidis, sia, sidis, psidis, dy, wasym, zrap, wzrv, wpol, SU23, lattice
from analysis.obslib import discalc, xlsx, reweight, zscore

#--from parlib
from analysis.parlib  import params, corr

#--primary working directory
try: wdir=sys.argv[1]
except: wdir = None

Q2 = 10.0

hist=True

#sidis.plot_obs(wdir)
#stf.gen_stf(wdir,Q2=5,HT=True,TMC=True)
#stf.gen_stf(wdir,Q2=5,HT=False,TMC=True)
#stf.gen_stf(wdir,Q2=5,HT=False,TMC=False)
#stf.gen_stf(wdir,Q2=10,HT=False,TMC=True)
#stf.gen_stf(wdir,Q2=10,HT=False,TMC=False)
#summary.print_summary(wdir)
#sys.exit()

######################
##--Initial Processing
######################

mahal = False

if mahal: inspect.get_msr_inspected_mahal(wdir,quantile=1-1e-10)
else:     inspect.get_msr_inspected(wdir,limit=1000000)

predict.get_predictions(wdir,force=False)
#classifier.gen_labels(wdir,kc)
jar.gen_jar_file(wdir,kc)
summary.print_summary(wdir)

classifier.plot_chi2_dist_per_exp(wdir)

###################
##--Optimize priors
###################

#optpriors.gen_priors(wdir,kc,10)

##---------------------------------------------------------------
##--Parameter distributions
##---------------------------------------------------------------
hist=False

params.plot_params(wdir,'off pdf0',hist=False)
params.plot_params(wdir,'off pdf1',hist=False)
params.plot_params(wdir,'pdf',hist=False)
params.plot_params(wdir,'ht4',hist=False)

params.plot_params(wdir,'ffpion',hist)
params.plot_params(wdir,'ffkaon',hist)
params.plot_params(wdir,'ffhadron',hist)

params.plot_norms (wdir,'idis')
params.plot_norms (wdir,'dy'  )
params.plot_norms (wdir,'jet' )
params.plot_norms(wdir,'pidis')
params.plot_norms(wdir,'wzrv')


####################
##--Observable plots
####################

idis. plot_obs (wdir,kc)
dy.   plot_obs (wdir,kc)
dy.   plot_SQ  (wdir,kc,mode=0)
zrap. plot_obs (wdir,kc,mode=0)
wasym.plot_obs (wdir,kc,mode=0)
wzrv. plot_obs (wdir,kc,mode=0)
wzrv. plot_star(wdir,kc,mode=0)
dy.   plot_SQ  (wdir,kc,mode=1)
zrap. plot_obs (wdir,kc,mode=1)
wasym.plot_obs (wdir,kc,mode=1)
wzrv. plot_obs (wdir,kc,mode=1)
wzrv. plot_star(wdir,kc,mode=1)
#jet.   plot_obs(wdir,kc)
dy.   plot_E866_ratio(wdir,kc,mode=0)
dy.   plot_E866_ratio(wdir,kc,mode=1)

sia.plot_pion  (wdir)
sia.plot_kaon  (wdir)
sia.plot_hadron(wdir)

sidis.plot_obs(wdir)



###################
#--Plot proton and offshell pdfs
###################

pdf.gen_xf(wdir,Q2)
pdf.plot_xf(wdir,Q2,mode=0,sets=False)                
pdf.plot_xf(wdir,Q2,mode=1,sets=False)                

offpdf.gen_xf (wdir,Q2)
offpdf.plot_xf(wdir,Q2,mode=0)
offpdf.plot_xf(wdir,Q2,mode=1)
offpdf.gen_xf2 (wdir,Q2)
offpdf.plot_xf2(wdir,Q2,mode=0)
offpdf.plot_xf2(wdir,Q2,mode=1)
offpdf.plot_xf2(wdir,Q2,mode=2)
offpdf.plot_dqNA(wdir,Q2,mode=0)
offpdf.plot_dqNA(wdir,Q2,mode=1)
offpdf.plot_dqNA(wdir,Q2,mode=2)

##---------------------------------------------------------------
##--Fragmentation functions
##---------------------------------------------------------------

ff.gen_xf(wdir,'pion'  ,Q2=Q2)
ff.gen_xf(wdir,'kaon'  ,Q2=Q2)
ff.gen_xf(wdir,'hadron',Q2=Q2)
ff.plot_xf_pion  (wdir ,Q2=Q2,mode=0)
ff.plot_xf_pion  (wdir ,Q2=Q2,mode=1)
ff.plot_xf_kaon  (wdir ,Q2=Q2,mode=0)
ff.plot_xf_kaon  (wdir ,Q2=Q2,mode=1)
ff.plot_xf_hadron(wdir ,Q2=Q2,mode=0)
ff.plot_xf_hadron(wdir ,Q2=Q2,mode=1)


###################
#--Plot structure functions and related quantities 
###################

ht.gen_ht_W2fixed (wdir,W2=3.5)
ht.plot_ht_W2fixed(wdir,W2=3.5,stf='F2',mode=1)
ht.plot_ht_W2fixed(wdir,W2=3.5,stf='F2',mode=0)
ht.plot_ht_W2fixed(wdir,W2=3.5,stf='FL',mode=1)
ht.plot_ht_W2fixed(wdir,W2=3.5,stf='FL',mode=0)

stf.gen_stf (wdir,Q2)
stf.plot_stf(wdir,Q2,mode=1)
stf.plot_stf(wdir,Q2,mode=0)
stf.plot_F2A_components(wdir,Q2=4)
stf.plot_R(wdir,Q2,mode=0)
stf.plot_R(wdir,Q2,mode=1)
stf.plot_R(wdir,Q2,mode=2)

nucpdf.gen_nuclear_pdf (wdir,Q2)
nucpdf.plot_nuclear_pdf(wdir,Q2,mode=1)
nucpdf.plot_delta3     (wdir,Q2,mode=1)
nucpdf.gen_nuclear_pdf2 (wdir,Q2)
nucpdf.plot_nuclear_pdf2(wdir,Q2,mode=0)
nucpdf.plot_nuclear_pdf2(wdir,Q2,mode=1)
#nucpdf.plot_delta32     (wdir,Q2,mode=0)
#nucpdf.plot_delta32     (wdir,Q2,mode=1)

off.gen_df0(wdir,Q2=10)
off.plot_df0(wdir,Q2=10,mode=1)


stf.plot_F2A_components(wdir,Q2=10,mode=0)
stf.plot_F2A_components(wdir,Q2=10,mode=2)

stf.gen_F2_off_components(wdir,Q2=10)
stf.plot_F2A_off_components(wdir,Q2=10,mode=0)
stf.plot_F2A_off_components(wdir,Q2=10,mode=2)







sys.exit()
##---------------------------------------------------------------
##--Polarized
##---------------------------------------------------------------


########################
#--Polarized proton pdfs
########################
PSETS = []

ppdf.gen_xf(wdir,Q2=Q2)         
ppdf.plot_xf(wdir,Q2=10,mode=0)
ppdf.plot_xf(wdir,Q2=10,mode=1)
ppdf.plot_polarization(wdir,Q2=Q2,mode=0)
ppdf.plot_polarization(wdir,Q2=Q2,mode=1)
ppdf.plot_helicity    (wdir,Q2=Q2,mode=0)
ppdf.plot_helicity    (wdir,Q2=Q2,mode=1)
ppdf.gen_moments_trunc(wdir,Q2=4)


########################
#--polarized structure functions and related quantities
########################
pstf.gen_g2res(wdir,tar='p',Q2=10)
pstf.gen_g2res(wdir,tar='n',Q2=10)
pstf.plot_g2res(wdir,Q2=10,mode=0)
pstf.plot_g2res(wdir,Q2=10,mode=1)
ht.plot_pol_ht(wdir,Q2=Q2,mode=0)
ht.plot_pol_ht(wdir,Q2=Q2,mode=1)



########################
#--polarized observables
########################

pidis .plot_obs(wdir,kc) 
SU23  .plot_obs(wdir,kc,mode=1)
wpol  .plot_obs(wdir,kc,mode=1)
#pjet  .plot_obs(wdir,kc)
psidis.plot_obs(wdir,kc,mode=0)
psidis.plot_obs(wdir,kc,mode=1)












