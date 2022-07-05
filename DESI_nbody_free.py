## RUN IT WITH NBODY-kit env!!!

# Grandparent = model.py basic stuff
# Parents = Bao pow  more complicated powersp stuff. E.g. apply window 
# Children = bao_power_beutlet2017 they inherit everything from parents and Grandparent files. Unless they have already have their function they don;tt get the same with the other name. 
# children have the model specific things e.g. adding polynomial things and beao shift. 
# module load anaconda3/5.0.1
# use baoenv envitornment but to run it needs to load first - we use fortran from ozstar and not from 
# in the .bash_profile the following: 
# For capow compilation on ozstar 
# module load gcc/9.2.0
# module load openmpi/4.0.2
# module load gsl/2.5
# module load fftw/3.3.8
# Run once before everything ./capow "file/fiducial_bao_parameters.dat", "gal_file", "rand_file" "PK_fiducial_out_file"  

from __future__ import division
import astropy 
import astropy.io
from astropy.io import fits
from astropy.table import Table
from astropy import cosmology
from astropy.cosmology import z_at_value
from astropy.cosmology import FlatLambdaCDM
import numpy as np
#import bilby 
import numpy as np
import camb
import scipy
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy import optimize
import emcee
import matplotlib
import matplotlib.pyplot as plt
import sys
from subprocess import check_call
from nbodykit.source.catalog import ArrayCatalog
import scipy.constants as constant
from nbodykit.lab import  FFTPower


sys.path.append("..")

from barry.datasets.dataset_power_spectrum           import PowerSpectrum_SDSS_DR12  #models.bao_power_Beutler2017 import  PowerBeutler2017
from barry.models.bao_power_Beutler2017              import PowerBeutler2017
from barry.cosmology.power_spectrum_smoothing        import smooth_hinton2017
# we want to call camb each steps so need to modify for each step the linear power spectrum from camb. 


###root = abacus_cosm000.
# Baseline LCDM, Planck 2018 base_plikHM_TTTEEE_lowl_lowE_lensing mean
omegam_fiducial =  0.3137721  
dofidcamb = 1
num_mocks = 999
mink = 0.0
maxk = 0.2
delk = 0.005
 
#################  1. Cosmology   #################################################### 

cosmo = FlatLambdaCDM(H0=100, Om0=omegam_fiducial)# put fiducial value mocks )       #, Ob0=0.047, Tcmb0=2.725)

def com_dis(zzd):          return cosmo.comoving_distance(zzd).to_value()

Zint = np.linspace(0.0 , 1.0 , 10000) # #x = np.arange(Chi0, Chiinf, ) #  
Dint = cosmo.comoving_distance(Zint).to_value()    


def zd(_Chiv ):    return np.interp( _Chiv , Dint ,  Zint )
def Hv(zzd): return cosmo.H( zzd ).to_value() 

MTKM      =  10**(-3) #convert only c in km instread of meters ! 
#define constant and parameters for the problem: 
c = constant.c * MTKM
G = constant.G 

 ##################  end   ########################################################### 



#################  2. Read fits file and transform into RA DEC Z    #################################################### 
d_seed= 111

globx = [0.0]
globy = [0.0]
globz = [0.0]
#kmid = np.linspace(0.0, 0.3, 60, endpoint=False) + 0.0025 # relate thos to kmin kmax above in a smarter way

for region in range(0, 64):
    

       namefileinput = "/fred/oz073/rossi/freebao/DESI/EZ_LRG_fits/EZmock_B2000G512Z0.8N8015724_b0.385d4r169c0.3_seed%i/seed%i.sub%i.fits.gz" %(d_seed, d_seed, region)
       hdulist = fits.open( namefileinput )

       table = hdulist[1].data
       #table2 = hdulist[0].data
       #print(table2)


       globx = np.concatenate([ globx,  table.field('x')  ]) 
       globy = np.concatenate([ globy,  table.field('y')  ]) 
       globz = np.concatenate([ globz,  table.field('z')  ])       
       
       hdulist.close()



sumxyz2 = np.sqrt(globx**2 + globy**2 + globz**2 ) 

glob_r    = sumxyz2
glob_dec  = np.arcsin(globz/glob_r) # arcsin not arccos as DEC is  90 - theta or smt like that of sph coordin
glob_ra   = np.arctan(globy/globx)

glob_red = zd(glob_r) # using fiducial from mocks - # use a spline for that of r and z with fid from mocks



data = np.empty( len( globx ), dtype=[('Position', ('f8', 3) ) ] )#, ('Mass', 'f8') ]) 
f = ArrayCatalog({'Position' : data['Position']   })

 ##################  end   ########################################################### 




#################  3. Measure PK for a given Omega(=p1) Ra Dec Z catalogue     #################################################### 
def get_pk( p1):  #  data, randoms, outputfile, p1): #p1 = omega_m
    #CONVERT TO X Y Z BOX COORDINATE AGAIN - USING A DIFFERENT VALUE OF OMEGA_M - ADDING AP EFFECT MYSELF. #TO HAVE X Y Z
    # cosmo2 = FlatLambdaCDM(H0=100, Om0=p1 )       #, Ob0=0.047, Tcmb0=2.725)
    # gal_dist = cosmo2.comoving_distance(0.8).to_value() #JUST FOR NOW LIKE THIS!! # distance as function of omega_m and z 

    gal_dist =glob_r #only for now!!!!

    # to check are the radiants or degree 

    xcord = globx # gal_dist *  np.cos(glob_dec)*np.cos(glob_ra )
    ycord = globy # gal_dist *  np.cos(glob_dec)*np.sin(glob_ra )
    zcord = globz # gal_dist *  np.sin(glob_dec)
    
    #CREATE DATA: 
    f['Position'] = np.c_[ xcord, ycord, zcord]# maybe .T   # numpy.random.random(size=(1024, 3)).  #data['Mass'] = numpy.random.random(size=1024)

 
    mesh = f.to_mesh(Nmesh=[512, 512 , 512] , BoxSize=[2000, 2000, 2000],  resampler='tsc', interlaced=True)
    pk   = FFTPower(mesh, poles=[0], mode='1d', dk=delk, kmin=mink, kmax=maxk ) #poles only for monopole poles=[0],
    poles = pk.poles
    
    return(pk.poles["k"] + 0.0025 , pk.poles["power_0"].real - poles.attrs['shotnoise'] )
    
     # I can normalize it myself by using box size = 1 ? # P0 = delta_ k^2/Nmodes 
    # P0 = 2pi^3/  apha^3 x Volume - Volume of the box times alpha (as we change omegam) or compute from survey particles. we actually can ignore it and inglobate it in the bias parameters same with the shotnoise I think. 
    
 ##################  end   ########################################################### 




def getcambpk(p1): 
# Parameters
  om   =  p1        # Omega_m  is omega total dm (see line 75)
  h    =  0.6736     # Hubble parameter
  ob   =  0.02237/h**2 #sarebbe 0.04930   # Omega_b
  sig8 =  0.807952   # sigma_8
  ns   =  0.9649     # spectral index
  z    =  0.8        # redshift

# Range of k-values to compute (log-spaced)
  kmin = 1.e-3 # minimum k [h/Mpc]
  kmax = 1.e+1 # maximum k [h/Mpc]
  nk   = 400   # number of k values

# Compute variables needed by CAMB
  obh2 = ob*h*h      # physical density of baryons
  och2 = (om-ob)*h*h # physical density of cold dark matter
# Set up CAMB code
  pars = camb.CAMBparams()
  pars.set_cosmology(H0=100.*h,ombh2=obh2,omch2=och2)
  pars.InitPower.set_params(As=2.e-9,ns=ns,r=0)
# First compute linear P(k) and determine sigma_8 at z=0
  pars.NonLinear = camb.model.NonLinear_none
  pars.set_matter_power(redshifts=[0.],kmax=kmax)
  results = camb.get_results(pars)
  sig8test = np.array(results.get_sigma8())
# Now determine non-linear P(k), scaling A_s to give desired sigma_8
  pars.NonLinear = camb.model.NonLinear_both
  pars.InitPower.set_params(As=2.e-9*((sig8/sig8test)**2),ns=ns,r=0)
  pars.set_matter_power(redshifts=[z],kmax=kmax)
  results.calc_power_spectra(pars)
  karr,zarr1,pkarr1 = results.get_matter_power_spectrum(minkh=kmin,maxkh=kmax,npoints=nk)
  pkarr = pkarr1[0,:]
# karr is a 1D array of the k values in h/Mpc
# pkarr is a 1D array of the P(k) values in (Mpc/h)^3

  return(karr, pkarr)



def get_camb_smooth(p1):

    k, pk = getcambpk(p1)
    pksmooth = smooth_hinton2017(k, pk)
    pkratio = pk/pksmooth - 1. 

    return(k, pksmooth, pkratio )

def getcomp(ks):
    #this func is creating matrix 5 x length of k and 3 x lenght of k. If no wide angle-effect, we are creating a 3N_K x 5 N_k with value 1 over sub diagonal and 0 elsewere. 
    matrix = np.zeros((5 * ks.size, 3 * ks.size))
    matrix[: ks.size, : ks.size] = np.diag(np.ones(ks.size))
    matrix[2 * ks.size : 3 * ks.size, ks.size : 2 * ks.size] = np.diag(np.ones(ks.size))
    matrix[4 * ks.size :, 2 * ks.size :] = np.diag(np.ones(ks.size))
    return matrix




def get_data( infile, covfile ): #give those parameters directly here below like max_k example : 0.2
        f =  np.loadtxt(infile)
        kfid  =  f[:,0] 

        pkfid =  f[:,1]
        covfid =  np.loadtxt(covfile) #read auto as a matrix 
        # print(np.shape(covfid))
        # exit()

        data = PowerSpectrum_SDSS_DR12()
        d = data.get_data()[0]
        d["ks"]= kfid 

        d["cov"]= covfid    
        d["icov"]= np.linalg.inv(covfid)    
        d["icov_m_w"]= [None, None, None, None, None] # this is to speed up the code - avoiding the mutiplication between window,  wide angle effects and  model    

        d["ks_input"]= d["ks"]
        d["w_scale"]= np.zeros(kfid.size)   
        d["w_transform"]= np.eye(kfid.size) #if change this to multiples need to be 5 * Kfid ask Cullan in case you add multipoles. Window function is always 5 x k. But if model is isotropic barry would cut it down to right size. But since we are overwriting barry we do need to cut it down to the right side by ourselves   
        d["ks_output"] = kfid # output after window function convolution 

        d["w_pk"]= np.zeros(len(d["ks_output"]))     
        
        d["corr"]= covfid/(np.sqrt( np.diag(covfid)) *  np.atleast_2d(np.sqrt(np.diag(covfid ) ) ).T)     
        d["name"]= "mydata"     
        #for now use the old cosmo. This is overwritten anyway - later will be updated with barry for Abacus. 
        d["cosmology"]={   "z" : 0.9873,            #put all of those to fid values 
                           "om": 0.3,   #to be set to fiducial value(0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2, #modify to fiducial entries - it is probably not used - as I am overwriting Pk model at each iteration
                           "h0": 0.6774,                          
                            # "redshift": 0.9873,
                           "ob": 0.02230 / 0.6774 ** 2,
                           "ns": 0.9667,
                           "mnu": 0.00064 * 93.14,
                           "reconsmoothscale": 15 }   # reconstruction would be the only one relevant. Not here = no recon.   
        # this is the DESI COSMOLOGY - TO BE UPDTED LATER WHEN BARRY HAS THIS 

        # d["cosmology"]={   "z" : 0.8,            #put all of those to fid values 
        #                    "om": omegam_fiducial,  #from abacus values : (0.02237 + 0.1200)/h^2 ,   #to be set to fiducial value(0.1188 + 0.02230 + 0.00064) / 0.6774 ** 2, #modify to fiducial entries - it is probably not used - as I am overwriting Pk model at each iteration
        #                    "h0": 0.6736,                          
        #                     # "redshift": 0.9873,
        #                    "ob": 0.02237 / 0.6736** 2,
        #                    "ns": 0.9649,
        #                    "mnu": 0.00064 * 93.14,
        #                    "reconsmoothscale": 15 }   # reconstruction would be the only one relevant. Not here = no recon.   
     
        d["isotropic"]= True      
        d["m_transform"] = None #it is not needed if isotropic -only needed if anisotropic.  getcomp(kfid)

        d["w_m_transform"] = None # d["w_transform"]@d["m_transform"]

        d["poles"]=[0] #bc only fitting for monopoles (it is what data contain )
        d["fit_poles"]=[0] # it is what model contain - what it is actually fitted.  (e.g. maybe I want to ignore some poles included in data above e.g. hexadecapole)
        d["fit_pole_indices"]=[0] #index poles we are fitting (eg I am fitting P2 P0 P4 in this order , it would 1, 0, 2)
        d["min_k"]= mink
        d["max_k"]= maxk #match the one from capow. #self.max_k,
        d["w_mask"]= np.ones(len(kfid), dtype=bool ) # create an array or true/false this is used only for plotting - not for model comparison
        d["m_w_mask"]= np.ones(len(kfid), dtype=bool )  # this is used by get_likelihood in the isotropic model for model comparison
    #   print(np.shape(d["w_mask"] ) )
      #  exit()   
        d["pk0"]   = pkfid #just monopole
        d["pk"]    = pkfid #concatenated version of all multipoles. 
        
        return[d]
      

#if rebenning is needed
def _agg_data(step_size, k, pk, nk):

     
        if step_size == 1:
            k_rebinned = k
            pk_rebinned = pk
        else:
            add = k.size % step_size
            weight = nk
            if add:
                to_add = step_size - add
                k = np.concatenate((k, [k[-1]] * to_add))
                pk = np.concatenate((pk, [pk[-1]] * to_add))
                weight = np.concatenate((weight, [0] * to_add))
            k = k.reshape((-1, step_size))
            pk = pk.reshape((-1, step_size))
            weight = weight.reshape((-1, step_size))
            # Take the average of every group of step_size rows to rebin
            k_rebinned = np.average(k, axis=1)
            pk_rebinned = np.average(pk, axis=1, weights=weight) 
        
        return k_rebinned, pk_rebinned,
 

def post_Mylikelihood(p1, p2, p3 ): #corresponds to omega_m, sigma_s, sigma_nl. Where omega_m is fixed below but actually changed in the pk template by using model.kvals, model.pksmooth, model.pkratio = getcambpk()
   
   kk, pk = get_pk( omegam_fiducial) 
  
   model.data[0][ "pk" ]  =  pk[:] #pk[:-1] # this is removing the last element to match the precomputed P and cov (where last bin is removed manually b c = 0)
   model.data[0][ "k"  ]  =  kk[:] #kk[:-1] 
   
   model.kvals, model.pksmooth, model.pkratio = get_camb_smooth(p1)
  # model.plot([1.0, p2, p3])  #1.0 is doe alpha

   #for iii in range(len(model.pksmooth)):
    #   print( model.kvals[iii], model.pksmooth[iii])

#   print(np.c_[ model.kvals, model.pksmooth, model.pkratio ])

   return model.get_posterior([p2, p3]) #p2 and p3 are passed in the right order p2 =sigma_s and p3 = sigma_nl. In this order!
 

model = PowerBeutler2017(recon=None, isotropic=True, marg="full" , fix_params=["om", "alpha"] ) #first part of declare_parameters is in bao_power.py

print(model) 
#marg = full is a proper bayesian margin. integrating over the bias. Partial only vary alpha and for each value of alpha find the best fit of the others. None (mathmt equal to full. ) everything is free in MCMC
data  = get_data("/fred/oz073/rossi/freebao/DESI/PK/PK_EZ_512_1.txt", "cov_DESI_fiducial_512_file")
#print(data)
model.set_data(data)


print(post_Mylikelihood(0.3137721 , 3.2 , 5.7 ))
 













