import math
import sys
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import emcee
import pyfits
from matplotlib.ticker import MaxNLocator
import george
from george import kernels
import matplotlib.mlab as mlab
from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d

filein = 'Kepler91.fits'
hdulist = fits.open(filein)

# hdulist[1].data

# Leio os vetores (mas pus apenas os primeiros 1000)

time = hdulist[1].data.TIME[0:500]  # em dias
#flux = hdulist[1].data.SAP_FLUX
flux = hdulist[1].data.PDCSAP_FLUX[0:500]
error = hdulist[1].data.SAP_FLUX_ERR[0:500]
#bck = hdulist[1].data.SAP_BKG[0:1000]
#plt.plot(time,flux,'-')
#plt.show()

# Retiro uma serie de NAN que existem no time e no flux:

ind = ~np.isnan(time)
flux2 = flux[ind]
error2 = error[ind]
time2 = time[ind]

ind = ~np.isnan(flux2)
error3 = error2[ind]
time3 = time2[ind]
flux3 = flux2[ind]

# reponho os nomes:

phase = time3
flux = flux3-np.mean(flux3)
error = error3

plt.plot(phase,flux)
plt.show()

#sys.exit()

#
# Do my MCMC:
#

import numpy

# Prior parameters of the gaussian process
t1_prior = scipy.stats.uniform(loc = 0., scale = 0.1)  # amplitude
t2_prior = scipy.stats.uniform(loc = 0., scale = 1.) # termo de damping
# prior for the jitter
jitter_prior = scipy.stats.uniform(loc = 0., scale = 0.0005)

def lnprior(parameters):
    t1, t2, jitter = parameters
    return t1_prior.logpdf(t1) + t2_prior.logpdf(t2) + jitter_prior.logpdf(jitter)

def lnlike(parameters, phase, flux, error):
    t1, t2, jitter = parameters
    kernel = t1*t1 * kernels.ExpKernel(t2*t2) + kernels.WhiteKernel(jitter)
    gp = george.GP(kernel, solver=george.HODLRSolver)
    gp.compute(phase, error, sort = True)
    return gp.lnlikelihood(flux, quiet = False)

def lnprob(parameters, phase, flux, error):
    lp = lnprior(parameters)   # part of the priors
    return lp + lnlike(parameters, phase, flux, error) if numpy.isfinite(lp) else -numpy.inf   

ndim, nwalkers = 3, 12

t1init = t1_prior.rvs(nwalkers)
t2init = t2_prior.rvs(nwalkers)
jitterinit = jitter_prior.rvs(nwalkers)
p0 = numpy.array([t1init, t2init, jitterinit])
print(p0)
p0=p0.T  #FACO A TRANSPOSTA


print("Running first burn-in...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(phase, flux, error))
p0, lnprob, _ = sampler.run_mcmc(p0, 2000)
print(sampler.acceptance_fraction)

#print("Running production") 
# Here he will start a new set of chains using the last values from the 3rd bunrnin as starting point
sampler.reset()
num_construction_run = 2000
p0, _, _ = sampler.run_mcmc(p0, num_construction_run)
print(sampler.acceptance_fraction)

# Here I will analyze the results:

real = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*numpy.percentile(sampler.flatchain, [5, 50, 95], axis=0))) 
parametersnew = [real[0][0], real[1][0], real[2][0]]   
print("Final resulting parameters (t1, t2, jitter):")
print(parametersnew)
#print(lowererror)


chains = numpy.asarray(sampler.chain)
new = numpy.zeros((num_construction_run, (ndim*nwalkers)))
i = 0   # chain, de 1 a nwalkers
j = 0   # tiragem, de 1 a num_construction_run
while i < len(p0):
    while j < num_construction_run:
        new[j,(3*i+0)] = chains[i,j,0]
        new[j,(3*i+1)] = chains[i,j,1]
        new[j,(3*i+2)] = chains[i,j,2]
        j = j + 1
    i = i + 1

numpy.savetxt('teste.txt', new)

#Cria o array com a solucao final da orbita:

#array_param = [parametersnew[0],parametersnew[1],parametersnew[2],parametersnew[3]]

# Faco o modelo "m" final completo

flux_new = numpy.zeros(len(phase))
kernel = parametersnew[0] * kernels.ExpKernel(parametersnew[1]) + kernels.WhiteKernel(parametersnew[2])
gp = george.GP(kernel, solver=george.HODLRSolver)
gp.compute(phase, error)
m = gp.sample_conditional(flux, phase)

plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.figure (figsize = (15,7))
plt.xlabel('PHASE', fontsize = 30, color = 'black') 
plt.ylabel('Flux', fontsize = 30, color = 'black')
#plt.ylim(0.9997, 1.0001)
plt.errorbar(phase, flux, yerr=error, fmt='o', color='k', label = 'Flux')
plt.plot(phase, m, color = 'red',label='Fit')
plt.legend(title = '')
#plt.ion()
plt.show(block=True)

#sys.exit()

import corner
#make the corner plot
fig = corner.corner(sampler.flatchain[:,:], labels=["$t_1$", "$t_2$", "jitter"], truths = parametersnew)

fig.savefig("teste.png")

#plt.clf()
fig, axes = plt.subplots(ndim, 1, sharex=True)#, figsize=(3, 15))

axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(parametersnew[0], color="#888888", lw=2)
axes[0].set_ylabel("$t1$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(parametersnew[1], color="#888888", lw=2)
axes[1].set_ylabel("$t2$")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(parametersnew[2], color="#888888", lw=2)
axes[2].set_ylabel("$jitter$")

fig.tight_layout(h_pad=0.0)
fig.savefig("teste2.png")

numpy.savetxt('teste2.txt', real)

plt.show()
plt.close('all')

