import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pyfits
import george
from george import kernels
import emcee
import corner

def main(file, plot=False):

	# Read the data from the FITS file
	'''
	hdulist = pyfits.open(filein)
	time = hdulist[1].data.TIME[0:500]  # in days
	flux = hdulist[1].data.PDCSAP_FLUX[0:500]
	error = hdulist[1].data.SAP_FLUX_ERR[0:500]
	'''

	# Read data from a txt file
	data = np.loadtxt(file, usecols=(0,1,2))
	Nmax = 200
	time = data[:Nmax,0] - data[0,0]
	flux = data[:Nmax,1]
	error = data[:Nmax,2]

	# Remove all Nan from the data
	ind = np.logical_and(~np.isnan(time), ~np.isnan(flux))
	ntime = np.array(time[ind]) 
	nflux = np.array(flux[ind]) * 1e-8
	nerror = error[ind]
	nerror = np.mean(abs(nflux))/3 * np.random.random(len(time)) * 1e-8

	# Plot data and exit if True
	if plot == True:
		plt.scatter(ntime, nflux, s=1)
		plt.show()
		sys.exit('Exiting')

	# Kernel hyperparameters
	a = np.mean(abs(nflux))
	tau = 0.02
	jitter = 0.02
	hyperpar = (a, tau, jitter)

	'''
	def model(params, t):
		_, _, amp, loc, sig2 = params
		return amp * (-0.5 * (t - loc) ** 2 / sig2)

	def lnlike(p, t, y, yerr):
		a, tau = np.exp(p)
		gp = george.GP(a * kernels.ExpSquaredKernel(tau))
		gp.compute(t, yerr)
		return gp.lnlikelihood(y - model(p, t))
	'''

	def setup_gp(t1, t2, jitter):
		k1 = t1**2 * kernels.ExpSquaredKernel(t2**2)
		k2 = kernels.WhiteKernel(jitter)
		kernel = k1 + k2
		gp = george.GP(kernel, solver=george.HODLRSolver)
		return gp

	def lnprior(p):
		t1, t2, jitter = np.exp(p)
		if (-1 < t1 < 1 and -1 < t2 < 1 and -1 < jitter < 1):
			return 0.0
		return -np.inf

	def lnprob(p, ntime, nflux, nerror):
		prior = lnprior(p)
		t1, t2, jitter = np.exp(p)

		gp = setup_gp(t1, t2, jitter)

		gp.compute(ntime, nerror)

		return prior + gp.lnlikelihood(nflux, quiet=True) if np.isfinite(prior) else -np.inf

	data = (ntime, nflux, nerror)	
	nwalkers = 30
	ndim = len(hyperpar)

	p0 = [np.log(hyperpar) + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

	p0, _, _ = sampler.run_mcmc(p0, 500)
	#sampler.reset()

	sampler.run_mcmc(p0, 3000)

	plt.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Original Flux')

	x = np.linspace(min(ntime), max(ntime), Nmax*5)
		
	samples = sampler.flatchain
	for s in samples[np.random.randint(len(samples), size=1)]:

		# Set up the GP for this sample.
		t1, t2, jitter = np.exp(s)
		gp = setup_gp(t1, t2, jitter)
		gp.compute(ntime, nerror)

		output = 'Hyperparameters: alpha = ' + str(np.sqrt(t1)) + ' ; tau = ' + str(np.sqrt(t2)) + ' ; jitter = ' + str(jitter)
		print(output)

		# Compute the prediction conditioned on the observations and plot it.
		m = gp.sample_conditional(nflux, x)
		plt.plot(x, m, color="#4682b4", alpha=0.5, label='GP sample')
		#plt.errorbar(ntime, nflux-gp.sample_conditional(nflux, ntime), yerr=nerror, fmt=".r", capsize=0, label='Flux - GP sample')
		
		plt.xlabel('Time')		
		plt.ylabel('Flux')
		plt.legend()
		plt.title(output)

	samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
	figure = corner.corner(samples)
	plt.show()

if __name__ == "__main__":
	# File with data
	file = 'kplr_redgiant.dat'

	main(file, plot=False)
