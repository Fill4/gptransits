import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pyfits
import george
from george import kernels
import emcee
import corner
#import kplr

def main(file, plot=False):

	# Open FITS file with pyfits
	hdulist = pyfits.open(file)

	# Test open using kplr module
	#client = kplr.API()
	#star = client.light_curves(8219268, fetch = True)
	#hdulist = star[-1].open()

	# Read the data from the FITS file
	time = hdulist[1].data.TIME
	flux = hdulist[1].data.PDCSAP_FLUX
	error = hdulist[1].data.PDCSAP_FLUX_ERR

	# Remove all Nan from the data
	ind = np.logical_and(~np.isnan(time), ~np.isnan(flux))
	ntime = time[ind]
	nflux = flux[ind]
	nerror = error[ind]

	# Other data
	period = 6.24668005
	foldtimes = ntime % period
	#foldtimes = ntime / period
	#foldtimes = foldtimes % 1

	plt.errorbar(foldtimes, nflux, fmt='.k', markersize=1)
	plt.show()
	sys.exit() # Exitting

	nbins = 50               # chosen number of bins across the period
	width = 1.0/float(nbins) # calculate the width of the bins

	# create arrays for bin values and weights
	bins = np.zeros(nbins)
	weights = np.zeros(nbins)

	for i in range(len(nflux)):
		n = int(foldtimes[i] / width) # calculate bin number for this value
		weight = nerror[i]**-2.0         # calculate weight == error^-2
		bins[n] += nflux[i] * weight   # add weighted value to bin (value times weight)
		weights[n] += weight          # add weight to bin weights

	bins /= weights     # normalise weighted values using sum of weights
	binErr = np.sqrt(1.0/(weights))    # calculate bin errors from squared weights
	binEdges = np.arange(nbins) * width     # create array of bin edge values for plotting

	foldtimes = foldtimes * 6.3
	if plot:
		plt.scatter(foldtimes, nflux, s=1)
		plt.errorbar(binEdges,bins,yerr=binErr,linestyle='none',marker='o')  # plot binned lightcurve
		plt.axis((0,6.3,0.997,1.003))
		plt.show()
		sys.exit('Plotting and exiting')

	def model(params, t):
		_, _, amp, loc, sig2 = params
		return amp * (-0.5 * (t - loc) ** 2 / sig2)

	def lnlike(p, t, y, yerr):
		a, tau = np.exp(p[:2])
		gp = george.GP(a * kernels.ExpSquaredKernel(tau))
		gp.compute(t, yerr)
		return gp.lnlikelihood(y - model(p, t))

	def lnprior(p):
		lna, lntau, amp, loc, sig2 = p
		if (-5 < lna < 5 and  -5 < lntau < 5 and -10 < amp < 10 and -5 < loc < 5 and 0 < sig2 < 3):
			return 0.0
		return -np.inf

	def lnprob(p, t, y, yerr):
		lp = lnprior(p)
		return lp + lnlike(p, t, y, yerr) if np.isfinite(lp) else -np.inf

	data = (ntime, nflux, nerror)
	nwalkers = 12
	initial = np.array([0, 0, -1.0, 0.1, 0.4])
	ndim = len(initial)
	p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

	p0, _, _ = sampler.run_mcmc(p0, 250)
	sampler.reset()

	sampler.run_mcmc(p0, 500)

	plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)

	x = np.linspace(min(ntime), max(ntime), 500)
		
	samples = sampler.flatchain
	for s in samples[np.random.randint(len(samples), size=10)]:
		# Set up the GP for this sample.
		a, tau = np.exp(s[:2])
		gp = george.GP(a * kernels.ExpSquaredKernel(tau))
		gp.compute(t, yerr)

		# Compute the prediction conditioned on the observations and plot it.
		m = gp.sample_conditional(y - model(s, t), x) + model(s, x)
		plt.plot(x, m, color="#4682b4", alpha=0.3)

if __name__ == "__main__":
	# File with data
	file = 'Kepler91.fits'

	main(file, plot=True)