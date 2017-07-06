#!/home/fill/anaconda3/bin/python
''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import emcee
import corner
import timeit
import pyfits
import scipy.stats
import scipy.optimize as op
from config_file import verbose

# Gaussian processes libraries
import george
import celerite

if verbose:
	def verboseprint(*args):
		# Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
		for arg in args:
			print(arg,)
		print()
else:   
	verboseprint = lambda *a: None # do-nothing function

def readfits(filename, Nmax):
	# Read data from fits and remove nans
	hdulist = pyfits.open(filename)

	ntime = getattr(hdulist[1].data, fits_options['time'])[:Nmax]
	nflux = getattr(hdulist[1].data, fits_options['flux'])[:Nmax]
	nerror = getattr(hdulist[1].data, fits_options['error'])[:Nmax]
	nerror[~np.isfinite(nerror)] = 1e-5

	ind = np.logical_and(~np.isnan(ntime), ~np.isnan(nflux))
	time = ntime[ind]
	time = time - time[0]
	flux = nflux[ind]/max(nflux[ind])
	error = nerror[ind]
	flux = flux - np.mean(flux)

	return (time, flux, error)

def readtxt(filename, Nmax):
	buffer = np.loadtxt(filename)

	ntime = buffer[:Nmax,0]
	nflux = buffer[:Nmax,1]
	nerror = buffer[:Nmax,2]
	nerror[~np.isfinite(nerror)] = 1e-5

	ind = np.logical_and(~np.isnan(ntime), ~np.isnan(nflux))
	time = ntime[ind]
	time = time - time[0]
	flux = nflux[ind]/max(nflux[ind])
	error = nerror[ind]
	flux = flux - np.mean(flux)

	return (time, flux, error)

# Calls the correct gp module according to user definition
def setup_gp(pars, module='george'):
	if module == 'george':
		gp = setup_george(pars)
	elif module == 'celerite':
		gp = setup_celerite(pars)
	else:
		sys.exit('Choose valid gaussian process module')
	return gp
	
# Returns a GP object from the george module with the provided parameters pars
def setup_george(pars):
	t1, t2, jitter = pars
	#t1, t2 = pars
	k1 = t1**2 * george.kernels.ExpSquaredKernel(t2**2)
	k2 = george.kernels.WhiteKernel(jitter**2)
	kernel = k1 + k2
	gp = george.GP(kernel)
	return gp

# Returns a GP object from the celerite module with the provided parameters pars
def setup_celerite(pars):
	S0, w0, jitter = pars
	#S0, w0 = pars
	Q = 1.0 / np.sqrt(2.0)
	kernel = celerite.terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0))
	kernel.freeze_parameter("log_Q")
	kernel += celerite.terms.JitterTerm(log_sigma=np.log(jitter**2))
	gp = celerite.GP(kernel)
	return gp

def print_pars(pars, priors):
	for par in priors:
		verboseprint("{:12} {:1} {:2} {:16.12f}".format(priors[par][0], "-", "", pars[par]))

# Defining functions to handle the Monte Carlo method calculations
# Calculates the value of the model for parameters pars in the positions phase
def model(pars, phase):
	return np.zeros(len(phase))

# Creates the dictionary priors with the instantiated probability distributions for each parameter 
def setup_priors(prior_settings):
	priors = {}
	for par in prior_settings:
		priors[par] = [prior_settings[par][0], getattr(scipy.stats, prior_settings[par][1])(loc=prior_settings[par][2], scale=prior_settings[par][3])]
	return priors

# Draws num number of samples from each of the parameters distribution
def sample_priors(priors, num = 1):
	sample = np.zeros([len(priors), num])
	for par in priors:
		sample[par] = priors[par][1].rvs(num)
	return sample

# Defines the priors for all the parameters
def log_prior(pars, priors):
	prior_sum = 0
	# Evaluate parameters according to priors. Return sum when working in log
	for par in priors:
		prior_sum += priors[par][1].logpdf(pars[par])
	return prior_sum

# Calculates the lnlikelihood of the parameters pars proposed. Can include model and
# prior calculation besides the gaussian process
def log_likelihood(pars, phase, flux, error, priors, minimization=False, module='george'):
	gp = setup_gp(pars, module)
	gp.compute(phase, error)
	if module == 'george':
		lnlikelihood = gp.lnlikelihood(flux - model(pars, phase), quiet=True)
	elif module == 'celerite':
		lnlikelihood = gp.log_likelihood(flux - model(pars, phase))
	else:
		sys.exit('Choose valid gaussian process module')

	if (minimization): # scipy minimize has problems with infinities
		return -lnlikelihood if np.isfinite(lnlikelihood) else 1e25
	else:
		prior = log_prior(pars, priors)
		return prior + lnlikelihood if np.isfinite(prior) else -np.inf

# Calculates the gradient of the lnlikelihood for the proposed parameters pars
# This quantity is used in minimization methods such as scipy minimize
# Commented because it's unnecessary to the minimization right now
''' 
def lnLikeGrad(pars, phase, flux, error, minimization=False):
	gp = setup_george(pars)
	gp.compute(phase, error)
	return -gp.grad_lnlikelihood(flux, quiet=True)
'''

# Minimization Method
def run_minimization(data, priors, plot=False, init_pars=None, module='george'):

	# Timing execution time
	verboseprint('\n{:_^60}\n'.format('Starting Minimization'))
	startTimeMinimization = timeit.default_timer()

	# Setting up initial parameters and runinng scipy minimize
	init_pars = sample_priors(priors, 1)
	results = op.minimize(log_likelihood, init_pars, args=data + (priors, True, module), method='nelder-mead', tol=1e-18)

	fullTimeMinimization = timeit.default_timer() - startTimeMinimization
	verboseprint("Minimization execution time: {:10.5} usec\n".format(fullTimeMinimization))

	phase, flux, error = data
	# Setting up GP using results from minimization
	final_pars = results.x
	gp = setup_gp(final_pars, module)
	gp.compute(phase, error)
	# Printing hyperparameters
	verboseprint("Hyperparameters from minimization:")
	print_pars(final_pars, priors)

	if plot:

		# Plotting the results from the minimization method
		fig = plt.figure("Minimization Method", figsize=(12, 12))
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		# Plot initial data with errors in both subplots
		ax1.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux')
		ax2.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux')
		x = np.linspace(min(phase), max(phase), len(phase)*5)
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(flux, x)
		std = np.sqrt(np.diag(cov))
		std = np.nan_to_num(std)
		ax1.plot(x, mu, color="#ff7f0e", label= 'Mean distribution GP')
		ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.2, edgecolor="none", label= '3 sigma')
		ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= '2 sigma')
		ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma')
		ax1.title.set_text("Probability distribution for the gaussian process with the parameters from the minimization")
		ax1.set_ylabel('Flux')
		ax1.legend(loc='upper left')
		# Plot sample from the conditional ditribution in lower plot
		#m = gp.sample_conditional(flux, x)
		#ax2.plot(x, m, color="#4682b4", label='Sample')
		ax2.plot(x, np.random.normal(loc=mu, scale=std), color="#4682b4", label='Sample')
		ax2.title.set_text("Sample drawn from the probability distribution above")
		ax2.set_xlabel('Time')	
		ax2.set_ylabel('Flux')
		ax2.legend(loc='upper left')

	return final_pars, fullTimeMinimization

def run_mcmc(data, priors, plot=False, init_pars=None, nwalkers=20, burnin=500, iterations=2000, module='george'):
	
	verboseprint('\n{:_^60}\n'.format('Starting MCMC'))
	startTimeMCMC = timeit.default_timer()

	# Draw samples from the prior distributions to have initial values for all the walkers
	init_pars = sample_priors(priors, nwalkers)
	ndim = len(init_pars)

	# Initiate the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=data + (priors, False, module))

	#Burn-in
	burnin_pars, _, _ = sampler.run_mcmc(init_pars.T, burnin)
	verboseprint("Acceptance fraction of burn-in walkers:")
	verboseprint(sampler.acceptance_fraction)
	
	sampler.reset()

	#Run the walkers
	results, _, _ = sampler.run_mcmc(burnin_pars, iterations)
	verboseprint("Acceptance fraction of final walkers:")
	verboseprint(sampler.acceptance_fraction)

	fullTimeMCMC = timeit.default_timer() - startTimeMCMC
	verboseprint("\nMCMC execution time: {:10.5} usec\n".format(fullTimeMCMC))
	
	phase, flux, error = data
	# Choosing a random chain from the emcee run
	samples = sampler.flatchain
	final_pars = np.median(samples, axis=0)
	verboseprint("Hyperparameters from MCMC:")
	print_pars(final_pars, priors)

	# Set up the GP for this sample.
	gp = setup_gp(final_pars, module)
	gp.compute(phase, error)

	if plot:

		# Plotting the results from the MCMC method
		fig1 = plt.figure("MCMC Method", figsize=(12, 12))
		ax1 = fig1.add_subplot(211)
		ax2 = fig1.add_subplot(212)
		# Plot initial data with errors in both subplots
		ax1.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux')
		ax2.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux')
		x = np.linspace(min(phase), max(phase), len(phase)*5)
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(flux, x)
		std = np.sqrt(np.diag(cov))
		std = np.nan_to_num(std)
		ax1.plot(x, mu, color="#ff7f0e", label= 'Mean distribution GP')
		ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.2, edgecolor="none", label= '3 sigma')
		ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= '2 sigma')
		ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma')
		ax1.title.set_text("Probability distribution for the gaussian process with the parameters from the MCMC")
		ax1.set_ylabel('Flux')
		ax1.legend(loc='upper left')
		# Compute the prediction conditioned on the observations and plot it.
		# Plot sample from the conditional ditribution in lower plot
		#m = gp.sample_conditional(flux, x)
		#ax2.plot(x, m, color="#4682b4", label= 'Sample')
		ax2.plot(x, np.random.normal(loc=mu, scale=std), color="#4682b4", label='Sample')
		ax2.title.set_text("Sample drawn from the probability distribution above")
		ax2.set_xlabel('Time')	
		ax2.set_ylabel('Flux')
		ax2.legend(loc='upper left')

		labels = [priors[i][0] for i in priors]
		fig2 = corner.corner(samples, labels=labels, 
							 quantiles=[0.5], show_titles=True, title_fmt='.6f',
							 truths=final_pars, figsize=(15, 15))

	return final_pars