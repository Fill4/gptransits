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
import scipy.stats
import scipy.optimize as op

# Gaussian processes libraries
import george
import celerite

# Returns a GP object from the george module with the provided parameters pars
def setup_george_gp(pars):
	t1, t2, jitter = pars
	#t1, t2 = pars
	k1 = t1**2 * george.kernels.ExpSquaredKernel(t2**2)
	k2 = george.kernels.WhiteKernel(jitter**2)
	kernel = k1 + k2
	gp = george.GP(kernel)
	return gp

def setup_celerite_gp(pars):
	w0, S0 = pars
	Q = 1.0 / np.sqrt(2.0)
	kernel = celerite.terms.SHOTerm(log_S0=np.log(S0**2), log_Q=np.log(Q), log_omega0=np.log(w0**2))
	kernel.freeze_parameter("log_Q")
	gp = celerite.GP(kernel)
	return gp

def print_pars(pars, priors):
	for par in priors:
		print("{:12} {:1} {:2} {:16.12f}".format(priors[par][0], "-", "", pars[par]))
	print("")

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
def log_likelihood(pars, phase, flux, error, priors, minimization=False):
	gp = setup_george_gp(pars)
	#gp = setup_celerite_gp(pars)
	gp.compute(phase, error)
	lnlikelihood = gp.lnlikelihood(flux - model(pars, phase), quiet=True)
	#lnlikelihood = gp.log_likelihood(flux - model(pars, phase))

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
	gp = setupGeorgeGP(pars)
	gp.compute(phase, error)
	return -gp.grad_lnlikelihood(flux, quiet=True)
'''

# Minimization Method
def run_minimization(data, priors, plot=False, init_pars=None):

	# Timing execution time
	print("-------------------------------------------------------")
	print("Starting Minimization")
	print("-------------------------------------------------------\n")
	startTimeMinimization = timeit.default_timer()

	# Setting up initial parameters and runinng scipy minimize
	init_pars = sample_priors(priors, 1)
	results = op.minimize(log_likelihood, init_pars, args=data + (priors, True), method='nelder-mead', tol=1e-18)

	fullTimeMinimization = timeit.default_timer() - startTimeMinimization
	print("Minimization execution time: {} usec\n".format(fullTimeMinimization))

	phase, flux, error = data
	# Setting up GP using results from minimization
	final_pars = results.x
	gp = setup_george_gp(final_pars)
	#gp = setup_celerite_gp(final_pars)
	gp.compute(phase, error)
	# Printing hyperparameters
	print("Hyperparameters from minimization:")
	print_pars(final_pars, priors)

	if plot:

		# Plotting the results from the minimization method
		fig = plt.figure("Minimization Method")
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		# Plot initial data with errors in both subplots
		ax1.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux')
		ax2.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux')
		x = np.linspace(min(phase), max(phase), len(phase)*5)
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(flux, x)
		std = np.sqrt(np.diag(cov))
		ax1.plot(x, mu, color="#ff7f0e", label= 'Mean distribution GP')
		ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.2, edgecolor="none", label= '3 sigma')
		ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= '2 sigma')
		ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma')
		ax1.title.set_text("Probability distribution for the gaussian process with the parameters from the minimization")
		ax1.set_ylabel('Flux')
		ax1.legend(loc='upper left')
		# Plot sample from the conditional ditribution in lower plot
		m = gp.sample_conditional(flux, x)
		ax2.plot(x, m, color="#4682b4", label='Sample')
		#ax2.plot(x, np.random.normal(loc=mu, scale=std), color="#4682b4", label='Sample')
		ax2.title.set_text("Sample drawn from the probability distribution above")
		ax2.set_xlabel('Time')	
		ax2.set_ylabel('Flux')
		ax2.legend(loc='upper left')

def run_mcmc(data, priors, plot=False, init_pars=None, nwalkers=20, burnin=500, iterations=2000):
	
	print("-------------------------------------------------------")
	print("Running MCMC")
	print("-------------------------------------------------------\n")
	startTimeMCMC = timeit.default_timer()

	# Draw samples from the prior distributions to have initial values for all the walkers
	init_pars = sample_priors(priors, nwalkers)
	ndim = len(init_pars)

	# Initiate the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=data + (priors, False))

	#Burn-in
	burnin_pars, _, _ = sampler.run_mcmc(init_pars.T, burnin)
	print("Acceptance fraction of walkers:")
	print(sampler.acceptance_fraction)
	print('')
	sampler.reset()

	#Run the walkers
	results, _, _ = sampler.run_mcmc(burnin_pars, iterations)
	print("Acceptance fraction of walkers:")
	print(sampler.acceptance_fraction)
	print('')

	fullTimeMCMC = timeit.default_timer() - startTimeMCMC
	print("Execution time: {} usec\n".format(fullTimeMCMC))
	
	phase, flux, error = data
	# Choosing a random chain from the emcee run
	samples = sampler.flatchain
	final_pars = np.median(samples, axis=0)
	print("Hyperparameters from MCMC:")
	print_pars(final_pars, priors)
	# Set up the GP for this sample.
	gp = setup_george_gp(final_pars)
	gp.compute(phase, error)

	if plot:

		# Plotting the results from the MCMC method
		fig1 = plt.figure("MCMC Method")
		ax1 = fig1.add_subplot(211)
		ax2 = fig1.add_subplot(212)
		# Plot initial data with errors in both subplots
		ax1.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux')
		ax2.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux')
		x = np.linspace(min(phase), max(phase), len(phase)*5)
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(flux, x)
		std = np.sqrt(np.diag(cov))
		ax1.plot(x, mu, color="#ff7f0e", label= 'Mean distribution GP')
		ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.2, edgecolor="none", label= '3 sigma')
		ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= '2 sigma')
		ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma')
		ax1.title.set_text("Probability distribution for the gaussian process with the parameters from the MCMC")
		ax1.set_ylabel('Flux')
		ax1.legend(loc='upper left')
		# Compute the prediction conditioned on the observations and plot it.
		# Plot sample from the conditional ditribution in lower plot
		m = gp.sample_conditional(flux, x)
		ax2.plot(x, m, color="#4682b4", label= 'Sample')
		ax2.title.set_text("Sample drawn from the probability distribution above")
		ax2.set_xlabel('Time')	
		ax2.set_ylabel('Flux')
		ax2.legend(loc='upper left')

		fig2 = corner.corner(samples, labels=["t_1", "t_2", "jitter"], 
							 quantiles=[0.5], show_titles=True, title_fmt='.8f',
							 truths=final_pars)
