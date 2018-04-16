import numpy as np
import emcee
import celerite
import timeit
import logging
import sys

# Internal functions
from backend import log_likelihood
#from backend import sample_priors, setup_gp, log_likelihood, print_params, scale_params
#import plotting
#from components_v2 import *


# def mcmc(data, gp, priors, plot_flags, nwalkers=20, iterations=2000, burnin=500):
def mcmc(data, gp, plot_flags, nwalkers=20, iterations=2000, burnin=500):

	init_time_mcmc = timeit.default_timer()

	# time, flux, error = data
	time, flux = data
	
	# Draw samples from the prior distributions to have initial values for all the walkers
	init_params = gp.model.prior_sample(nwalkers)
	ndim = init_params.shape[1]

	# Instanciate the sampler. Parameters arg is added by samples to log_likelihood function
	# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(time, flux, error, gp))
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(time, flux, gp))

	# Burn-in
	logging.info('Runnning Burn-in ...')
	burnin_params, _, _ = sampler.run_mcmc(init_params, burnin, progress=True)
	sampler.reset()

	# Main run
	logging.info('Running MCMC ...')
	results, _, _ = sampler.run_mcmc(burnin_params, iterations, progress=True)
	#logging.info("Acceptance fraction of walkers:")
	#logging.info(sampler.acceptance_fraction)

	time_mcmc = timeit.default_timer() - init_time_mcmc
	logging.info("MCMC execution time: {:.4f} usec".format(time_mcmc))
	
	# Get the samples from the MCMC run and calculate the median and 1-sigma uncertainties
	samples = sampler.flatchain
	results = np.percentile(samples.T, [16,50,84], axis=1)

	return samples, results