import numpy as np
import emcee
import celerite
import timeit
import logging
import sys

# Internal functions
from backend import sample_priors, setup_gp, log_likelihood, print_params, scale_params
import plotting
from components_v2 import *


def mcmc(data, model, priors, plot_flags, nwalkers=20, iterations=2000, burnin=500):

	itime_mcmc = timeit.default_timer()

	time, flux, error = data
	
	# # Draw samples from the prior distributions to have initial values for all the walkers
	# init_params = sample_priors(priors, nwalkers)
	# ndim = len(init_params)
	# # gp = setup_gp(init_params[:,0])
	# # gp.compute(time/1e6)

	gp = GP(model, time)
	init_params = gp.model.sample_prior(nwalkers)
	ndim = init_params.shape[1]

	# Initiate the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=data + (priors, gp))

	# Burn-in
	logging.info('Runnning Burn-in ...')
	burnin_params, _, _ = sampler.run_mcmc(init_params, burnin, progress=True)
	sampler.reset()

	#Run the walkers
	logging.info('Running MCMC ...')
	results, _, _ = sampler.run_mcmc(burnin_params, iterations, progress=True)
	#logging.info("Acceptance fraction of walkers:")
	#logging.info(sampler.acceptance_fraction)

	time_mcmc = timeit.default_timer() - itime_mcmc
	logging.info("MCMC execution time: {:.4f} usec".format(time_mcmc))
	
	# Choosing a random chain from the emcee run
	samples = sampler.flatchain
	#final_params = np.median(samples, axis=0)
	q_16 = np.percentile(samples.T, 16, axis=1) 
	q_50 = np.percentile(samples.T, 50, axis=1)
	q_84 = np.percentile(samples.T, 84, axis=1)
	
	final_params = np.zeros((q_50.size*3,), dtype=q_50.dtype)
	final_params[0::3] = q_50
	final_params[1::3] = q_16
	final_params[2::3] = q_84
	params = q_50

	logging.info("Hyperparameters from MCMC:")
	print_params(params, priors)

	if plot_flags["plot_gp"]:
		plotting.plot_gp(params, data)

	if plot_flags["plot_corner"]:
		plotting.plot_corner(params, samples, priors)

	if plot_flags["plot_psd"]:
		plotting.plot_psd(params, data)

	return final_params




	""" TEST of using bounds to control variables. 
	Problem: variable transformation makes it not simple to determine correct limits in the celerite params
	# ----------------------------------------------------------------------------
	# ----------------------------------------------------------------------------
	#TEST

	prior_low = [10,1.2,100,30,10,40,70,1e-10]
	prior_high = [250,10,200,200,90,200,200,100]
	scale_prior_low = scale_params(prior_low)
	scale_prior_high = scale_params(prior_high)
	print(scale_prior_low)
	print(scale_prior_high)
	
	bounds_bump = dict(log_S0=(.1,8.6), log_Q=(np.log(1.2), np.log(10)), log_omega0=(np.log(scale_prior_low[2]), np.log(scale_prior_high[2])))

	bounds_gran1 = dict(log_S0=(.1,8.6), log_Q=(np.log(0.5), np.log(1)), log_omega0=(np.log(scale_prior_low[4]), np.log(scale_prior_high[4])))

	bounds_gran2 = dict(log_S0=(.1,8.6), log_Q=(np.log(0.5), np.log(1)), log_omega0=(np.log(scale_prior_low[6]), np.log(scale_prior_high[6])))

	bounds_jitter = dict(log_sigma=(1e-6,5))


	scaled_params = scale_params(init_params[:,0])
	S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, S0_gran_2, w_gran_2, jitter = np.log(scaled_params)

	kernel = celerite.terms.SHOTerm(log_S0=S0_bump, log_Q=Q_bump, log_omega0=w_bump, bounds=bounds_bump)

	Q = 1.0 / np.sqrt(2.0)
	kernel_1 = celerite.terms.SHOTerm(log_S0=S0_gran_1, log_Q=np.log(Q), log_omega0=w_gran_1, bounds=bounds_gran1)
	kernel_1.freeze_parameter("log_Q")
	kernel += kernel_1

	kernel_2 = celerite.terms.SHOTerm(log_S0=S0_gran_2, log_Q=np.log(Q), log_omega0=w_gran_2, bounds=bounds_gran2)
	kernel_2.freeze_parameter("log_Q")
	kernel += kernel_2

	kernel += celerite.terms.JitterTerm(log_sigma=jitter, bounds=bounds_jitter)

	gp = celerite.GP(kernel)


	# ----------------------------------------------------------------------------
	# ----------------------------------------------------------------------------
	"""