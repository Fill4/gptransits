''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import sys, os
import scipy.stats
import logging

# Gaussian processes libraries
import george
import celerite

#----------------------------------------------------------------------------------
# Returns a GP object from the celerite module with the provided parameters params
def setup_gp(params):
	scaled_params = scale_params(params)
	nparams = len(scaled_params)

	
	bounds_bump = dict(log_S0=(np.log(10), np.log(250)), log_Q=(np.log(1.2), np.log(10)), log_omega0=(np.log(100), np.log(200)))
	bounds_gran1 = dict(log_S0=(np.log(30), np.log(200)), log_Q=(np.log(0.5), np.log(1)), log_omega0=(np.log(10), np.log(90)))
	bounds_gran2 = dict(log_S0=(np.log(40), np.log(200)), log_Q=(np.log(0.5), np.log(1)), log_omega0=(np.log(70), np.log(200)))
	bounds_jitter = dict(log_sigma=(np.log(1e-10), np.log(100)))
	

	if nparams == 5:
		S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1 = np.log(scaled_params)

		kernel = celerite.terms.SHOTerm(log_S0=S0_bump, log_Q=Q_bump, log_omega0=w_bump)
	
		Q = 1.0 / np.sqrt(2.0)
		kernel_1 = celerite.terms.SHOTerm(log_S0=S0_gran_1, log_Q=np.log(Q), log_omega0=w_gran_1)
		kernel_1.freeze_parameter("log_Q")
		kernel += kernel_1

	elif nparams == 6:
		S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, jitter = np.log(scaled_params)

		kernel = celerite.terms.SHOTerm(log_S0=S0_bump, log_Q=Q_bump, log_omega0=w_bump)
	
		Q = 1.0 / np.sqrt(2.0)
		kernel_1 = celerite.terms.SHOTerm(log_S0=S0_gran_1, log_Q=np.log(Q), log_omega0=w_gran_1)
		kernel_1.freeze_parameter("log_Q")
		kernel += kernel_1

		kernel += celerite.terms.JitterTerm(log_sigma=jitter)

	elif nparams == 7:
		S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, S0_gran_2, w_gran_2 = np.log(scaled_params)

		kernel = celerite.terms.SHOTerm(log_S0=S0_bump, log_Q=Q_bump, log_omega0=w_bump)
	
		Q = 1.0 / np.sqrt(2.0)
		kernel_1 = celerite.terms.SHOTerm(log_S0=S0_gran_1, log_Q=np.log(Q), log_omega0=w_gran_1)
		kernel_1.freeze_parameter("log_Q")
		kernel += kernel_1

		kernel_2 = celerite.terms.SHOTerm(log_S0=S0_gran_2, log_Q=np.log(Q), log_omega0=w_gran_2)
		kernel_2.freeze_parameter("log_Q")
		kernel += kernel_2

		kernel += celerite.terms.JitterTerm(log_sigma=jitter)

	elif nparams == 8:
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
	return gp


# Scale parameters from Diamonds to celerite according to Foreman-Mackey 2017
def scale_params(params):
	
	nparams = len(params)

	if nparams == 5:
		a_bump, Q_bump, b_bump, a_gran_1, b_gran_1 = params

		# Oscillation bump parameters
		S0_bump = a_bump / ((Q_bump**2) * np.sqrt(2/np.pi))
		w_bump = b_bump * (2*np.pi)

		# First granulation parameters
		S0_gran_1 = (a_gran_1**2 / b_gran_1) * (2/np.sqrt(np.pi))
		w_gran_1 = b_gran_1 * (2*np.pi)

		scaled_params = [S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1]

	elif nparams == 6:
		a_bump, Q_bump, b_bump, a_gran_1, b_gran_1, jitter = params

		# Oscillation bump parameters
		S0_bump = a_bump / ((Q_bump**2) * np.sqrt(2/np.pi))
		w_bump = b_bump * (2*np.pi)

		# First granulation parameters
		S0_gran_1 = (a_gran_1**2 / b_gran_1) * (2/np.sqrt(np.pi))
		w_gran_1 = b_gran_1 * (2*np.pi)

		scaled_params = [S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, jitter]

	elif nparams == 7:
		a_bump, Q_bump, b_bump, a_gran_1, b_gran_1, a_gran_2, b_gran_2 = params

		# Oscillation bump parameters
		S0_bump = a_bump / ((Q_bump**2) * np.sqrt(2/np.pi))
		w_bump = b_bump * (2*np.pi)

		# First granulation parameters
		S0_gran_1 = (a_gran_1**2 / b_gran_1) * (2/np.sqrt(np.pi))
		w_gran_1 = b_gran_1 * (2*np.pi)

		# Second granulation parameters
		S0_gran_2 = (a_gran_2**2 / b_gran_2) * (2/np.sqrt(np.pi))
		w_gran_2 = b_gran_2 * (2*np.pi)

		scaled_params = [S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, S0_gran_2, w_gran_2]

	elif nparams == 8:
		a_bump, Q_bump, b_bump, a_gran_1, b_gran_1, a_gran_2, b_gran_2, jitter = params

		# Oscillation bump parameters
		S0_bump = a_bump / ((Q_bump**2) * np.sqrt(2/np.pi))
		w_bump = b_bump * (2*np.pi)

		# First granulation parameters
		S0_gran_1 = (a_gran_1**2 / b_gran_1) * (2/np.sqrt(np.pi))
		w_gran_1 = b_gran_1 * (2*np.pi)

		# Second granulation parameters
		S0_gran_2 = (a_gran_2**2 / b_gran_2) * (2/np.sqrt(np.pi))
		w_gran_2 = b_gran_2 * (2*np.pi)

		scaled_params = [S0_bump, Q_bump, w_bump, S0_gran_1, w_gran_1, S0_gran_2, w_gran_2, jitter]

	return scaled_params



def print_params(params, priors):
	for par in priors:
		logging.info("{:16} {:1} {:1} {:10.4f}".format(priors[par][0], "-", "", params[par]))



# Creates the dictionary priors with the instantiated probability distributions for each parameter 
def setup_priors(prior_settings):
	priors = {}
	for par in prior_settings:
		priors[par] = [prior_settings[par][0], getattr(scipy.stats, 
			prior_settings[par][1])(loc=prior_settings[par][2], 
			scale=prior_settings[par][3]-prior_settings[par][2])]
	return priors


# Draws num number of samples from each of the parameters distribution
def sample_priors(priors, num = 1):
	sample = np.zeros([len(priors), num])
	for par in priors:
		sample[par] = priors[par][1].rvs(num)
	return sample



#-----------------------------------------------------------------------------------
# Defining functions to handle the Monte Carlo method calculations
# Calculates the value of the model for parameters params in the positions time
def mean_model(params, time):
	return np.zeros(len(time))

# Defines the priors for all the parameters
def log_prior(params, priors):
	prior_sum = 0
	# Evaluate parameters according to priors. Return sum when working in log
	for par in priors:
		prior_sum += priors[par][1].logpdf(params[par])
	return prior_sum

# Calculates the lnlikelihood of the parameters params proposed. Can include model and
# prior calculation besides the gaussian process
def log_likelihood(params, time, flux, error, priors, gp):
	# Check priors and return if any parameter is outside prior

	# lnprior = log_prior(params, priors)
	# if not np.isfinite(lnprior):
	# 	return -np.inf
	
	# scaled_params = scale_params(params)
	# gp.set_parameter_vector(np.log(scaled_params))

	# -----------------------------------------------

	gp.model.set_parameters(params)
	lnprior = gp.model.prior_evaluate()
	if not np.isfinite(lnprior):
		return -np.inf
	gp.set_parameters(params)

	lnlikelihood = gp.log_likelihood(flux - mean_model(params, time))
	
	return lnprior + lnlikelihood

# Calculates the gradient of the lnlikelihood for the proposed parameters params
# This quantity is used in minimization methods such as scipy minimize
# Commented because it's unnecessary to the minimization right now
''' 
def lnLikeGrad(params, phase, flux, error, minimization=False):
	gp = setup_george(params)
	gp.compute(phase, error)
	return -gp.grad_lnlikelihood(flux, quiet=True)
'''