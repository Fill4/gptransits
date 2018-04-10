''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import sys, os


# Calculates the value of the model for parameters params in the positions time
def mean_model(params, time):
	return np.zeros(len(time))

# def log_likelihood(params, time, flux, error, priors, gp):
def log_likelihood(params, time, flux, error, gp):

	gp.model.set_parameters(params)
	lnprior = gp.model.prior_evaluate()
	if not np.isfinite(lnprior):
		return -np.inf
	gp.set_parameters()

	lnlikelihood = gp.log_likelihood(flux - mean_model(params, time))
	
	return lnprior + lnlikelihood