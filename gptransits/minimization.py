import numpy as np
import scipy.optimize as op
import celerite
import timeit
import logging

# Internal functions
from backend import sample_priors, setup_gp, print_params
import plotting


def run_minimization(data, priors, plot_gp=False, init_params=None):

	# Timing execution time
	vprint('Minimization ...')
	itime_min = timeit.default_timer()

	time, flux, error = data

	# Setting up initial parameters and runinng scipy minimize
	init_params = sample_priors(priors, 1)

	gp = setup_gp(init_params)
	#gp.compute(time/1e6, error)
	gp.compute(time/1e6)

	results = op.minimize(log_likelihood, init_params, args=data + (priors, gp), method='nelder-mead', tol=1e-18)

	time_min = timeit.default_timer() - itime_min
	vprint("Minimization execution time: {:.4f} usec\n".format(time_min))

	# Setting up GP using results from minimization
	final_params = results.x
	vprint("Hyperparameters from minimization:")
	print_params(final_params, priors)
	
	if plot_gp:
		plotting.plot_gp(params, data)

	return final_params