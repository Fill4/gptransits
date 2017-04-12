''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remainig stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import emcee
import corner
import timeit
import scipy.optimize as op

# Gaussian processes libraries
import george
import celerite

# Returns a GP object from the george module with the provided parameters pars
def setupGeorgeGP(p):
	amplitude, timescale, jitter = np.exp(p)
	k1 = amplitude**2 * george.kernels.ExpSquaredKernel(timescale**2)
	k2 = george.kernels.WhiteKernel(jitter)
	kernel = k1 + k2
	gp = george.GP(kernel, solver=george.HODLRSolver)
	return gp

# Defining functions to handle the Monte Carlo method calculations
# Claculates the value of the model for parameters pars in the positions ntime
def model(pars, ntime):
	return 0

# Defines the priors for all the parameters
def lnPrior(pars):
	amplitude, timescale, jitter = np.exp(pars)
	if (-1 < amplitude < 1 and -1 < timescale < 1 and -1 < jitter < 1):
		return 0.0
	return -np.inf

# Calculates the lnlikelihood of the parameters pars proposed. Can include model and
# prior calculation besides the gaussian process
def lnLike(pars, ntime, nflux, nerror, minimization=False):
	gp = setupGeorgeGP(pars)
	gp.compute(ntime, nerror)
	lnlikelihood = gp.lnlikelihood(nflux, quiet=True)

	#return -(prior + gp.lnlikelihood(nflux - model(pars, ntime), quiet=True)) if np.isfinite(prior) else 1e25
	if (minimization): # scipy minimize has problems with infinities
		return -lnlikelihood if np.isfinite(lnlikelihood) else 1e25
	else:
		prior = lnPrior(pars)
		return prior + lnlikelihood if np.isfinite(prior) else -np.inf

# Calculates the gradient of the lnlikelihood for the proposed parameters pars
# This quantity is used in minimization methods such as scipy minimize
def lnLikeGrad(pars, ntime, nflux, nerror, minimization=False):
	gp = setupGeorgeGP(pars)
	gp.compute(ntime, nerror)
	return -gp.grad_lnlikelihood(nflux, quiet=True)

def main(file, plot=False):

	# Read data from a txt file
	data = np.loadtxt(file, usecols=(0,1,2))
	Nmax = 500
	time = data[:Nmax,0] - data[0,0]
	flux = data[:Nmax,1]
	error = data[:Nmax,2]

	# Remove all Nan from the data and generate errors for the flux
	ind = np.logical_and(~np.isnan(time), ~np.isnan(flux))
	ntime = np.array(time[ind])
	nflux = np.array(flux[ind]) * 1e-8 # Normalize flux
	nerror = error[ind] 
	nerror = np.mean(abs(nflux))/2 * np.random.random(len(time)) #Generate errors

	# Define initial kernel hyperparameters
	amplitude = np.mean(abs(nflux))
	timescale = 0.02
	jitter = 0.02

	# Join hyperparameters to use in minimization
	hyperPars = (amplitude, timescale, jitter)
	# Join complete data into a tuple to use in minimization
	dataTuple = (ntime, nflux, nerror)

	#--------------------------------------------------------------------------
	# Minimization Method
	# Setting up initial parameters and runinng scipy minimize
	inital_pars = np.log(hyperPars)
	results = op.minimize(lnLike, inital_pars, args=dataTuple + (True,), jac=lnLikeGrad)

	# Setting up GP using results from minimization
	hyperPars = np.exp(results.x)
	gp1 = setupGeorgeGP(np.log(hyperPars))
	gp1.compute(ntime, nerror)
	# Printing hyperparameters
	amplitude, timescale, jitter = hyperPars
	hyperPar_str = 'Hyperparameters: alpha = ' + str(np.sqrt(amplitude)) + ' ; tau = ' + str(np.sqrt(timescale)) + ' ; jitter = ' + str(jitter)
	print(hyperPar_str)

	# Plotting the results from the minimization method
	fig1 = plt.figure(1)
	ax1_1 = fig1.add_subplot(211)
	ax1_2 = fig1.add_subplot(212)
	# Plot initial data with errors in both subplots
	ax1_1.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Flux')
	ax1_2.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Flux')
	x = np.linspace(min(ntime), max(ntime), Nmax*5)
	# Plot conditional predictive distribution of the model in upper plot
	mu, cov = gp1.predict(nflux, x)
	std = np.sqrt(np.diag(cov))
	ax1_1.plot(x, mu, color="#ff7f0e", label= 'Mean distribution GP')
	ax1_1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.2, edgecolor="none", label= '3 sigma')
	ax1_1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= '2 sigma')
	ax1_1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma')
	ax1_1.title.set_text("Probability distribution for the gaussian process with the parameters from the minimization")
	ax1_1.set_ylabel('Flux')
	ax1_1.legend(loc='upper left')
	# Plot sample from the conditional ditribution in lower plot
	m = gp1.sample_conditional(nflux, x)
	ax1_2.plot(x, m, color="#4682b4", label='Sample')
	ax1_2.title.set_text("Sample drawn from the probability distribution above")
	ax1_2.set_xlabel('Time')	
	ax1_2.set_ylabel('Flux')
	ax1_2.legend(loc='upper left')
	
	#-----------------------------------------------------------------------
	# MCMC method
	# Define parameters for emcee
	nwalkers = 30
	ndim = len(hyperPars)

	# Initial parameters for the walkers are the parameters found in the minimization
	# perturbed by a random number 1/100th th scale of the pars
	p0 = [results.x + results.x*1e-5*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnLike, args=dataTuple)

	#Burn-in
	p0, _, _ = sampler.run_mcmc(p0, 500)
	sampler.reset()
	#Run the walkers
	sampler.run_mcmc(p0, 2000)

	# Choosing a random chain from the emcee run
	samples = sampler.flatchain
	hyperPars = np.exp(samples[len(samples)-5])
	# Set up the GP for this sample.
	gp2 = setupGeorgeGP(np.log(hyperPars))
	gp2.compute(ntime, nerror)
	# Printing hyperparameters
	amplitude, timescale, jitter = hyperPars
	hyperPar_str = 'Hyperparameters: alpha = ' + str(np.sqrt(amplitude)) + ' ; tau = ' + str(np.sqrt(timescale)) + ' ; jitter = ' + str(jitter)
	print(hyperPar_str)

	# Plotting the results from the MCMC method
	fig2 = plt.figure(2)
	ax2_1 = fig2.add_subplot(211)
	ax2_2 = fig2.add_subplot(212)
	# Plot initial data with errors in both subplots
	ax2_1.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Flux')
	ax2_2.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Flux')
	x = np.linspace(min(ntime), max(ntime), Nmax*5)
	# Plot conditional predictive distribution of the model in upper plot
	mu, cov = gp2.predict(nflux, x)
	std = np.sqrt(np.diag(cov))
	ax2_1.plot(x, mu, color="#ff7f0e")
	ax2_1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.2, edgecolor="none", label= '3 sigma')
	ax2_1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= '2 sigma')
	ax2_1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma')
	ax2_1.title.set_text("Probability distribution for the gaussian process with the parameters from the MCMC")
	ax2_1.set_ylabel('Flux')
	ax2_1.legend(loc='upper left')
	# Compute the prediction conditioned on the observations and plot it.
	# Plot sample from the conditional ditribution in lower plot
	m = gp2.sample_conditional(nflux, x)
	ax2_2.plot(x, m, color="#4682b4", label= 'Sample')
	ax2_2.title.set_text("Sample drawn from the probability distribution above")
	ax2_2.set_xlabel('Time')	
	ax2_2.set_ylabel('Flux')
	ax2_2.legend(loc='upper left')
	

	#samples2 = np.exp(sampler.chain[:, 500:, :].reshape((-1, ndim)))
	samples2 = np.exp(samples)
	samples3 = np.array([np.sqrt(samples2[:,0]), np.sqrt(samples2[:,1]), samples2[:,2]])
	print np.shape(samples3)
	fig3 = corner.corner(samples3.transpose(), labels=["amplitude", "timescale", "jitter"], quantiles=[0.5], show_titles=True)

	fig1.show()
	fig2.show()
	fig3.show()

if __name__ == "__main__":
	# File with data
	file = 'kplr_redgiant.dat'

	main(file, plot=False)