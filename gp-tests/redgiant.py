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
import scipy.optimize as op

# Gaussian processes libraries
import george
import celerite

# Returns a GP object from the george module with the provided parameters pars
def setupGeorgeGP(pars):
	amplitude, timescale, jitter = np.exp(pars)
	k1 = amplitude**2 * george.kernels.ExpSquaredKernel(timescale**2)
	k2 = george.kernels.WhiteKernel(jitter)
	kernel = k1 + k2
	gp = george.GP(kernel, solver=george.HODLRSolver)
	return gp

def gpPrint(pars):
	hyperParsNames = ["Amplitude", "Timescale", "Jitter"]
	for i in range(len(pars)):
		print("{:12} {:1} {:2} {:10.8f}".format(hyperParsNames[i], "-", "", pars[i]))
	print("")

# Defining functions to handle the Monte Carlo method calculations
# Calculates the value of the model for parameters pars in the positions ntime
def model(pars, ntime):
	return np.zeros(len(ntime))

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
	lnlikelihood = gp.lnlikelihood(nflux - model(pars, ntime), quiet=True)

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

# Minimization Method
def runMinimization(pars, dataTuple, plot=False):

	print("-------------------------------------------------------")
	print("Running Minimization")
	print("-------------------------------------------------------\n")

	# Setting up initial parameters and runinng scipy minimize
	# Timing execution time
	startTimeMinimization = timeit.default_timer()
	inital_pars = np.log(pars)
	results = op.minimize(lnLike, inital_pars, args=dataTuple + (True,), jac=lnLikeGrad)
	fullTimeMinimization = timeit.default_timer() - startTimeMinimization
	print("Execution time: {} usec\n".format(fullTimeMinimization))

	ntime, nflux, nerror = dataTuple
	# Setting up GP using results from minimization
	hyperPars = np.exp(results.x)
	gp = setupGeorgeGP(np.log(hyperPars))
	gp.compute(ntime, nerror)
	# Printing hyperparameters
	print("Hyperparameters from minimization:")
	gpPrint(hyperPars)

	if plot:

		# Plotting the results from the minimization method
		fig = plt.figure("Minimization Method")
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		# Plot initial data with errors in both subplots
		ax1.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Flux')
		ax2.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Flux')
		x = np.linspace(min(ntime), max(ntime), len(ntime)*5)
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(nflux, x)
		std = np.sqrt(np.diag(cov))
		ax1.plot(x, mu, color="#ff7f0e", label= 'Mean distribution GP')
		ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.2, edgecolor="none", label= '3 sigma')
		ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= '2 sigma')
		ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma')
		ax1.title.set_text("Probability distribution for the gaussian process with the parameters from the minimization")
		ax1.set_ylabel('Flux')
		ax1.legend(loc='upper left')
		# Plot sample from the conditional ditribution in lower plot
		m = gp.sample_conditional(nflux, x)
		ax2.plot(x, m, color="#4682b4", label='Sample')
		ax2.title.set_text("Sample drawn from the probability distribution above")
		ax2.set_xlabel('Time')	
		ax2.set_ylabel('Flux')
		ax2.legend(loc='upper left')

	return hyperPars

def runMCMC(pars, dataTuple, plot=False, nwalkers=20, burnin=200, iterations=1000):
	
	print("-------------------------------------------------------")
	print("Running MCMC")
	print("-------------------------------------------------------\n")

	# Define parameters for emcee
	nwalkers = 30
	ndim = len(pars)

	# Initial parameters for the walkers are the parameters found in the minimization
	# perturbed by a random number 1/100th th scale of the pars
	# Timing the sampler execution
	startTimeMCMC = timeit.default_timer()
	inital_pars = np.log(pars)
	p0 = [inital_pars + inital_pars*1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnLike, args=dataTuple)

	#Burn-in
	p0, _, _ = sampler.run_mcmc(p0, burnin)
	sampler.reset()
	#Run the walkers
	results, _, _ = sampler.run_mcmc(p0, iterations)

	fullTimeMCMC = timeit.default_timer() - startTimeMCMC
	print("Execution time: {} usec\n".format(fullTimeMCMC))
	
	ntime, nflux, nerror = dataTuple
	# Choosing a random chain from the emcee run
	samples = sampler.flatchain
	print("Hyperparameters from MCMC:")
	hyperPars = results[np.random.randint(len(results))]
	gpPrint(hyperPars)
	# Set up the GP for this sample.
	gp = setupGeorgeGP(np.log(hyperPars))
	gp.compute(ntime, nerror)

	# Plotting the results from the MCMC method
	fig1 = plt.figure("MCMC Method")
	ax1 = fig1.add_subplot(211)
	ax2 = fig1.add_subplot(212)
	# Plot initial data with errors in both subplots
	ax1.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Flux')
	ax2.errorbar(ntime, nflux, yerr=nerror, fmt=".k", capsize=0, label= 'Flux')
	x = np.linspace(min(ntime), max(ntime), len(ntime)*5)
	# Plot conditional predictive distribution of the model in upper plot
	mu, cov = gp.predict(nflux, x)
	std = np.sqrt(np.diag(cov))
	ax1.plot(x, mu, color="#ff7f0e")
	ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.2, edgecolor="none", label= '3 sigma')
	ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.4, edgecolor="none", label= '2 sigma')
	ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma')
	ax1.title.set_text("Probability distribution for the gaussian process with the parameters from the MCMC")
	ax1.set_ylabel('Flux')
	ax1.legend(loc='upper left')
	# Compute the prediction conditioned on the observations and plot it.
	# Plot sample from the conditional ditribution in lower plot
	m = gp.sample_conditional(nflux, x)
	ax2.plot(x, m, color="#4682b4", label= 'Sample')
	ax2.title.set_text("Sample drawn from the probability distribution above")
	ax2.set_xlabel('Time')	
	ax2.set_ylabel('Flux')
	ax2.legend(loc='upper left')

	samples2 = np.exp(samples)
	samples3 = np.array([np.sqrt(samples2[:,0]), np.sqrt(samples2[:,1]), samples2[:,2]])
	fig2 = corner.corner(samples3.transpose(), labels=["amplitude", "timescale", "jitter"], quantiles=[0.5], show_titles=True)


def main(file, Nmax = 500, plot=False):

	print("\nStarting GP fitting procedure")
	print("-------------------------------------------------------\n")
	startTimeScript = timeit.default_timer()

	# Read data from a txt file
	data = np.loadtxt(file, usecols=(0,1,2))
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

	# Join complete data into a tuple to use in minimization
	initialHyperPars = (amplitude, timescale, jitter)
	dataTuple = (ntime, nflux, nerror)
	print("Initial hyperparameters:")
	gpPrint(initialHyperPars)

	# Run minimization
	resultsMinimization = runMinimization(initialHyperPars, dataTuple, plot=plot)
	
	# Run MCMC
	#runMCMC(initialHyperPars, dataTuple, plot=plot, nwalkers=25)

	fullTimeScript = timeit.default_timer() - startTimeScript
	print("-------------------------------------------------------\n")
	print("Complete execution time: {} usec".format(fullTimeScript))

	if plot:
		plt.show()

if __name__ == "__main__":
	
	file = 'kplr_redgiant.dat'
	main(file, Nmax=200, plot=True)