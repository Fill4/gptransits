#!/usr/bin/env python3

''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import timeit
import sys, os, os.path
import argparse
import logging

#Internal imports
from backend import setup_priors
from mcmc import *
from read_data import read_data
import plotting

def main(dataFolder, resultsFolder, model, prior_settings, plot_flags, nwalkers, iterations, burnin):

	#parser = argparse.ArgumentParser(description='Parse arguments')
	#group = parser.add_mutually_exclusive_group(required=True)
	#group.add_argument('file', '-f', type=str, help='Filename with lightcurve data', required=False)
	#group.add_argument('list', '-l', type=str, help='Filename with list of lightcurve files', required=False)
	#parser.add_argument('output', '-o', type=str, help='Folder where to write the results. Folder will be created if it doesnt exist')
	#parser.add_argument('verbose', '-v', type=bool, default=True, help='Enable verbose mode')
	#args = parser.parse_args()

	# If there is no folder, create it. Same for parameters file
	# resfolder = 'results/{}'.format(resultsFolder)
	resfolder = resultsFolder
	resfile = '{}/parameters.txt'.format(resfolder)
	if os.path.isdir(resfolder):
		pass
	else:
		os.mkdir(resfolder)
	if os.path.exists(resfile):
		pass
	else:
		z = open(resfile, 'w')
		#z.write('#{:17}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}  {:^10}\n'.format(
		#'Filename', 'Amp_1', 'Tscale_1', 'Amp_2', 'Tscale_2', 'Amp_osc', 'nu_max', 'osc_bump'))
		z.close()

	for filename in os.listdir(dataFolder):

		#name = os.path.splitext(os.path.basename(filename))[0]
		name = os.path.splitext(filename)[0]
		# Add filename to buffer and define variable with full path to file
		write_buffer = '{:15}'.format(name)

		# Start timer
		logging.info('Starting {:} ...'.format(name))
		itime_script = timeit.default_timer()

		# Bundle data in tuple for organisation
		data = read_data(dataFolder + filename)#, Nmax, offset, fits_options)

		# Initiate prior distributions according to options set by user
		priors = setup_priors(prior_settings)
		gp = GP(model, data[0])

		# Run run_minimizationn
		#backend.run_minimization(data, priors, plot=plot)

		# Run MCMC
		samples, final_params = mcmc(data, gp, priors, plot_flags, nwalkers=nwalkers, iterations=iterations, burnin=burnin)

		# Write final params from mcmc to buffer
		for i in range(len(final_params)):
			write_buffer += '  {:^8.3f}'.format(final_params[i])
		write_buffer += '\n'

		# Write buffer to results file
		z = open(resfile, 'a')
		z.write(write_buffer)
		z.close()

		# Plotting
		# fig_index = plt.get_fignums()

		if plot_flags['plot_gp']:
			plotting.plot_gp(gp, data)
		if plot_flags['plot_corner']:
			plotting.plot_corner(gp, samples)
		if plot_flags['plot_psd']:
			plotting.plot_psd(gp, data)
		if any(plot_flags.values()):
			plt.show()
			plt.close('all')

		# Print execution time
		time_script = timeit.default_timer() - itime_script
		logging.info("Complete execution time: {:.4f} usec".format(time_script))
		logging.info('End\n')

		sys.exit()

def diamondsRunAll():
	home = '/mnt/c/Users/Filipe/Downloads/work/phd'
	quarters = ['Q12', 'Q13', 'Q14', 'Q15', 'Q16']
	intervals = ['1', '2']
	for quarter in quarters:
		for interval in intervals:
			dataFolder = '{}/Data Diamonds GPs/{}.{}/lightcurve/'.format(home, quarter, interval)
			#resultsFolder = '{}/gptransits/gptransits/results/{}.{}_model1_new'.format(home, quarter, interval)
			resultsFolder = '{}/gptransits/gptransits/results/TEST'.format(home)
			print('Quarter {}.{} ...\n'.format(quarter, interval))
			main(dataFolder, resultsFolder)

if __name__ == '__main__':
	diamondsRunAll()
	#main()