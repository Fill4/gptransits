#!/usr/bin/env python3

''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats
import timeit
import sys, os
import argparse
import logging

#Internal imports
# from backend import setup_priors
from components_v2 import GP
from mcmc import *
import plotting

# def main(dataFolder, resultsFolder, model, prior_settings, plot_flags, nwalkers, iterations, burnin):
def main(dataFolder, resultsFolder, model, plot_flags, nwalkers, iterations, burnin):

	# Start timer
	logging.info('Starting {:} ...'.format(filename))
	init_time = timeit.default_timer()
	
	#------------------------------------------------------------------
	#	INPUT
	#------------------------------------------------------------------

	# Handle the input flags of the script
	parser = argparse.ArgumentParser(description='Parse arguments')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--file', '-f', type=str, dest='input_file', help='Filename with lightcurve data', required=False)
	group.add_argument('--list', '-l', type=str, dest='input_list', help='Filename with list of lightcurve files', required=False)
	parser.add_argument('--output', '-o', type=str, dest='output', help='Folder where to write the results. Folder will be created if it doesnt exist')
	parser.add_argument('--verbose', '-v', type=bool, default=True, dest='verbose', help='Enable verbose mode')
	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(format='%(message)s', level=logging.INFO)

	if args.output:
		if not os.path.isdir(args.output):
			os.mkdir(args.output)
	# if not os.path.exists('{}/parameters.txt'.format(args.output)):

	filename = os.path.splitext(os.path.basename(args.input_file))[0]

	#------------------------------------------------------------------
	#	MAIN
	#------------------------------------------------------------------

	# Read data from input file and instanciate the GP using the time array and model
	data = np.loadtxt(args.input_file, unpack=True)
	gp = GP(model, data[0])

	# Run Minimizationn
	#backend.run_minimization(data, priors, plot=plot)

	# Run MCMC
	samples, results = mcmc(data, gp, plot_flags, nwalkers=nwalkers, iterations=iterations, burnin=burnin)

	# Replace model and gp parameters with median from results
	gp.model.set_parameters(results[1])
	gp.set_parameters(results[1])

	# If verbose mode, display the values obtained for each parameter
	params, names = (gp.model.get_parameters(), get_parameters_names())
	logging.info(''.join(["{:10}{:3}{:10.4f}\n".format(names[i], "-", params[i]) for i in range(len(params))]))

	#------------------------------------------------------------------
	#	OUTPUT
	#------------------------------------------------------------------

	# Write final parameters and uncertainties to output buffer
	header_buffer = '{:16}'.format('File')
	output_buffer = '{:16}'.format(filename)
	header_buffer += ''.join(['{:8}'.format(parameter_name for parameter_name in model.get_parameters_names())])

	output_buffer += '\n'

	# Write buffer to results file
	z = open(resfile, 'a')
	z.write(output_buffer)
	z.close()

	#------------------------------------------------------------------
	#	PLOT
	#------------------------------------------------------------------
	
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
	execution_time = timeit.default_timer() - init_time
	logging.info("Complete execution time: {:.4f} usec".format(execution_time))
	logging.info('End\n')



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