''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys, os
import argparse
import logging

#Internal imports
from model import Model
import mcmc
import plot

def main(mean_model, gp_model, plot_flags, nwalkers, iterations, burnin):

	# Log start time
	init_time = timeit.default_timer()

	#------------------------------------------------------------------
	#	INPUT
	#------------------------------------------------------------------

	# Handle the input flags of the script
	parser = argparse.ArgumentParser(description='Parse arguments')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--file', '-f', type=str, dest='input_file', help='Filename with lightcurve data', required=False)
	group.add_argument('--list', '-l', type=str, dest='input_list', help='Filename with list of lightcurve files', required=False)
	parser.add_argument('--output', '-o', dest='output', action='store_true', help='Whether to write results to a file or not')
	parser.add_argument('--verbose', '-v', dest='verbose', action='store_false', help='Enable verbose mode')
	args = parser.parse_args()

	if args.verbose:
		logging.basicConfig(format='%(message)s', level=logging.INFO)

	if args.input_file:
		file_list = [args.input_file]
	elif args.input_list:
		file_list = [line.rstrip() for line in open(args.input_list, 'r')]
	else:
		raise ValueError("Please choose a file or a list")

	# Run main method for each of the stars
	for file in file_list:
		run(file, mean_model, gp_model, args.output, plot_flags, nwalkers, iterations, burnin)

	# Print execution time
	execution_time = timeit.default_timer() - init_time
	logging.info("Complete execution time: {:.4f} usec".format(execution_time))
	logging.info('End\n')



# Execution for each of the stars. Runs the mcmc and handles the results
def run(file, mean_model, gp_model, output, plot_flags, nwalkers, iterations, burnin):
	
	# Log start time
	init_star_time = timeit.default_timer()

	filename = os.path.splitext(os.path.basename(file))[0]
	logging.info('Starting {:} ...'.format(filename))


	# Read data from input file and instanciate the GP using the time array and model
	data = np.loadtxt(file, unpack=True)
	model = Model(mean_model, gp_model, data)

	# Run Minimizationn
	#backend.run_minimization(data, priors, plot=plot)

	# Run MCMC
	samples, results = mcmc.run_mcmc(model, nwalkers=nwalkers, iterations=iterations, burnin=burnin)

	# Replace model and gp parameters with median from results
	model.gp.gp_model.set_parameters(results[1])
	model.gp.set_parameters()

	# If verbose mode, display the values obtained for each parameter
	names = model.gp.gp_model.get_parameters_names()
	sigma_minus, median, sigma_plus = results
	logging.info(''.join(["{:10}{:3}{:10.4f}\n".format(names[i], "-", median[i]) for i in range(median.size)]))

	#------------------------------------------------------------------
	#	OUTPUT
	#------------------------------------------------------------------

	if output:
		z = open('{}/{}.out'.format(os.path.dirname(file), filename), 'a+')
		
		# Write final parameters and uncertainties to output buffer
		output_buffer = "# {:}: {:}\n".format("Filename", filename)
		output_buffer += "# {:>10}{:>12}{:>10}{:>10}\n".format("Parameter", "Median", "-Std", "+Std")

		output_buffer += ''.join(["  {:>10}{:>12.4f}{:>10.4f}{:>10.4f}\n".format(names[i], median[i], median[i]-sigma_minus[i], sigma_plus[i]-median[i]) for i in range(median.size)])
		# output_buffer += ''.join(['{:<12}'.format(parameter_name for parameter_name in model.gp.gp_model.get_parameters_names())])

		output_buffer += '# --------------------------------------------------------------------------\n'

		# Write buffer to results file
		z.write(output_buffer)
		z.close()

	#------------------------------------------------------------------
	#	PLOT
	#------------------------------------------------------------------
	
	if plot_flags['plot_gp']:
		plot.plot_gp(model, data)
	if plot_flags['plot_corner']:
		plot.plot_corner(model, samples)
	if plot_flags['plot_psd']:
		plot.plot_psd(model, data)
	if any(plot_flags.values()):
		plt.show()
		plt.close('all')

	# Print execution time
	execution_time_star = timeit.default_timer() - init_star_time
	logging.info("{:} elapsed time: {:.4f} usec".format(filename, execution_time_star))


# Loop to run all Diamonds stars. Need to move it elsewhere
"""
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
"""