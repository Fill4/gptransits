#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
from scipy.stats import linregress

# Join the results of each of the GP runs
def joinGPResults():
	dataDictionary = {}

	quarters = ['Q12', 'Q13', 'Q14', 'Q15', 'Q16']
	intervals = ['1', '2']
	for model in [1, 2]:
		f = open('results/gp_model{}.txt'.format(model), 'w')
		for quarter in quarters:
			for interval in intervals:
				
				data = pd.read_csv('results/{}.{}_model{}/parameters.dat'.format(quarter, interval, model), sep='\s+', header=None)
				numParameters = data.iloc[0].size-1
				dataDictionary['{}_{}'.format(quarter, interval)] = data

		for i in range(19):
			
			starName, starMatrix = joinValuesByStar(dataDictionary, numParameters, i)
			starParameters = calculateStatisticsForParameters(starMatrix)
			# Write star data in line to file and continue
			s = '{:}  ' + '  '.join(starParameters.astype('str')) + '\n'
			f.write(s.format(starName))

def joinDiamondsResults():
	dataDictionary = {}

	quarters = ['Q12', 'Q13', 'Q14', 'Q15', 'Q16']
	intervals = ['1', '2']
	for model in [1, 2]:
		f = open('results/diamonds_model{}.txt'.format(model), 'w')
		for quarter in quarters:
			for interval in intervals:
				
				data = pd.read_csv('results/diamonds_comparison/diamonds_q{}.{}_model{}.txt'.format(quarter[1:], interval, model), sep='\s+', header=None, comment='#')
				numParameters = data.iloc[0].size-1
				dataDictionary['{}_{}'.format(quarter, interval)] = data

		for i in range(19):
			
			starName, starMatrix = joinValuesByStar(dataDictionary, numParameters, i)
			starParameters = calculateStatisticsForParameters(starMatrix)
			# Write star data in line to file and continue
			s = '{:}  ' + '  '.join(starParameters.astype('str')) + '\n'
			f.write(s.format(starName))


# Values of stars are divided by quarter in the dictionary
# Join them in a matrix per star and return the matrix
def joinValuesByStar(dataDictionary, numParameters, i):
	starMatrix = np.zeros((10, numParameters))
	quarter = 0

	for key, value in dataDictionary.items():
		
		'''
		print(numParameters)
		print(np.array(value.iloc[i][1:]).size)
		print(starMatrix[quarter].size)
		print() '''

		if quarter == 0:
			starName = value.iloc[i][0][:12]
		starMatrix[quarter] = np.array(value.iloc[i][1:])
		quarter += 1
	
	return starName, starMatrix

# Use each star's data matrix to determine the median and sigma of each of the parameters according to the model
def calculateStatisticsForParameters(starMatrix):
	starParameters = np.array([])
	for i in range(0, starMatrix[0].size, 3):
		parameterMatrix = starMatrix[:,i:i+3]

		# Add median of the chosen quarter
		starParameters = np.append(starParameters, parameterMatrix[5,0])
		
		# Determine the sigma of all the medians
		stdCenter = (abs(parameterMatrix[1,1]-parameterMatrix[1,0]) + abs(parameterMatrix[1,2]-parameterMatrix[1,0])) / 2
		stdRemaining = np.std(parameterMatrix[:,0])
		stdFinal = np.sqrt(stdCenter ** 2 + stdRemaining ** 2)

		# Add the sigma found to the final values array
		starParameters = np.append(starParameters, stdFinal)

	return starParameters



'''
############################################################################
############################################################################
'''



def plotDiamondsGPComparisonFull():

	resultsFolder = 'presentation/'
	for model in [1, 2]:
		diamondsResultsFile = 'results/diamonds_model{}.txt'.format(model)
		gpResultsFile = 'results/gp_model{}.txt'.format(model)
		diamondsRawData = pd.read_csv(diamondsResultsFile, sep='\s+', header=None)
		gpRawData = pd.read_csv(gpResultsFile, sep='\s+', header=None, comment='#')

		modelNames = ['1 Granulation + Oscillation Bump', '2 Granulation + Oscillation Bump']

		parameterNames = ['A$_{gran,1}$', 'w$_{gran,1}$', 'A$_{gran,2}$', 'w$_{gran,2}$', 'A$_{bump}$', '$\\nu_{max}$', '$\sigma_{bump}$/Q$_{bump}$', 'Jitter']
		units = ['[ppm]', '[$\mu$Hz]', '[ppm]', '[$\mu$Hz]', '[ppm]', '[$\mu$Hz]', '[$\mu$Hz]', '[ppm]']
		if model == 1:
			parameterNames = parameterNames[0:2] + parameterNames[4:8]

		diamondsIndex = [[3,4,5,6,7,8,9,10,11,12,1,2], [3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2]]
		gpIndex = [[1,2,3,4,5,6,9,10,7,8,11,12], [1,2,3,4,5,6,7,8,9,10,13,14,11,12,15,16]]

		starsData = pd.read_csv("sample19/sample19.dat", sep='\s+', header=None, comment='#')
		diamondsData = np.array(diamondsRawData.loc[:,diamondsIndex[model-1]])
		gpData = np.array(gpRawData.loc[:,gpIndex[model-1]])
		#gpData[:,-2:len(gpData)] /= 2*np.pi

		
		ncols = 2 
		nrows = 2 + model
		font = 18

		# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(12, 12 + (model*6)))
		fig, axs = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(12, 12 + (model*6)))
		# fig.suptitle(modelNames[model-1], fontsize=16)
		#------------
		i=0
		p=0

		# gpData[:,-2] = np.log(gpData[:,-2])
		# gpData[:,-1] = np.log(gpData[:,-1])
		gpData[:,-2] = gpData[:,-2]**2 / 1300
		gpData[:,-1] = gpData[:,-1] / 10

		if model == 1:
			S0_gran = diamondsData[:,0] - gpData[:,0]
			S0_bump = diamondsData[:,4] - gpData[:,4]
			print(np.median(S0_gran))
			print(np.median(S0_bump))
			print(np.median(diamondsData[:,-2]-gpData[:,-2]))

		for ax in axs.reshape(-1): #axs.reshape(-1, order='F') to get columns first
			

			plot = ax.scatter(gpData[:,i], diamondsData[:,i], zorder=5, c=starsData[2], cmap="autumn")
			ax.errorbar(gpData[:,i], diamondsData[:,i], yerr=diamondsData[:,i+1], xerr=gpData[:,i+1], fmt=None, marker=None, mew=0, color=(0,0,0.8,0.4))
			#if i == len(axs)-1:
			#	cbar = fig.colorbar(plot)
			#	cbar.set_label(r'log$g$')

			x = np.linspace(gpData[:,i].min(), gpData[:,i].max(), num=500)
			if not parameterNames[p] == '$\sigma_{bump}$/Q$_{bump}$':
				ax.plot(x, x, ls="--", c="0.2", label='1:1 Line', alpha=0.6)

			linreg = linregress(gpData[:,i], diamondsData[:,i])
			rvalue = linreg.rvalue
			ax.plot(x, x*linreg.slope + linreg.intercept, ls="-", c="k", label='Linear fit', alpha=0.6)

			# ax.set_title(r'{:}  -->  RValue = {:.4f}'.format(parameterNames[p], abs(rvalue)), fontsize=16)
			ax.set_title(r'{:}'.format(parameterNames[p]), fontsize=font+2)
			#if ax.is_last_row():
			ax.set_xlabel(r'Celerite {:}'.format(units[p]),fontsize=font)
			#if ax.is_first_col():
			ax.set_ylabel(r'Diamonds {:}'.format(units[p]),fontsize=font)
			ax.tick_params(axis='both', which='major', labelsize=font, direction='out')
			ax.legend(fontsize=font, loc='upper left')

			i+=2
			p+=1

		#fig.subplots_adjust(right=0.9)
		cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.015])
		cbar_ax.tick_params(labelsize=font) 
		cbar = fig.colorbar(plot, cax=cbar_ax, orientation='horizontal')
		#cbar = fig.colorbar(plot, ax=axs.ravel().tolist())
		cbar.set_label(r'log($g$)', fontsize=font)

		fig.tight_layout(rect=[0.03, 0.07, 0.92, 0.98])
		# plt.savefig('{}/model{}_comparison.svg'.format(resultsFolder, model))
		plt.savefig('{}/model{}_comparison.png'.format(resultsFolder, model), dpi=300)
		# plt.show()
		plt.close('all')



def plotDiamondsGPComparisonLondres2018():

	resultsFolder = 'results/diamonds_comparison/'
	for model in [1, 2]:
		diamondsResultsFile = 'results/diamonds_model{}.txt'.format(model)
		gpResultsFile = 'results/gp_model{}.txt'.format(model)
		diamondsRawData = pd.read_csv(diamondsResultsFile, sep='\s+', header=None)
		gpRawData = pd.read_csv(gpResultsFile, sep='\s+', header=None, comment='#')

		#modelNames = ['1 Granulation + Oscillation Bump', '2 Granulation + Oscillation Bump']
		modelNames = ['Model 1', 'Model 2']

		parameterNames = ['A$_{gran,1}$', 'w0$_{gran,1}$', 'A$_{gran,2}$', 'w0$_{gran,2}$', 'A$_{bump}$', '$\\nu_{max}$', '$\sigma_{bump}$/Q$_{bump}$', 'Jitter']
		units = ['[ppm]', '[$\mu$Hz]', '[ppm]', '[$\mu$Hz]', '[ppm]', '[$\mu$Hz]', '[$\mu$Hz]', '[ppm]']
		if model == 1:
			parameterNames = parameterNames[0:2] + parameterNames[4:8]

		diamondsIndex = [[3,4,5,6,7,8,9,10,11,12,1,2], [3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2]]
		gpIndex = [[1,2,3,4,5,6,9,10,7,8,11,12], [1,2,3,4,5,6,7,8,9,10,13,14,11,12,15,16]]

		starsData = pd.read_csv("sample19/sample19.dat", sep='\s+', header=None, comment='#')
		diamondsData = np.array(diamondsRawData.loc[:,diamondsIndex[model-1]])
		gpData = np.array(gpRawData.loc[:,gpIndex[model-1]])
		#gpData[:,-2:len(gpData)] /= 2*np.pi

		

		ncols = model + 2
		nrows = 2

		fig, axs = plt.subplots(nrows=model, ncols=3, num=1, figsize=(12 + (model*2), model * 5))
		#fig.suptitle(modelNames[model-1], fontsize=16)
		#------------
		i=0
		p=0
		for ax in axs.reshape(-1, order='F'): #axs.reshape(-1, order='F') to get columns first
			if (i == 4 and model == 1):
				i = 6
				p = 3
			elif (i == 8 and model == 2):
				i = 10
				p = 5
			elif (i == 8 and model == 1) or (i == 12 and model == 2):
				break

			plot = ax.scatter(gpData[:,i], diamondsData[:,i], zorder=5, c=starsData[2], cmap="autumn")
			ax.errorbar(gpData[:,i], diamondsData[:,i], yerr=diamondsData[:,i+1], xerr=gpData[:,i+1], fmt=None, marker=None, mew=0, color=(0,0,0.8,0.4))
			#if i == len(axs)-1:
			#	cbar = fig.colorbar(plot)
			#	cbar.set_label(r'log$g$')

			x = np.linspace(gpData[:,i].min(), gpData[:,i].max(), num=500)
			if not parameterNames[p] == '$\sigma_{bump}$/Q$_{bump}$':
				ax.plot(x, x, ls="--", c="0.2", label='1:1 Line', alpha=0.6)

			linreg = linregress(gpData[:,i], diamondsData[:,i])
			rvalue = linreg.rvalue
			ax.plot(x, x*linreg.slope + linreg.intercept, ls="-", c="k", label='Linear fit', alpha=0.6)

			#ax.set_title(r'{:}  -->  RValue = {:.4f}'.format(parameterNames[p], abs(rvalue)), fontsize=16)
			ax.set_title(r'{:}'.format(parameterNames[p]), fontsize=16)
			#if ax.is_last_row():
			ax.set_xlabel(r'Celerite {:}'.format(units[p]),fontsize=14)
			#if ax.is_first_col():
			ax.set_ylabel(r'Diamonds {:}'.format(units[p]),fontsize=14)
			ax.tick_params(axis='both', which='major', labelsize=14, direction='out')
			ax.legend(fontsize=14, loc='upper left')

			i+=2
			p+=1

		#fig.subplots_adjust(right=0.9)
		cbar_ax = fig.add_axes([0.91, 0.08 + model*0.04, 0.02, 0.7])
		cbar = fig.colorbar(plot, cax=cbar_ax)
		#cbar = fig.colorbar(plot, ax=axs.ravel().tolist())
		cbar.set_label(r'log($g$)', fontsize=14)

		fig.tight_layout(rect=[0, 0.03, 0.90, 0.95])
		#plt.savefig('{}/model{}_comparison4.svg'.format(resultsFolder, model))
		plt.savefig('{}/model{}_comparisonLondres.png'.format(resultsFolder, model), dpi=500)
		plt.show()
		plt.close('all')

if __name__ == '__main__':
	# joinGPResults()
	# joinDiamondsResults()
	# plotDiamondsGPComparisonLondres2018()
	plotDiamondsGPComparisonFull()