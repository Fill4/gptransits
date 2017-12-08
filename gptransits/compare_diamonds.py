#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

folder = 'Results/diamonds_comparison/'
diamonds_data = pd.read_csv(folder + 'diamonds_1gran_osc.dat', sep='\s+', header=None)
gp_data = pd.read_csv(folder + 'sample19_model2a.dat', sep='\s+', header=None, comment='#')

model = 2
model_names = ['1 Granulation', '1 Granulation + Oscillation Bump', '2 Granulation', '2 Granulation + Oscillation Bump']
model_list = ['1gran', '1gran_osc', '2gran', '2gran_osc']

param_names = ['A$_{gran,1}$', 'w$_{gran,1}$', 'A$_{gran,2}$', 'w$_{gran,2}$', 'A$_{bump}$', 'w$_{bump}$', '$\sigma_{bump}$/Q$_{bump}$']
units = ['[ppm]', '[$\mu$Hz]', '[ppm]', '[$\mu$Hz]', '[ppm]', '[$\mu$Hz]']
if model == 2:
	param_names = param_names[0:2]+param_names[4:7]

diamonds_idx = [[2,3], [2,3,4,5,6], [2,3,4,5], [2,3,4,5,6,7,8]]
gp_idx = [[1,2], [1,2,3,5,4], [1,2,3,4], [1,2,3,4,5,7,6]]

diamonds_array = np.array(diamonds_data.loc[:,diamonds_idx[model-1]])
gp_array = np.array(gp_data.loc[:,gp_idx[model-1]])


############################################################################
############################################################################
ncols = int(np.ceil(gp_array[0].size/2))
nrows = int(np.ceil(gp_array[0].size/3.6))
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, num=1, figsize=(14, 10))
fig.suptitle(model_names[model-1], fontsize=16)
#------------
i=0
for ax in axs.reshape(-1, order='F'): #axs.reshape(-1, order='F') to get columns first
	if i == gp_array[0].size:
		fig.delaxes(ax)
		break
	ax.plot(gp_array[:,i], diamonds_array[:,i], 'ok', zorder=5)

	x = np.linspace(gp_array[:,i].min(), gp_array[:,i].max(), num=500)
	if not param_names[i] == '$\sigma_{bump}$/Q$_{bump}$':
		ax.plot(x, x, ls="--", c=".3", label='1:1 Line')

	linreg = linregress(gp_array[:,i], diamonds_array[:,i])
	rvalue = linreg.rvalue
	ax.plot(x, x*linreg.slope + linreg.intercept, ls="-", c=".5", label='Linear fit')

	ax.set_title(r'{:}  -->  RValue = {:.4f}'.format(param_names[i], abs(rvalue)), fontsize=16)
	#if ax.is_last_row():
	ax.set_xlabel(r'Celerite {:}'.format(units[i]),fontsize=14)
	#if ax.is_first_col():
	ax.set_ylabel(r'Diamonds {:}'.format(units[i]),fontsize=14)
	ax.tick_params(axis='both', which='major', labelsize=14, direction='out')
	ax.legend(fontsize=14)

	i+=1

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(folder + model_list[model-1] + '_comparison.svg')
plt.savefig(folder + model_list[model-1] + '_comparison.png', dpi=200)
plt.close('all')
#plt.show()

