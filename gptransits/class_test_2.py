import numpy as np
import scipy.stats as stats

class Parameter(object):
	def __init__(self, name, latexName):
		self.name = name
		self.latexName = latexName
		self.value = self.samplePrior()

	def setPrior(self, priorType, priorParameters):
		self.priorType = priorType
		self.priorParameters = priorParameters

	def samplePrior(self):
		stats.uniform()
		

class Granulation(object):
	name = 'Granulation'
	parameterNames = ['Amplitude', 'Frequency']
	parameterArray = []

	def __init__():
		parameterArray.append(Parameter('A_gran', r'$A_{gran}$'))
		parameterArray.append(Parameter('A_gran', r'$A_{gran}$'))

	def samplePrior():	
		sample = [x.samplePrior() for x in parameterArray]


class GPModel(object):
	def __init__():
