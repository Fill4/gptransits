import logging
import numpy as np

class Parameter(object):
	def __init__(self, name, latexName):
		self.name = name
		self.latexName = latexName

	def setPrior(self, priorType, priorParameters):
		self.priorType = priorType
		self.priorParameters = priorParameters
		self.priorBool = True

	def samplePrior(self):
		if self.priorBool:
			# Implement prior print
			pass
		else:
			print('Prior of parameter {} not defined'.format(self._name))

class Component(object):
	name = ''
	parameterArray = []
	parameterNames = []

	def __init__:

	def setPrior(self, name, priorType, priorParameters):
		try:
			i = self.parameterNames.index('name')
		except ValueError:
			print('No parameter {} in this component'.format(name))
		self.parameterArray[i].setPrior(priorType, priorParameters)

	def samplePrior(self):
		priorArray = np.array([])
		for parameter in self.parameterArray:
			prior = parameter.samplePrior()
			priorArray = np.append(priorArray, prior)

		return priorArray

	def asModel():
		return Model.asModel(self)

	def __add__(self, other):
		return Model(self, other)

	def scaleParametersToCelerite(self):
		raise NotImplementedError

	def scaleParameterFromCelerite(self):
		raise NotImplementedError

	def getKernel(self):
		raise NotImplementedError

class Granulation(Component, Model):
	name = 'Granulation'
	parameterNames = ['Amplitude', 'Frequency']
	parameterArray = [Parameter('A_gran', r'$A_{gran}$'), Parameter('w_gran', r'$\omega_{gran}$')]


class OscillationBump(Component, Model):
	name = 'OscillationBump'
	parameterNames = ['Amplitude', 'Q', 'Frequency']
	parameterArray = [Parameter('A_bump', r'$A_{bump}$'), Parameter('Q_bump', r'$Q_{bump}$'), Parameter('w_bump', r'$\omega_{bump}$')]

class Model(object):
	componentArray = []
	componentNames = []

	def	__init__(self, component, *args):
		
