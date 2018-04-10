import numpy as np
from scipy.stats import uniform
import celerite

class Component(object):
	name = ''
	npars = 0
	parameter_names = []
	default_priors = []

	def __init__(self, *args, **kwargs):
		if 'priors' in kwargs:
			self.priors = np.array(kwargs.get(priors))
		else:
			means = [prior[0] for prior in self.default_priors]
			stds = [prior[1]-prior[0] for prior in self.default_priors]
			self.priors = uniform(means, stds)

		if len(args):
			if len(args) != self.npars:
				raise ValueError("Expected {0} arguments but got {1}".format(npars, len(args)))
			self.parameter_array = np.array(args)
		else:
			self.parameter_array = self.sample_prior()[0]

	def sample_prior(self, num=1):
		return self.priors.rvs([num, self.npars])

	def eval_prior(self, log=True): # Add non log evaluation of prior
		return sum(self.priors.logpdf(self.parameter_array))

	def get_parameters_celerite(self):
		raise NotImplementedError

	def get_kernel(self):
		raise NotImplementedError
	
	def get_psd(self, freq):
		kernel = self.get_kernel()
		power = kernel.get_psd(2*np.pi*freq)
		return (self.name, power)
		

class Granulation(Component):
	name = 'Granulation'
	npars = 2
	parameter_names = ['A_gran', 'w0_gran']
	parameter_latex_names = [r'$A_{gran}$', r'$\omega_{gran}$']
	parameter_units = ['ppm', r'$\mu$Hz']
	default_priors = np.array([[20, 200], [10, 200]])

	def __repr__(self):
		return '{0} ({names[0]}:{values[0]:.3f} {units[0]}, {names[1]}:{values[1]:.3f} {units[1]})'.format(self.name, 
				names=self.parameter_names, values=self.parameter_array, units=self.parameter_units)

	def get_parameters_celerite(self):
		a, b = self.parameter_array

		S = (2/np.sqrt(np.pi)) * (a**2 / b)
		w = b * (2*np.pi)

		# return np.array([S, w])
		return np.log([S, w])

	def get_kernel(self):
		Q = 1.0 / np.sqrt(2.0)
		S, w = self.get_parameters_celerite()

		kernel = celerite.terms.SHOTerm(log_S0=S, log_Q=np.log(Q), log_omega0=w)
		# kernel = celerite.terms.SHOTerm(log_S0=np.log(S), log_Q=np.log(Q), log_omega0=np.log(w))
		kernel.freeze_parameter("log_Q")
		return kernel

class OscillationBump(Component):
	name = 'OscillationBump'
	npars = 3
	parameter_names = ['A_bump', 'Q', 'nu_max']
	parameter_latex_names = [r'$A_{bump}$', r'$Q_{bump}$', r'$\nu_{max}$']
	parameter_units = ['ppm', '', r'$\mu$Hz']
	default_priors = np.array([[10, 250], [1.2, 10], [100, 200]])

	def __repr__(self):
		return '{0}({names[0]}:{values[0]:.3f} {units[0]}, {names[1]}:{values[1]:.3f} {units[1]}, {names[2]}:{values[2]:.3f} {units[2]})'.format(self.name, 
				names=self.parameter_names, values=self.parameter_array, units=self.parameter_units)

	def get_parameters_celerite(self):
		A, Q, numax = self.parameter_array

		S = A / ((Q**2) * np.sqrt(2/np.pi))
		w = numax * (2*np.pi)

		# return np.array([S, Q, w])
		return np.log([S, Q, w])

	def get_kernel(self):
		S, Q, w = self.get_parameters_celerite()

		kernel = celerite.terms.SHOTerm(log_S0=S, log_Q=Q, log_omega0=w)
		# kernel = celerite.terms.SHOTerm(log_S0=np.log(S), log_Q=np.log(Q), log_omega0=np.log(w))

		return kernel

class WhiteNoise(Component):
	name = 'WhiteNoise'
	npars = 1
	parameter_names = ['jitter']
	parameter_latex_names = ['jitter']
	parameter_units = ['ppm']
	default_priors = np.array([[10, 100]])

	def __repr__(self):
		return '{0}({names[0]}:{values[0]:.3f} {units[0]})'.format(self.name, names=self.parameter_names, values=self.parameter_array, units=self.parameter_units)

	def get_parameters_celerite(self):
		jitter, = self.parameter_array

		# return np.array([jitter])
		return np.log([jitter])

	def get_kernel(self):
		jitter, = self.get_parameters_celerite()
		kernel = celerite.terms.JitterTerm(log_sigma=jitter)
		# kernel = celerite.terms.JitterTerm(log_sigma=np.log(jitter))
		return kernel





class Model(object):
	component_array = []

	def __repr__(self):
		string = 'Model with {0} components:\n'.format(len(self.component_array))
		for component in self.component_array:
			string += repr(component) + '\n'
		return string

	def	__init__(self, *args):
		if len(args):
			for arg in args:
				if isinstance(arg, Component):
					self.component_array.append(arg)
				else:
					raise ValueError("Args must be of type Component")
		else:
			raise ValueError("Model must have at least one component")

	def add(self, *args):
		if len(args):
			for arg in args:
				if isinstance(arg, Component):
					self.component_array.append(arg)
				else:
					raise ValueError("Args must be of type Component")
		else:
			raise ValueError("Must add at least one component to model")

	def set_parameters(self, params):
		i = 0
		for component in self.component_array:
			component.parameter_array = params[i:i+component.npars]
			i += component.npars

	def get_parameters(self):
		return np.hstack([component.parameter_array for component in self.component_array])

	def get_parameters_celerite(self):
		return np.hstack([component.get_parameters_celerite() for component in self.component_array])
		
	def get_parameters_names(self):
		return np.hstack([component.parameter_names for component in self.component_array])

	def get_parameters_latex(self):
		return np.hstack([component.parameter_latex_names for component in self.component_array])

	def get_parameters_units(self):
		return np.hstack([component.parameter_names for component in self.component_array])


	def prior_evaluate(self):
		prior = sum([component.eval_prior() for component in self.component_array])
		if not np.isfinite(prior):
			return -np.inf
		return prior

	def prior_sample(self, num=1):
		return np.hstack([component.sample_prior(num) for component in self.component_array])

	def get_kernel(self):
		kernel = celerite.terms.TermSum()
		for component in self.component_array:
			kernel += component.get_kernel()
		return kernel

	def get_psd(self, time):
		nyquist = (1 / (2*(time[1]-time[0])))*1e6
		f_sampling = 1 / (27.4*24*3600 / 1e6)
		freq = np.linspace(0.0, nyquist, (nyquist/f_sampling)+1 )

		psd_dict = dict(([component.get_psd(freq) for component in self.component_array]))
		return [freq, psd_dict]

	def get_kernel_list(self):
		pass



class GP(object):
	def __init__(self, model, time):
		if isinstance(model, Model):
					self.model = model
				else:
					raise ValueError("model arg must be of type Model")
		self.gp = celerite.GP(model.get_kernel())
		self.gp.compute(time/1e6)

	def set_parameters(self):
		celerite_params = self.model.get_parameters_celerite()
		self.gp.set_parameter_vector(celerite_params)

	def log_likelihood(self, residuals):
		return self.gp.log_likelihood(residuals)

	def predict(self, y, t=None, return_cov=True, return_var=False):
		return self.gp.predict(y, t, return_cov, return_var)