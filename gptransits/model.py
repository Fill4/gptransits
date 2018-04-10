# TODO
# Class to implement the logic to integrate both MeanModel and GPModel in the same container
# Purpose is to abstract the logic of the likelihood calculation when dealing with parameters of both models together
class Model(object):
	pass

# TODO
# Parametric model of the data that has a defined functional form
class MeanModel(object):
	pass

# Model that contains the components of the GP. Might be joined with GP in the future.
class GPModel(object):
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

# Class that implements the GP methods of the GPModel and interfaces with celerite methods
class GP(object):
	def __init__(self, model, time):
		if isinstance(model, GPModel):
					self.model = model
				else:
					raise ValueError("model arg must be of type GPModel")
		self.gp = celerite.GP(model.get_kernel())
		self.gp.compute(time/1e6)

	def set_parameters(self):
		celerite_params = self.model.get_parameters_celerite()
		self.gp.set_parameter_vector(celerite_params)

	def log_likelihood(self, residuals):
		return self.gp.log_likelihood(residuals)

	def predict(self, y, t=None, return_cov=True, return_var=False):
		return self.gp.predict(y, t, return_cov, return_var)