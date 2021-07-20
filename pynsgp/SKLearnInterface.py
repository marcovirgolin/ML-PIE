from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import inspect

from pynsgp.Nodes.BaseNode import Node
from pynsgp.Nodes.SpecificNodes import *
from pynsgp.Fitness.FitnessFunction import FitnessFunction
from pynsgp.Evolution.Evolution import pyNSGP
from pynsgp.Interpretability.InterpretabilityFeatures import FeatureInterpretabilityExtractor as FIE


class pyNSGPEstimator(BaseEstimator, RegressorMixin):

	def __init__(self, 
		pop_size=100, 
		max_generations=100, 
		max_evaluations=-1,
		max_time=-1,
		error_metric = 'mse',
		functions=[ AddNode(), SubNode(), MulNode(), DivNode() ], 
		terminals=[],
		crossover_rate=0.9,
		mutation_rate=0.1,
		op_mutation_rate=1.0,
		initialization_max_tree_height=6,
		min_depth=2,
		tournament_size=4,
		max_tree_size=100,
		use_linear_scaling=True,
		use_offline_interpretability_model=False, 
		use_online_interpretability_model=False, 
		online_phi_warmup_epochs=20,
		penalize_duplicates = True,
		verbose=False,
		log_folder="out",
		online_model_log_folder="out"
		):

		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop('self')
		for arg, val in values.items():
			setattr(self, arg, val)

		assert(len(self.terminals) > 0)


	def fit(self, X, y):

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		n_features = X.shape[1]
		self.X_ = X
		self.y_ = y

		featInterpExtractor = None
		if self.use_offline_interpretability_model or self.use_online_interpretability_model:
			featInterpExtractor = FIE(self.terminals, self.functions)

		fitness_function = FitnessFunction( X, y, 
			error_metric = self.error_metric,
			use_linear_scaling=self.use_linear_scaling, 
			use_offline_interpretability_model=self.use_offline_interpretability_model,
			use_online_interpretability_model=self.use_online_interpretability_model,
			online_phi_warmup_epochs=self.online_phi_warmup_epochs,
			featInterpExtractor=featInterpExtractor,
			online_model_log_folder=self.online_model_log_folder
			)

		nsgp = pyNSGP(fitness_function, 
			self.functions, 
			self.terminals, 
			pop_size=self.pop_size, 
			max_generations=self.max_generations,
			max_time = self.max_time,
			max_evaluations = self.max_evaluations,
			crossover_rate=self.crossover_rate,
			mutation_rate=self.mutation_rate,
			op_mutation_rate=self.op_mutation_rate,
			initialization_max_tree_height=self.initialization_max_tree_height,
			min_depth=self.min_depth,
			max_tree_size=self.max_tree_size,
			tournament_size=self.tournament_size,
			penalize_duplicates = self.penalize_duplicates,
			verbose=self.verbose, 
			log_folder=self.log_folder
			)

		nsgp.Run()
		self.nsgp_ = nsgp

		return self

	def predict(self, X):
		# Check fit has been called
		check_is_fitted(self, ['nsgp_'])

		# Input validation
		X = check_array(X)
		fifu = self.nsgp_.fitness_function
		prediction = fifu.elite.ls_a + fifu.elite.ls_b * fifu.elite.GetOutput( X )

		return prediction

	def score(self, X, y=None):
		if y is None:
			raise ValueError('The ground truth y was not set')
		
		# Check fit has been called
		prediction = self.predict(X)
		return -1.0 * np.mean(np.square(y - prediction))

	def get_params(self, deep=True):
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		attributes = [a for a in attributes if not (a[0].endswith('_') or a[0].startswith('_'))]

		dic = {}
		for a in attributes:
			dic[a[0]] = a[1]

		return dic

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_elitist_obj1(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.fitness_function.elite

	def get_front(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.latest_front

	def get_population(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.population