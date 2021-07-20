import numpy as np
from copy import deepcopy

from pynsgp.Interpretability.InterpretabilityModel import OfflinePHIModel as OffPM
from pynsgp.Interpretability.InterpretabilityModel import OnlineRankingNetModel as OnNeuM

class FitnessFunction:

	def __init__( self, X_train, y_train, error_metric='mse', use_linear_scaling=True, 
		use_offline_interpretability_model=False, 
		use_online_interpretability_model=False, 
		online_phi_warmup_epochs=20,
		featInterpExtractor=None, online_model_log_folder='./out/' ):

		self.X_train = X_train
		self.y_train = y_train
		self.y_train_var = np.var(y_train)
		self.use_linear_scaling = use_linear_scaling
		self.use_offline_interpretability_model = use_offline_interpretability_model
		self.use_online_interpretability_model = use_online_interpretability_model
		self.elite = None
		self.evaluations = 0
		self.error_metric = error_metric

		self.uncert_scores = []

		if self.use_online_interpretability_model:
			if self.use_offline_interpretability_model:
				raise ValueError("cannot have both online and offline interpretability models set")
			self.interp_model  = OnNeuM(featInterpExtractor=featInterpExtractor, warm_start_n_epochs=online_phi_warmup_epochs, online_model_log_folder=online_model_log_folder)
		elif self.use_offline_interpretability_model:
			self.interp_model = OffPM(featInterpExtractor=featInterpExtractor)


	def __EvaluateInner(self, individual, only_interpretability=False):

		# compute obj1
		if only_interpretability:
			assert(len(individual.objectives) > 0)
		else:
			individual.objectives = []
			output = individual.GetOutput( self.X_train )
			a = 0.0
			b = 1.0
			if self.use_linear_scaling:
				b = np.cov(self.y_train, output)[0,1] / (np.var(output) + 1e-10)
				a = np.mean(self.y_train) - b*np.mean(output)
				individual.ls_a = a
				individual.ls_b = b
			output = a + b*output
			obj1 = self.ComputeError(output)
			individual.objectives.append( obj1 )
			individual.cached_output = ','.join([str(np.round(oi,6)) for oi in output])

		if self.use_online_interpretability_model or self.use_offline_interpretability_model:
			interp_score = self.EvaluateInterpModel(individual)
			# check if we have an uncertainty score
			if not isinstance(interp_score,float) and len(interp_score) == 2:
				uncert_score = interp_score[1]
				interp_score = interp_score[0]
				# keep track of uncertainty
				individual.interpr_uncertainty = uncert_score
			obj2 = -1 * interp_score
		else:
			obj2 = self.EvaluateNumberOfNodes(individual)	

		if only_interpretability:
			individual.objectives[1] = obj2
		else:
			individual.objectives.append( obj2 )



	def Evaluate( self, individual, only_interpretability=False ):
		self.evaluations = self.evaluations + 1
		self.__EvaluateInner(individual, only_interpretability=only_interpretability)
		if not self.elite or individual.objectives[0] < self.elite.objectives[0]:
			del self.elite
			self.elite = deepcopy(individual)

	def ComputeError(self, output):
		if self.error_metric == 'mse':
			error = np.mean( np.square( self.y_train - output ) )
		elif self.error_metric == 'binary_acc':
			output[ output < .5 ] = 0.0
			output[ output >= .5 ] = 1.0
			error = np.mean(self.y_train != output)
		else:
			raise ValueError('Unrecognized error metric',self.error_metric)
		
		if np.isnan(error):
			error = np.inf
		
		return error


	def EvaluateNumberOfNodes(self, individual):
		result = len(individual.GetSubtree())
		return result


	def EvaluateInterpModel(self, individual):
		result = self.interp_model.PredictInterpretability(individual)
		return result