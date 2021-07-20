import numpy as np
from numpy.random import random, randint
import time, json, os
from copy import deepcopy


from pynsgp.Variation import Variation
from pynsgp.Selection import Selection


class pyNSGP:

	def __init__(
		self,
		fitness_function,
		functions,
		terminals,
		pop_size=500,
		crossover_rate=0.9,
		mutation_rate=0.1,
		op_mutation_rate=1.0,
		max_evaluations=-1,
		max_generations=-1,
		max_time=-1,
		initialization_max_tree_height=4,
		min_depth=2,
		max_tree_size=100,
		tournament_size=4,
		penalize_duplicates=True,
		verbose=False,
		log_folder="out/",
		):

		self.pop_size = pop_size
		self.fitness_function = fitness_function
		self.functions = functions
		self.terminals = terminals
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.op_mutation_rate = op_mutation_rate

		self.max_evaluations = max_evaluations
		self.max_generations = max_generations
		self.max_time = max_time

		self.initialization_max_tree_height = initialization_max_tree_height
		self.min_depth = min_depth
		self.max_tree_size = max_tree_size
		self.tournament_size = tournament_size

		self.penalize_duplicates = penalize_duplicates

		self.num_generations = 0

		self.verbose = verbose
		self.log_folder = log_folder


	def __ShouldTerminate(self):
		must_terminate = False
		elapsed_time = time.time() - self.start_time
		if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
			must_terminate = True
		elif self.max_generations > 0 and self.num_generations >= self.max_generations:
			must_terminate = True
		elif self.max_time > 0 and elapsed_time >= self.max_time:
			must_terminate = True

		if must_terminate and self.verbose:
			print('Terminating at\n\t', 
				self.num_generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t', np.round(elapsed_time,2), 'seconds')

		return must_terminate

	def __PassesConstraints(self, individual):
		too_large = len(individual.GetSubtree()) > self.max_tree_size 
		too_tiny = individual.GetHeight() < self.min_depth
		return not (too_large or too_tiny)



	def __InitializePopulation(self):
		
		self.population = []

		# ramped half-n-half
		curr_max_depth = self.min_depth
		init_depth_interval = self.pop_size / (self.initialization_max_tree_height - self.min_depth + 1)
		next_depth_interval = init_depth_interval

		for i in range( self.pop_size ):
			if i >= next_depth_interval:
				next_depth_interval += init_depth_interval
				curr_max_depth += 1

			g = None
			while g is None:
				g = Variation.GenerateRandomTree( self.functions, self.terminals, curr_max_depth, curr_height=0, 
					method='grow' if np.random.random() < .5 else 'full', min_depth=self.min_depth )
				if not self.__PassesConstraints(g):
					g = None
					continue
			self.fitness_function.Evaluate( g )
			self.population.append( g )


	def __CreateOffspring(self, selected):
		O = []

		for i in range( self.pop_size ):
			o = deepcopy(selected[i])
			if ( random() < self.crossover_rate ):
				donor = selected[ randint( self.pop_size ) ]
				o = Variation.SubtreeCrossover( o, donor )
			if ( random() < self.mutation_rate ):
				o = Variation.SubtreeMutation( o, self.functions, self.terminals, max_height=self.initialization_max_tree_height, min_depth=self.min_depth )
			if ( random() < self.op_mutation_rate ):
				o = Variation.OnePointMutation( o, self.functions, self.terminals )
			
			if not self.__PassesConstraints(o):
				del o
				o = deepcopy( selected[i] )

			self.fitness_function.Evaluate(o)

			O.append(o)

		return O


	def LogInfo(self):
		# population stuff
		info_population = { 
			'prefnot_str_repr': [], 
			'ls_a' : [],
			'ls_b' : [],
			'error': [],
			'interpr': [],
			'interpr_uncertainty': []
		}
		for p in self.population:
			info_population['prefnot_str_repr'].append(str(p.GetSubtree()))
			info_population['ls_a'].append(np.round(p.ls_a,5))
			info_population['ls_b'].append(np.round(p.ls_b,5))
			info_population['error'].append(np.round(p.objectives[0],5))
			info_population['interpr'].append(-np.round(p.objectives[1],5))
			info_population['interpr_uncertainty'].append(np.round(p.interpr_uncertainty,5))

		# time to conclude generation
		generation_time = np.round(time.time() - self.start_time,5)

		# interpretability feedbacks
		self.fitness_function.interp_model.update_info_lock.acquire()
		feedbacks = self.fitness_function.interp_model.feedbacks
		for i in range(len(feedbacks)):
			logtime = float(feedbacks[i]['logtime'])
			feedbacks[i]['logtime'] = np.round(logtime - self.start_time,5)
		# re-set 
		self.fitness_function.interp_model.feedbacks = list()
		self.fitness_function.interp_model.update_info_lock.release()

		# save to file
		log_obj = {}
		log_obj['info_population'] = info_population
		log_obj['generation_time'] = generation_time
		log_obj['feedbacks'] = feedbacks
		log_obj['generations_to_go'] = self.max_generations - self.num_generations

		os.makedirs(self.log_folder, exist_ok=True)
		json.dump(log_obj, open(os.path.join(self.log_folder,"log_generation_"+str(self.num_generations)+".json"), 'w'))



	def Run(self):

		self.start_time = time.time()
		self.__InitializePopulation()	

		while not self.__ShouldTerminate():

			selected = Selection.TournamentSelect(self.population, self.pop_size, tournament_size=self.tournament_size)
			O = self.__CreateOffspring(selected)

			# re-evaluate interpretability of parents because the interpretability estimator might have changed
			if self.fitness_function.use_online_interpretability_model:
				for i in range(len(self.population)):
					self.fitness_function.Evaluate(self.population[i], only_interpretability=True)

			# proceed with NSGA-II-style selection
			PO = self.population+O
			new_population = []
			fronts = self.FastNonDominatedSorting(PO, penalize_duplicates=self.penalize_duplicates)
			self.latest_front = deepcopy(fronts[0])

			curr_front_idx = 0
			while curr_front_idx < len(fronts) and (len(fronts[curr_front_idx]) + len(new_population)) <= self.pop_size:
				self.ComputeCrowdingDistances( fronts[curr_front_idx] )
				for p in fronts[curr_front_idx]:
					new_population.append(p)
				curr_front_idx += 1

			if len(new_population) < self.pop_size:
				# fill-in remaining
				self.ComputeCrowdingDistances( fronts[curr_front_idx] )
				fronts[curr_front_idx].sort(key=lambda x: x.crowding_distance, reverse=True) 

				while len(fronts[curr_front_idx]) > 0 and len(new_population) < self.pop_size:
					new_population.append( fronts[curr_front_idx].pop(0) )	# pop first because they were sorted in desc order

			self.population = new_population
			self.num_generations = self.num_generations + 1

			if self.verbose:
				print ('g:',self.num_generations,'elite obj1:', np.round(self.fitness_function.elite.objectives[0],3), ', size:', len(self.fitness_function.elite.GetSubtree()))
				pass

			# log info
			if self.fitness_function.use_online_interpretability_model:
				self.LogInfo()

		# save neural model
		if self.fitness_function.use_online_interpretability_model:
			self.fitness_function.interp_model.model.save(self.log_folder)

			


	def FastNonDominatedSorting(self, population, penalize_duplicates=False):

		rank_counter = 0
		nondominated_fronts = []
		dominated_individuals = {}
		domination_counts = {}
		current_front = []

		for i in range( len(population) ):
			p = population[i]

			dominated_individuals[p] = []
			domination_counts[p] = 0

			for j in range( len(population) ):
				if i == j:
					continue
				q = population[j]

				if p.Dominates(q):
					dominated_individuals[p].append(q)
				elif q.Dominates(p):
					domination_counts[p] += 1

			if domination_counts[p] == 0:
				p.rank = rank_counter
				current_front.append(p)

		while len(current_front) > 0:
			next_front = []
			for p in current_front:
				for q in dominated_individuals[p]:
					domination_counts[q] -= 1
					if domination_counts[q] == 0:
						q.rank = rank_counter + 1
						next_front.append(q)
			nondominated_fronts.append(current_front)
			rank_counter += 1
			current_front = next_front

		if penalize_duplicates:
			already_seen = set()
			discard_front = []
			sorted_pop = sorted(population, key=lambda x: x.rank)
			for p in sorted_pop: 
				summarized_representation = p.cached_output
				if summarized_representation not in already_seen:
					already_seen.add(summarized_representation)
				else:
					# find p and remove it from its front
					for i, q in enumerate(nondominated_fronts[p.rank]):
						if nondominated_fronts[p.rank][i] == p:
							nondominated_fronts[p.rank].pop(i)
							break
					p.rank = np.inf
					discard_front.append(p)
			if len(discard_front) > 0:
				nondominated_fronts.append(discard_front)
				# fix potentially-now-empty fronts
				nondominated_fronts = [front for front in nondominated_fronts if len(front) > 0]

		return nondominated_fronts


	def ComputeCrowdingDistances(self, front):
		number_of_objs = len(front[0].objectives)
		front_size = len(front)

		for p in front:
			p.crowding_distance = 0

		for i in range(number_of_objs):
			front.sort(key=lambda x: x.objectives[i], reverse=False)

			front[0].crowding_distance = front[-1].crowding_distance = np.inf

			min_obj = front[0].objectives[i]
			max_obj = front[-1].objectives[i]

			if min_obj == max_obj:
				continue

			for j in range(1, front_size - 1):

				if np.isinf(front[j].crowding_distance):
					# if extrema from previous sorting
					continue

				prev_obj = front[j-1].objectives[i]
				next_obj = front[j+1].objectives[i]

				front[j].crowding_distance += (next_obj - prev_obj)/(max_obj - min_obj)
