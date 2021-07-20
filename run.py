#!/usr/bin/env python

# Libraries
import numpy as np 
import os, sys, shutil, json
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SS
from copy import deepcopy

# Internal imports
from pynsgp.Nodes.BaseNode import Node
from pynsgp.Nodes.SpecificNodes import *
from pynsgp.Evolution.Evolution import pyNSGP

from pynsgp.SKLearnInterface import pyNSGPEstimator as NSGP

os.chdir('../')

INTERP_MODEL = 'online'

# set large recursion limit
sys.setrecursionlimit(10**7) 

# get random seeds which we then store if we need to reproduce results
data_split_seed = np.random.randint(9999)
evo_seed = np.random.randint(9999)

np.random.seed(evo_seed)
problem_name = sys.argv[1]
user_id = sys.argv[2]

print(problem_name)


online_model_log_folder = "./interface/sessions/"+user_id+"/"
os.makedirs(online_model_log_folder, exist_ok=True)
log_folder = "./interface/sessions/"+user_id+"/out/"+problem_name
if os.path.exists(log_folder):
  # if it's the user and there was content, delete that content
  shutil.rmtree(log_folder)
os.makedirs(log_folder, exist_ok=True)


error_metric = 'mse'
# Load regression dataset 
if problem_name == "boston":
  error_metric = 'mse'
  X, y = sklearn.datasets.load_boston( return_X_y=True )
elif problem_name == "german":
  error_metric = 'binary_acc'
  Xy = np.loadtxt("./data/german.csv", delimiter=',')
  X = Xy[:, 0:Xy.shape[1]-1]
  y = Xy[:, -1]


# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=data_split_seed )
# touch a file that tells us the seed for the data split
with open(log_folder+"/seed.txt", "w") as file:
  file.write('{ data_split:' +str(data_split_seed) + ', evolution:' +str(evo_seed) + ' }\n')

# scale features
scaler = SS()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Prepare NSGP settings

# Terminal set
n_features = X_train.shape[1]
terminal_set = list()
for i in range(n_features):
  terminal_set.append(FeatureNode(i)) # one feature node for each column of the training set
# do you want to use ERC?
terminal_set.append(EphemeralRandomConstantNode())

# Function set
function_set = [AddNode(), SubNode(), MulNode(), DivNode(), CubeNode(), LogNode(), MaxNode()]


nsgp = NSGP(pop_size=256, max_generations=50, 
  verbose=True, 
  log_folder=log_folder,
  online_model_log_folder=online_model_log_folder,
  error_metric=error_metric,
  max_tree_size=25, crossover_rate=0.334, mutation_rate=0.333, op_mutation_rate=0.333, 
  min_depth=1, initialization_max_tree_height=3, 
	tournament_size=2, use_linear_scaling=True,
  use_offline_interpretability_model=INTERP_MODEL == 'phi',
  use_online_interpretability_model=INTERP_MODEL == 'online',
  online_phi_warmup_epochs='auto',
  penalize_duplicates=True,
	functions = function_set,
  terminals = terminal_set
)

# Fit like any sklearn estimator
nsgp.fit(X_train,y_train)

# Obtain the front of non-dominated solutions (according to the training set)
front = sorted(nsgp.get_front(), key=lambda x: x.objectives[0], reverse=True)

# Now we find formulas at certain quantiles to prepare the survey
if INTERP_MODEL == 'online':
  quantiles = [.5, .7, .9]
  # re-adjust quantiles to the front size
  chosen_ones = list()
  indices_before = list()
  for q in quantiles:
    idx = int(q*len(front))
    # let's try to avoid selecting a same model twice
    if len(indices_before) > 0 and idx == indices_before[-1]:
      # if we can, shift this index by one ahead
      if idx < len(front)-1:
        idx += 1
      # else, shift all previous ones back by one
      elif indices_before[0] > 0:
        for i in range(len(indices_before)):
          indices_before[i] = indices_before[i] - 1
      # if nothing works, well whoops, we'll present the same twice
      # (there's a check below to avoid a duplicate in the comparison anyway)
    indices_before.append(idx)

  for idx in indices_before:
    chosen_ones.append(front[idx])

  # for each of the chosen ones, we retrive a similar one from archive of phi and size
  other_models = json.load(open('data/other_models.json'))
  phi_models = other_models[problem_name]['phi']
  size_models = other_models[problem_name]['size']

  def find_closest_model_by_error(target, models_w_errors):
    chosen = None
    min_dist = np.inf
    for item in models_w_errors:
      are_same = item["prefnot_repr"] == target["prefnot_repr"]
      dist = np.abs(item["error"] - target["error"])
      if not are_same and dist < min_dist:
        min_dist = dist
        chosen = item
    return chosen

  model_triplets = list()
  for model in chosen_ones:
    target = {"error": model.objectives[0], "prefnot_repr": str(model.GetSubtree())}
    closest_phi = find_closest_model_by_error(target, phi_models)
    # remove this model from phi_models so that, if we have identical chosen_ones from the user runs', 
    # at least we're going to have non-identical model from the premade runs
    if len(phi_models) > 1:
      phi_models.remove(closest_phi)

    closest_size = find_closest_model_by_error(target, size_models)
    # similarly remove this guy
    if len(size_models) > 1:
      size_models.remove(closest_size)
    #print(len(size_models), closest_size['prefnot_repr'], closest_size['str_repr'], closest_size['error'], target['error'])

    model_triplets.append({
      'online': model.GetLatexExpression(), 
      'phi': closest_phi["str_repr"], 
      'size': closest_size["str_repr"],
      })
  # save 
  json.dump(model_triplets, open(log_folder+'/survey_models.json', 'w'))
else: 
  # in this case we just want to save results
  front = sorted(nsgp.get_front(), key=lambda x: x.objectives[0])
  to_log = {
    'str_repr': list(),
    'prefnot_repr' : list(),
    'ls_a' : list(),
    'ls_b' : list(),
    'error': list(),
    'interpr': list(),
  }
  for model in front:
    to_log['str_repr'].append(model.GetLatexExpression())
    to_log['prefnot_repr'].append(str(model.GetSubtree()))
    to_log['ls_a'].append(model.ls_a)
    to_log['ls_b'].append(model.ls_b)
    to_log['error'].append(model.objectives[0])
    to_log['interpr'].append(-model.objectives[1])
  json.dump(to_log, 
    open('interface/out/'+problem_name+'/finalfront_'+INTERP_MODEL+'_seeddatasplit_'+str(data_split_seed)+'_seedevolution_'+str(evo_seed)+'.json', 
      'w'))
  os.remove('interface/out/'+problem_name+'/seed.txt')