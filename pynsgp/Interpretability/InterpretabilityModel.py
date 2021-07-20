from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import scale
import numpy as np
import threading, time, logging, json, os
from pynsgp.Utils import GenerateTreeFromPrefixNotationString
import sympy

# this must match with the number of features
# of interest for the online model
N_FEATURES_ONLINE = 6

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )

class OfflinePHIModel:
    def __init__(self, featInterpExtractor):
        self.featInterpExtractor = featInterpExtractor
        # learned from survey study (Virgolin et al. 2020)
        self.coeffs = np.array([-0.00195041, -0.00502375, -0.03351907, -0.04472121]).reshape((1, -1))

    def _ComputePHI(self, features):
        phi = np.sum(np.multiply(features, self.coeffs), axis=1).squeeze()
        return phi

    def PredictInterpretability(self, individual):
        f = self.featInterpExtractor.ExtractInterpretabilityFeatures(individual)
        features_of_interest = [f['n_nodes'], f['n_ops'], f['n_naops'], f['n_nacomp']]
        interp_score = self._ComputePHI(features_of_interest) * 100
        return interp_score


class OnlineModel:

    def __init__(self, featInterpExtractor, k=10,
                 update_interval=0.2, online_model_log_folder="./out"):

        # at least 2 points in memory because we pass pairs
        assert (k > 2)

        self.k = k
        self.X = None
        self.y = None
        # initialization values for normalization
        self.X_mean = 0
        self.y_mean = 0
        self.X_std = 1
        self.y_std = 1
        # keep track of latest points for incremental fit
        self.X_latest = None
        self.y_latest = None

        # stuff to keep track of what models to display to the user
        self.already_seen_outputs = set()
        self.already_seen_features = set()
        self.top_k_uncertain_predicted_individuals = list()
        self.featInterpExtractor = featInterpExtractor

        # stuff useful for logging
        self.feedbacks = list()

        # update info as process in a different thread
        self.update_interval = update_interval
        self.exposing_info_file = os.path.join(online_model_log_folder, "exp_info.json")
        self.feedback_file = os.path.join(online_model_log_folder, "human_feedback.json")
        self.last_info_logtime = None
        self.update_info_thread = threading.Thread(target=self._ExposeInfoAndReadFeedbackFileForUpdates,
                                                   daemon=False)  # TODO not sure if deamon=True is a problem when going out of __init__
        self.update_info_thread.master_thread = threading.current_thread()
        self.update_info_lock = threading.Lock()
        self.update_info_thread.start()

    
    def Update(self, individual_pairs, new_labels):

        new_points = []
        for individual in individual_pairs:
            for i in individual:
                new_point = self._GetFeaturesOfInterest(i)
                new_points.append(new_point)
        new_points = np.array(new_points)

        # update the points
        if self.X is None:
            self.X = new_points
            self.y = new_labels
        else:
            self.X = np.vstack([self.X, new_points])
            self.y = np.hstack([self.y, new_labels])

        self.X_latest = new_points
        self.y_latest = np.array([0, 0])
        self.y_latest[int(new_labels[0])] = 1

        # keep track of means and stds for prediction time
        # we do not want 'predict' to be called while updating this info
        self.update_info_lock.acquire()
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0) + 1e-6
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y) + 1e-6

        # update the model (fit on normalized data, better for gaussian processes)
        self._SpecificFit()

        # also clear (because now outdated) the features for which the estimator is most uncertain
        self.top_k_uncertain_predicted_individuals.clear()
        self.update_info_lock.release()


    def _GetFeaturesOfInterest(self, individual):
        fmap = self.featInterpExtractor.ExtractInterpreabilityFeatures(individual)
        features_of_interest = np.array(
            [fmap['n_nodes'], fmap['n_ops'], fmap['n_naops'], fmap['n_nacomp'], fmap['n_const'], fmap['n_dim']])
        assert (len(features_of_interest) == N_FEATURES_ONLINE)
        return features_of_interest

    ''' returns predictions of the form [intepretability_scores, uncertainty_scores]'''
    def _SpecificPredict(self, X):
        pass

    ''' fits the model '''
    def _SpecificFit(self, X, y):
        pass

   

    def PredictInterpretability(self, individual):

        features = self._GetFeaturesOfInterest(individual).reshape((1, -1))
        normalized_features = (features - self.X_mean) / self.X_std

        # get predictions if fit was called before
        try:
            self.update_info_lock.acquire()
            self.update_info_lock.release()
            predictions = self._SpecificPredict(normalized_features)
        # no interpretability, max uncertainty
        except:
            predictions = np.array([[0.0], [1.0]])

        interp_score = predictions[0][0] * self.y_std + self.y_mean
        uncert_score = predictions[1][0]

        if np.round(individual.objectives[0],6) not in self.already_seen_outputs:
          self.already_seen_outputs.add(individual.cached_output)
          if str(normalized_features) not in self.already_seen_features:
              self.already_seen_features.add(str(normalized_features))
              self.top_k_uncertain_predicted_individuals.append((individual, uncert_score, interp_score))
              if len(self.top_k_uncertain_predicted_individuals) > self.k:
                  temp = sorted(self.top_k_uncertain_predicted_individuals, key=lambda x: x[1], reverse=True)
                  # if they are all uncertain the same, shuffle
                  if (temp[0][1] == temp[self.k - 1][1]):
                      np.random.shuffle(temp)
                  # retain top k that are most uncertain
                  self.top_k_uncertain_predicted_individuals = temp[0:self.k]

        return interp_score, uncert_score
      

    def _ExposeInfoAndReadFeedbackFileForUpdates(self):
        # part 0: if the master thread is dead, kill myself
        if not self.update_info_thread.master_thread.is_alive():
          # TODO: is quit() the best way?
          quit()

        # part 1: expose info for the user
        exp_info = []
        for item in self.top_k_uncertain_predicted_individuals:
            str_repr = item[0].GetLatexExpression()
            nodes = item[0].GetSubtree()
            n_components = len(nodes)
            prefnot_str_repr = str(nodes).replace(']', '').replace('[', '').replace(' ', '')
            uncert_score = np.round(item[1], 3)
            interp_score = np.round(item[2], 3)
            info_item = {'str_repr': str_repr, 'prefnot_repr': prefnot_str_repr, 'uncertainty': uncert_score,
                         'interpretability': interp_score, 'n_components': n_components}
            exp_info.append(info_item)
        json.dump(exp_info, open(self.exposing_info_file, 'w'))

        # part 2: scan for feedback
        if os.path.exists(self.feedback_file):
            try:
              info = json.load(open(self.feedback_file))
            except:
              # it was being written by the interface, try again soon
              # back to sleep for a brief time
              time.sleep(self.update_interval)
              # call myself again
              self._ExposeInfoAndReadFeedbackFileForUpdates()

            if self.last_info_logtime is None or self.last_info_logtime != info['logtime']:
                self.last_info_logtime = info['logtime']
                self.feedbacks.append(info)
                # update stuff
                individual_pairs = []
                labels = []  # 0 if first is better, 1 if second is better
                for item in info["feedback"]:
                    individual1 = GenerateTreeFromPrefixNotationString(item["prefnot_repr1"],
                                                                       self.featInterpExtractor.function_set,
                                                                       self.featInterpExtractor.terminal_set)
                    individual2 = GenerateTreeFromPrefixNotationString(item["prefnot_repr2"],
                                                                       self.featInterpExtractor.function_set,
                                                                       self.featInterpExtractor.terminal_set)
                    label = float(item["label"])
                    individual_pairs.append((individual1, individual2))
                    labels.append(label)
                # now we have ordered invidual-label pairs from the feedback
                self.Update(individual_pairs, labels)
        # back to sleep
        time.sleep(self.update_interval)
        # call myself again
        self._ExposeInfoAndReadFeedbackFileForUpdates()



import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras import optimizers
import tensorflow as tf


class OnlineRankingNetModel(OnlineModel):

    def __init__(self, featInterpExtractor, k=3, update_interval=0.2, warm_start_n_epochs=20, warm_start_n_obs=100,
                 batch_size=2, n_epochs=4, n_trials=10, n_hidden_layers=3, n_act_per_layer=100, rate_dropout=0.25,
                 learning_rate=1e-3, online_model_log_folder='./out'):
        super().__init__(featInterpExtractor=featInterpExtractor, k=k, update_interval=update_interval,
                         online_model_log_folder=online_model_log_folder)
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.model, self.optimizer = self.DefineModel(N_FEATURES_ONLINE, n_hidden_layers=n_hidden_layers,
                                                      n_act_per_layer=n_act_per_layer, rate_dropout=rate_dropout,
                                                      )
        if warm_start_n_epochs is not None:
            self._WarmStart(featInterpExtractor, warm_start_n_epochs, warm_start_n_obs)

    def DefineModel(self, n_features, rate_dropout=0.25,
                    activation="relu", n_hidden_layers=3, n_act_per_layer=100):
        tinput = Input(shape=(n_features,), name="ts_input")
        network = tinput
        for i in range(n_hidden_layers):
            network = Dense(n_act_per_layer, activation='relu')(network)
            network = Dropout(rate_dropout)(network)
        activation = "tanh"
        out = Dense(1, activation=activation)(network)
        model = Model(inputs=[tinput], outputs=out)
        optimizer = optimizers.SGD(lr=1e-3, momentum=0.9)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        return model, optimizer

    def _SpecificPredict(self, X):
        f = Model(self.model.layers[0].input, self.model.layers[-1].output)
        result = []
        for i in range(self.n_trials):
            result.append(f(X, training=True))
        y_pred_do = np.array(result, dtype=float).reshape(self.n_trials, len(X))
        mean_pred = np.mean(y_pred_do, axis=0)
        std_pred = np.std(y_pred_do, axis=0)
        return [mean_pred, std_pred]

    def _SpecificFit(self):
        X = (self.X_latest - self.X_mean)/self.X_std
        y = (self.y_latest - self.y_mean)/self.y_std

        weights = np.ones_like(y)
        weights[np.argmax(y)] = -1
        with tf.GradientTape() as tape:
            output = self.model(X, training=True)
            loss = weights[0] * output[0] + weights[1] * output[1]
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


    def _WarmStart(self, featInterpExtractor, warm_start_n_epochs, warm_start_n_obs):
        # create a data set
        n_nodes = np.random.randint(low=1, high=31, size=(warm_start_n_obs))
        n_ops = np.random.randint(low=0, high=15, size=(warm_start_n_obs))
        n_naops = np.random.randint(low=0, high=7, size=(warm_start_n_obs))
        n_nacomp = np.random.randint(low=0, high=7, size=(warm_start_n_obs))
        X = np.stack((n_nodes, n_ops, n_naops, n_nacomp), axis=1)

        # X = np.vstack((X, np.zeros(4).reshape(1,-1))) # to debug phi
        # use offline phi model to get scores
        ofm = OfflinePHIModel(featInterpExtractor)
        y = 10 + ofm._ComputePHI(X) * 10

        # we now add the number of feat necessary for the input layer of the net
        # (even if not used)
        for i in range(N_FEATURES_ONLINE - X.shape[1]):
            some_other_feature = np.random.randint(low=0, high=10, size=(warm_start_n_obs)).reshape((-1, 1))
            X = np.hstack((X, some_other_feature))

        # set up warmup inputs and outputs
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
        y = (y - np.mean(y)) / (np.std(y) + 1e-6)

        # fit the net
        if warm_start_n_epochs == 'auto':
          # we warmup until the net has a decent level of understanding (passes the peak of uncertainty growth)
          uncertainties = list()
          for i in range(10): # use at most 10 epochs (haven't seen it being reached TBH)
              self.model.fit(X, y, epochs=1, verbose=False)
              pred = self._SpecificPredict(X)
              std_pred = pred[1]
              uncertainties.append(np.mean(std_pred))
              if len(uncertainties) < 4:
                  continue
              latest = np.mean(uncertainties[-2:-1])
              before = np.mean(uncertainties[-4:-2])
              if latest < before:
                  print('warmed-up on phi for',i,'epochs')
                  break
        else:
          self.model.fit(X, y, epochs=warm_start_n_epochs, verbose=False)
