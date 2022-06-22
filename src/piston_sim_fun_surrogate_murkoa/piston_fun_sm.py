'''
Author: Anze Murko <anze.murko@outlook.com>

This package is distributed under MIT license.
Piston simulation function problem from:
Surjanovic, S., Bingham, D. Virtual Library of Simulation Experiments: Test Functions and Datasets,
Simon Fraser University, accessed 18 June 2022, <https://www.sfu.ca/~ssurjano/piston.html>.

'''

import os
import sys
import yaml
import csv
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from smt.sampling_methods import LHS
import sys

sys.path.append('mod')
from show_folds_mod import show_folds_fun
from compare_true_pred_mod import compare_true_pred_fun
from performance_mod import performance_fun
from predict_mod import predict_fun

class SurrogateModel():
	'''
    A class to develop surrogate model of piston simulation function.

    ...

    Attributes
    ----------
    file_name : str, default = 'data/param_data.yml'
        File path of .yml file with defined limits of parameters or .csv file with defined
		values of parameters --> look at example_parm.csv in 'data/' folder
    n_splt : int, default = 5
        Number of folds that KFold method should apply on dataset. Must be at least 2.
    rand_state : int, default=None
		When shuffle is True, random_state affects the ordering of the indices, 
		which controls the randomness of each fold. Otherwise, this parameter has no effect.
    shuffle : bool, default=False
		Whether to shuffle the data before splitting into batches.
		Note that the samples within each split will not be shuffled    

    Methods
    -------
	show_folds_class():
		Returns number of KFold() generated folds --> number of splits.

	def predict(*mdls, predict_data):
		Takes predict_data dataset and returns predicted by models defined in *mdls

	def performance(*mdls_pf, perf_df = False, predict_data = None, plot_perf = None):
		Takes trained data defined by defining SurrogateModel() and perform a performance of 
		surrogate models in *mdls_pf. Evaluation of performance can be done with defined predict_data
		dataset or by default test data	defined by defining SurrogateModel().
		Performance evaluation results can be presented by enabling perf_df argument.
		Performance evaluation results can be ploted with argument 
		plot_perf --> {'box-plot' or 'lolipop-plot'}.
	
	def compare_true_pred(*mdls_pf, plot_type = None, df = False):
		Takes trained data and test data defined by defining SurrogateModel() and surrogate models
		in *mdls_pf and perform comparison of prediction results and true values of defined 
		surrogate models. Argument plot_type has multiple options for parameter vs. piston cycle time
		comparison --> {'M','S', 'V0', 'k', 'P0', 'Ta', 'T0'} or option 'true-pred' which returns
		comparison of true values vs predicted values. With enabling df argument, 
		prediction results can be obtained.
	
    '''
	def __init__(self, file_name = None, n_splt=5, rand_state = None, shuffle = False):
		self.file_name = file_name or os.path.join(os.getcwd(), 'data/param_data.yml')
		split_tup = os.path.splitext(self.file_name)
		
		try:
			if split_tup[1] == '.yml':
				with open(self.file_name,'r') as cfg_file:
						self.config_yaml = yaml.safe_load(cfg_file)
						self.config_yaml = self.config_yaml['surrogate-model-limits']
			if split_tup[1] == '.csv':
					with open(self.file_name,'r') as cfg_file:
						self.config_csv = pd.read_csv(cfg_file, delimiter=' ')
		except FileNotFoundError as fe: 
			print("Config not found")
			sys.exit(2)
		
		if hasattr(self, 'config_yaml'):
			self.xlimits = np.array([self.config_yaml['M'], self.config_yaml['S'], self.config_yaml['V_0'], self.config_yaml['k'],
			self.config_yaml['P_0'], self.config_yaml['T_a'], self.config_yaml['T_0']])
			self.sampling = LHS(xlimits = self.xlimits)
			self.x = self.sampling(1000)
			self.dataset = pd.DataFrame(self.x, columns=['M','S', 'V_0', 'k', 'P_0', 'T_a', 'T_0'])
		elif hasattr(self, 'config_csv') :
			self.dataset = pd.DataFrame(self.config_csv, columns=['M','S', 'V_0', 'k', 'P_0', 'T_a', 'T_0'])

		self.dataset['A'] = self.dataset['P_0']*self.dataset['S'] + 19.62*self.dataset['M'] - (self.dataset['k']*self.dataset['V_0'])/self.dataset['S']
		self.dataset['V'] = self.dataset['S']/(2*self.dataset['k'])*(np.sqrt(self.dataset['A']**2 
									+ 4*self.dataset['k']*(self.dataset['P_0']*self.dataset['V_0']*self.dataset['T_a'])/self.dataset['T_0'])
													- self.dataset['A'])
		self.dataset['C'] = 2*np.pi * np.sqrt(self.dataset['M']/(self.dataset['k']
													+ (self.dataset['S']**2*self.dataset['P_0']*self.dataset['V_0']*self.dataset['T_a'])
													/(self.dataset['T_0']*self.dataset['V']**2)))
		
		self.X = self.dataset.iloc[:, 0:7].values
		self.y = self.dataset.iloc[:, 9].values

		self.kfold = KFold(n_splits=n_splt, shuffle=True, random_state=rand_state, shuffle=shuffle)

		for train_ix, test_ix in self.kfold.split(self.X):
			self.train_X, self.test_X = self.X[train_ix], self.X[test_ix]
			self.train_y, self.test_y = self.y[train_ix], self.y[test_ix]

		self.models = {'RFR': RandomForestRegressor(),
						'SVR': SVR(),
						'LR': LinearRegression(),
						'KNR': KNeighborsRegressor()
							}

	def show_folds(self):
		return show_folds_fun(self)	

	def predict(self, *mdls, predict_data):
		return predict_fun(self, *mdls, predict_data = predict_data)
		
	def performance(self, *mdls_pf, perf_df=False, predict_data = None, plot_perf = None):
		performance_fun(self, *mdls_pf, perf_df = perf_df, predict_data = predict_data, plot_perf = plot_perf)

	def compare_true_pred(self, *mdls_pf, plot_type = None, df = False):
		return compare_true_pred_fun(self, *mdls_pf, plot_type = plot_type, df = df)
