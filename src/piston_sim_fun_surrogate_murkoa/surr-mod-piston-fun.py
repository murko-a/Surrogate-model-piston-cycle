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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from smt.sampling_methods import LHS