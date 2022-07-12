"""
Author: Anze Murko <anze.murko@rwth-aachen.com>

This package is distributed under MIT license.
Piston simulation function problem from:
Surjanovic, S., Bingham, D. Virtual Library of Simulation Experiments: Test Functions and Datasets,
Simon Fraser University, accessed 18 June 2022, <https://www.sfu.ca/~ssurjano/piston.html>.

"""
import os
import sys
sys.path.append("mod")
from mod.param_pred_true import param_true_pred_fun
from mod.predict_mult_mod import predict_mult_fun
from mod.predict_mod import predict_fun
from mod.performance_mod import performance_fun
from mod.compare_true_pred_mod import compare_true_pred_fun
from mod.parameter_depend import param_depend_fun
from mod.show_folds_mod import show_folds_fun
from smt.sampling_methods import LHS
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import yaml


class SurrogateModel():
    """Surrogate modelling of piston simulation function

This module is created to develop surrogate model of piston simulation function, to evaluate
    prediction of the function result value, with defined parameters, faster that in general
    calculation methods. Module also performs performance overview of the used surrogate modeling
    methods and enables to compare true and predicted values of the function and also compare that
    with defined parameter values.

    Example:
                    $ from piston_fun_sm import SurrogateModel
                    $ path = r"C:\\..\\param_data.yml"
                    $ sm = SurrogateModel(file_name = path, samples = 200, n_splt = 5,
                                                                     shuffle= True, rand_state = True)
                    $ sm.performance("RFR","SVR", "MLR", plot_type = "boxplot")

Attributes:
            file_name : str, default = "data/param_data.yml"
                    File path of .yml file with defined limits of parameters or .csv file with defined
                    values of parameters --> look at example_parm.csv in "data/" folder
            samples : int, default = 1000
                    Number of created samples by LHS method, if you provide limits of parameters in .yml file format.
            n_splt : int, default = 5
                    Number of folds that KFold method should apply on dataset. Must be at least 2.
            shuffle : bool, default=False
                    Whether to shuffle the data before splitting into batches.
                    Note that the samples within each split will not be shuffled
            rand_state : int, default=None
                    When shuffle is True, random_state affects the ordering of the indices,
                    which controls the randomness of each fold. Otherwise, this parameter has no effect.

"""

    def __init__(
            self,
            file_name=None,
            samples=1000,
            n_splt=5,
            shuffle=False,
            rand_state=None):
        self.file_name = file_name or os.path.join(
            os.getcwd(), "data/param_data.yml")
        split_tup = os.path.splitext(self.file_name)

        try:
            if split_tup[1] == ".yml":
                with open(self.file_name, "r") as cfg_file:
                    self.config_yaml = yaml.safe_load(cfg_file)
                    self.config_yaml = self.config_yaml["surrogate-model-limits"]
            elif split_tup[1] == ".csv":
                self.config_csv = pd.read_csv(
                    file_name, delimiter=",", header=None, names=[
                        "M", "S", "V_0", "k", "P_0", "T_a", "T_0"])

        except FileNotFoundError as fe:
            print("Config not found")
            sys.exit(2)

        if hasattr(self, "config_yaml"):
            self.xlimits = np.array([self.config_yaml["M"],
                                     self.config_yaml["S"],
                                     self.config_yaml["V_0"],
                                     self.config_yaml["k"],
                                     self.config_yaml["P_0"],
                                     self.config_yaml["T_a"],
                                     self.config_yaml["T_0"]])
            self.sampling = LHS(xlimits=self.xlimits)
            self.x = self.sampling(samples)
            self.dataset = pd.DataFrame(
                self.x,
                columns=[
                    "M",
                    "S",
                    "V_0",
                    "k",
                    "P_0",
                    "T_a",
                    "T_0"])
        elif hasattr(self, "config_csv"):
            self.dataset = pd.DataFrame(self.config_csv)

        self.dataset["A"] = self.dataset["P_0"] * self.dataset["S"] + 19.62 * \
            self.dataset["M"] - (self.dataset["k"] * self.dataset["V_0"]) / self.dataset["S"]
        self.dataset["V"] = self.dataset["S"] / (2 * self.dataset["k"]) * (np.sqrt(self.dataset["A"]**2 + 4 * self.dataset["k"] * (
            self.dataset["P_0"] * self.dataset["V_0"] * self.dataset["T_a"]) / self.dataset["T_0"]) - self.dataset["A"])
        self.dataset["C"] = 2 * np.pi * np.sqrt(self.dataset["M"] / (self.dataset["k"] + (
            self.dataset["S"]**2 * self.dataset["P_0"] * self.dataset["V_0"] * self.dataset["T_a"]) / (self.dataset["T_0"] * self.dataset["V"]**2)))

        self.X = self.dataset.iloc[:, 0:7].values
        self.y = self.dataset.iloc[:, 9].values

        self.kfold = KFold(
            n_splits=n_splt,
            shuffle=shuffle,
            random_state=rand_state)

        for train_ix, test_ix in self.kfold.split(self.X):
            self.train_X, self.test_X = self.X[train_ix], self.X[test_ix]
            self.train_y, self.test_y = self.y[train_ix], self.y[test_ix]

        self.models = {"RFR": RandomForestRegressor(),
                       "SVR": SVR(),
                       "MLR": LinearRegression(),
                       "KNR": KNeighborsRegressor()
                       }

    def show_folds(self):
        """
        Function shows number of folds, generated by KFold option from sklearn
        python pacakage, and number of training and testing data in every fold.

        Args:
                /
        Returns:
                Number of folds and number of trained/test data in every fold.

        """
        return show_folds_fun(self)

    def param_depend(self, plot_type="all"):
        """Comparison of true and predicted values function.

        Function takes plot_type which defines which input/output variable dependency
        should be ploted.

        Args:
            plot_type (sting): returns true/predicted values vs. parameter plot.
                                Default is "all". Plot options:
                                    ["all","M","S", "V_0", "k", "P_0", "T_a", "T_0"]

        Returns:
        Plot of dependency of output values regarding the input parameters.

        Raises:
            KeyError: If used plot type in `plot_type` parameter is not defined
                            in package description..

        """
        if plot_type not in ["all", "M", "S", "V0", "k", "P0", "Ta", "T0"]:
            raise KeyError("Defined plot type not in this package.")
        else:
            return param_depend_fun(self, plot_type=plot_type)

    def predict(self, *models, predict_data):
        """Quick prediction function.

        Function takes user-defined models from models option list and list
        of defined values of parameters and calculate true and predicted values of
        piston cycle time for every specified surrogate model.

        Args:
                predict_data (list): List of parameter values to predict piston
                        cycle time, by defined models in *models argument. List of
                        parameters should be organized as:
                        ["M","S", "V_0", "k", "P_0", "T_a", "T_0"].

                *models: Surrogate models argument list.
                                Options:   "RFR" - Random Forrest Regression,
                                                        "MLR" - Multiple Linear Regression,
                                                        "SVR" - Support Vector Regression,
                                                        "KNR" - K Nearest Neighbour Regression

        Returns:
                Calculated true and predicted values of piston cycle time.

        Raises:
                KeyError: If item in `models` is not defined in package description.
                Exception: If there is not enough parameter defined in "predict_data", data is not shape (,7)

        """
        for model in models:
            if model not in ["RFR", "MLR", "SVR", "KNR"]:
                raise KeyError("Defined models not in this package.")
            else:
                pass
        if len(predict_data) != 7:
            raise Exception(
                "There is not enough parameters defined in predict_data. Parameter \
				data to perform prediction is not right shape, it must be shape-like (, 7).")
        else:
            return predict_fun(self, *models, predict_data=predict_data)

    def predict_multiple(self, *models, predict_data):
        """Multiple prediction function.

        Function takes user-defined models from models option list and
        multiple dimension array of defined values of parameters and
        calculate predicted values of piston cycle time for every
        specified surrogate model.

        Args:
                predict_data (array-like): multiple dimension array of parameter values
                        to predict piston cycle time, by defined models in *models argument.
                        Sub-array should have parameters organized as:
                        ["M","S", "V_0", "k", "P_0", "T_a", "T_0"].

                *models: Surrogate models argument list.
                                Options:   "RFR" - Random Forrest Regression,
                                                        "MLR" - Multiple Linear Regression,
                                                        "SVR" - Support Vector Regression,
                                                        "KNR" - K Nearest Neighbour Regression

        Returns:
                Calculated predicted values of piston cycle time for every
                specified surrogate model.

        Raises:
                KeyError: If item in `models` is not defined in package description.
                TypeError: If `predict_data` is not type of numpy.ndarray.
                Exception: If there is not enough parameter defined in "predict_data", data is not shape (,7)

        """
        for model in models:
            if model not in ["RFR", "MLR", "SVR", "KNR"]:
                raise KeyError("Defined models not in this package.")
            else:
                pass
        if not isinstance(predict_data, np.ndarray):
            raise TypeError(
                "Parameter data to perform prediction is not type of numpy.ndarray")
        if predict_data.shape[1] != 7:
            raise Exception(
                "There is not enough parameters defined in predict_data. Parameter \
				data to perform prediction is not right shape, it must be shape-like (, 7).")
        else:
            return predict_mult_fun(self, *models, predict_data=predict_data)

    def performance(self, *models, perf_df=True, plot_type=None):
        """Model performance evaluation function.

        Function takes user-defined models from models option list,
        boolean argument perf_df which returns performance dataframe if enabled,
        function uses parameter test data defined by running the class to predict
        and perform performance evaluation on that prediction. With plot_perf
        argument can be defined which performance plot should function returns.

        Args:
                *models: Surrogate models argument list.
                        Options:   "RFR" - Random Forrest Regression,
                                                "MLR" - Multiple Linear Regression,
                                                "SVR" - Support Vector Regression,
                                                "KNR" - K Nearest Neighbour Regression

                perf_df (bool): returns performance dataframe. Default is True.

                plot_type (str): returns performance plot. Default is None.
                        Plot options: ["boxplot", "lolipop", "spyderweb"]

        Returns:
                Calculated performance of selected models and (if enabled) performance plot.

        Raises:
                KeyError: If item in `models` is not defined in package description.
                KeyError: If used plot type in `plot_type` parameter is not defined
                                in package description.

        """
        for model in models:
            if model not in ["RFR", "MLR", "SVR", "KNR"]:
                raise KeyError("Defined models not in this package.")
            else:
                pass
        if plot_type not in ["boxplot", "lolipop", "spyderweb"]:
            raise KeyError("Defined plot type not in this package.")
        else:
            return performance_fun(
                self, *models, perf_df=perf_df, plot_perf=plot_type)

    def compare_true_pred(self, *models, plot=True, results=True):
        """Comparison of true and predicted values function.

        Function takes user-defined models from models option list,
        boolean argument plot which returns comparison plot if enabled,
        and boolean argument results which if enabled, returns dataframe of
        parameter space generated with running the class, with predicted
        values of defined models and calculated true values.

        Args:
                *models: Surrogate models argument list.
                        Options:   "RFR" - Random Forrest Regression,
                                                "MLR" - Multiple Linear Regression,
                                                "SVR" - Support Vector Regression,
                                                "KNR" - K Nearest Neighbour Regression

                plot (bool): returns plot of true vs. predicted values plot.
                                         Default is True.

                results (bool): returns dataframe of parameters, true and predicted values.
                                                 Default is True.

        Returns:
                Dataframe of parameter space, calculated true values and predicted values
                of selected models and comparison true vs. predicted values plot.

        Raises:
                KeyError: If item in `models` is not defined in package description.

        """
        for model in models:
            if model not in ["RFR", "MLR", "SVR", "KNR"]:
                raise KeyError("Defined models not in this package.")
            else:
                pass
        else:
            return compare_true_pred_fun(
                self, *models, plot=plot, results=results)

    def param_true_pred(self, *models, plot_type="all", results=True):
        """Comparison of true and predicted values function.

        Function takes user-defined models from models option list,
        boolean argument plot which returns comparison plot if enabled,
        and boolean argument results which if enabled, returns dataframe of
        parameter space generated with running the class, with predicted
        values of defined models and calculated true values.

        Args:
                *models: Surrogate models argument list.
                                Options:   "RFR" - Random Forrest Regression,
                                                        "MLR" - Multiple Linear Regression,
                                                        "SVR" - Support Vector Regression,
                                                        "KNR" - K Nearest Neighbour Regression

                plot_type (sting): returns true/predicted values vs. parameter plot.
                                                        Default is "all". Plot options:
                                                                ["all","M","S", "V0", "k", "P0", "Ta", "T0"]


                results (bool): returns dataframe of parameters, true and predicted values.
                                                 Default is True.

        Returns:
                Dataframe of parameter space, calculated true values and predicted values
                of selected models and  true/predicted values vs. parameter plot.

        Raises:
                KeyError: If item in `models` is not defined in package description.
                KeyError: If used plot type in `plot_type` parameter is not defined
                                in package description.

        """
        for model in models:
            if model not in ["RFR", "MLR", "SVR", "KNR"]:
                raise KeyError("Defined models not in this package.")
            else:
                pass
        if plot_type not in ["all", "M", "S", "V0", "k", "P0", "Ta", "T0"]:
            raise KeyError("Defined plot type not in this package.")
        else:
            return param_true_pred_fun(
                self, *models, plot_type=plot_type, results=results)
