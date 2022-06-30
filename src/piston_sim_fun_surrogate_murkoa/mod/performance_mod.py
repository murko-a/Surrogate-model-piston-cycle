from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from IPython.display import display
from tabulate import tabulate
from lolipop_plot import plot_lolipop
from box_plot import plot_boxplot
from performance_calc import perf_calc

def performance_fun(self, *models, perf_df, plot_perf):
    """Model performance evaluation function.

    Function takes user-defined models from models option list,
    boolean argument perf_df which returns performance dataframe if enabled,
    function uses parameter test data defined by running the class to predict 
    and perform performance evaluation on that prediction. With plot_perf 
    argument can be defined which performance plot should function returns. 

    Args:
        *models: Surrogate models argument list. 
            Options:   "RFR" - Random Forrest Regression,
                        "LR" - Linear Regression,
                        "SVR" - Support Vector Regression,
                        "KNR" - K Nearest Neighbour Regression

        perf_df (bool): returns performance dataframe. Default is True.

        plot_perf (str): returns performance plot. Default is None.
            Plot options: ["boxplot", "lolipop"]

    Returns:
        Calculated performance of selected models and (if enabled) performance plot.

    """
    self.mdls_pf = models
    performance_df, results_mae, results_mse, results_rmse, results_r2 = perf_calc(self)

    if perf_df == True:
        print(tabulate(performance_df, headers = 'keys', tablefmt = 'psql'))

    ordered_data_acc = performance_df.sort_values(by='Accuracy', ascending=True)
    ordered_data_mae = performance_df.sort_values(by='mean(MAE)', ascending=True)
    ordered_data_mse = performance_df.sort_values(by='mean(MSE)', ascending=True)
    ordered_data_rmse = performance_df.sort_values(by='mean(RMSE)', ascending=True)
    ordered_data_r2 = performance_df.sort_values(by='mean(R2)', ascending=True)
    ordered_data_time = performance_df.sort_values(by='time', ascending=True)
    

    def plot_performance_lolipop(self, ordered_data_acc, ordered_data_mae, ordered_data_mse, 
    ordered_data_rmse, ordered_data_r2, ordered_data_time):
        """
		Function takes ordered performance results of accuracy, MAE, MSE, RMSE, R2
        and time and returns lolipop plot of that results.

		"""
        return plot_lolipop(self, od_acc = ordered_data_acc, od_mae = ordered_data_mae, od_mse = ordered_data_mse, 
            od_rmse = ordered_data_rmse, od_r2 = ordered_data_r2, od_time = ordered_data_time)

    def plot_performance_boxplot(self,results_mae, results_mse, results_rmse, results_r2):
        """
		Function takes performance results of MAE, MSE, RMSE, R2 and 
        returns boxplot plot of that results.
        
		"""
        return plot_boxplot(self,res_mae = results_mae, res_mse = results_mse, res_rmse = results_rmse, res_r2 = results_r2)

    if plot_perf == 'lolipop':
        plot_performance_lolipop(self, ordered_data_acc, ordered_data_mae, ordered_data_mse,
        ordered_data_rmse, ordered_data_r2, ordered_data_time)

    elif plot_perf == 'boxplot':
        plot_performance_boxplot(self, results_mae, results_mse, results_rmse, results_r2)

    