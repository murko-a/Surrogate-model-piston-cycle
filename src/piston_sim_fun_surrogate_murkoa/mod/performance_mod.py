from numpy import absolute
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

def performance_fun(self, *models, perf_df, predict_data, plot_perf):
    predict_data = predict_data or self.test_X
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
        return plot_lolipop(self, od_acc = ordered_data_acc, od_mae = ordered_data_mae, od_mse = ordered_data_mse, 
            od_rmse = ordered_data_rmse, od_r2 = ordered_data_r2, od_time = ordered_data_time)

    def plot_performance_boxplot(self,results_mae, results_mse, results_rmse, results_r2):
        return plot_boxplot(self,res_mae = results_mae, res_mse = results_mse, res_rmse = results_rmse, res_r2 = results_r2)

    if plot_perf == 'lolipop-plot':
        plot_performance_lolipop(self, ordered_data_acc, ordered_data_mae, ordered_data_mse,
        ordered_data_rmse, ordered_data_r2, ordered_data_time)

    elif plot_perf == 'box-plot':
        plot_performance_boxplot(self, results_mae, results_mse, results_rmse, results_r2)

    