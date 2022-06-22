from numpy import absolute
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from IPython.display import display

def performance_fun(self, *mdls_pf, perf_df, predict_data, plot_perf):
    predict_data = predict_data or self.test_X
    self.mdls_pf = mdls_pf
    performance_df = pd.DataFrame(columns = ['Accuracy', 'mean(MAE)', 'std(MAE)','mean(MSE)', 'std(MSE)',
        'mean(RMSE)', 'std(RMSE)','mean(R2)', 'std(R2)','time'])
    results_mae = []
    results_mse = []
    results_rmse = []
    results_r2 = []
    
    def acc(self):
        for mm in self.mdls_pf:
            data_y, data_yhat = list(), list()
            mdl = self.models[mm]
            mdl.fit(self.train_X, self.train_y)
            # make predictions
            yhat =  mdl.predict(self.test_X)
            # store
            data_y.extend(self.test_y)
            data_yhat.extend(yhat)
            errors = abs(np.array(data_yhat) - np.array(data_y))
            mape = 100 * (errors / np.array(data_y))
            accuracy = 100 - np.mean(mape)
            performance_df.loc[mm, performance_df.columns[0]] = accuracy	
    acc(self)

    def mae(self):
        for mm in self.mdls_pf:
            mdl = self.models[mm]
            scc_mae = cross_val_score(mdl, self.X, self.y, scoring='neg_mean_absolute_error', cv=self.kfold, n_jobs=-1)
            results_mae.append(absolute(scc_mae))
            performance_df.loc[mm, performance_df.columns[1]] = absolute(scc_mae.mean())
            performance_df.loc[mm, performance_df.columns[2] ] = absolute(scc_mae.std())	
    mae(self)
    
    def mse(self):
        for mm in self.mdls_pf:
            mdl = self.models[mm]
            scc_mse = cross_val_score(mdl, self.X, self.y, scoring='neg_mean_squared_error', cv=self.kfold, n_jobs=-1)
            results_mse.append(absolute(scc_mse))
            performance_df.loc[mm, performance_df.columns[3]] = absolute(scc_mse.mean())
            performance_df.loc[mm, performance_df.columns[4]] = absolute(scc_mse.std())
    mse(self)

    def rmse(self):
        for mm in self.mdls_pf:
            mdl = self.models[mm]
            scc_rmse = cross_val_score(mdl, self.X, self.y, scoring='neg_mean_squared_error', cv=self.kfold, n_jobs=-1)
            results_rmse.append(np.sqrt(absolute(scc_rmse)))
            performance_df.loc[mm, performance_df.columns[5]] = np.sqrt(absolute(scc_rmse.mean()))
            performance_df.loc[mm, performance_df.columns[6]] = np.sqrt(absolute(scc_rmse.std()))
    rmse(self)

    def r2(self):
        for mm in self.mdls_pf:
            mdl = self.models[mm]
            scc_r2 = cross_val_score(mdl, self.X, self.y, scoring='r2', cv=self.kfold, n_jobs=-1)
            results_r2.append(absolute(scc_r2))
            performance_df.loc[mm, performance_df.columns[7]] = absolute(scc_r2.mean())
            performance_df.loc[mm, performance_df.columns[8]] = absolute(scc_r2.std())
    r2(self)

    def time_mod(self):
        for mm in self.mdls_pf:
            startk = process_time()
            mdl = self.models[mm]
            mdl.fit(self.train_X, self.train_y)
            mdl.predict(self.test_X)
            endk = process_time()
            tt = ((endk-startk)/60)
            performance_df.loc[mm, performance_df.columns[9]] = tt
    time_mod(self)

    if perf_df == True:
        display(performance_df)

    ordered_data_acc = performance_df.sort_values(by='Accuracy', ascending=True)
    ordered_data_mae = performance_df.sort_values(by='mean(MAE)', ascending=True)
    ordered_data_mse = performance_df.sort_values(by='mean(MSE)', ascending=True)
    ordered_data_rmse = performance_df.sort_values(by='mean(RMSE)', ascending=True)
    ordered_data_r2 = performance_df.sort_values(by='mean(R2)', ascending=True)
    ordered_data_time = performance_df.sort_values(by='time', ascending=True)
    

    def plot_performance_lolipop(self, ordered_data_acc, ordered_data_mae, ordered_data_mse, 
    ordered_data_rmse, ordered_data_r2, ordered_data_time):
        data_range=range(1,len(ordered_data_acc.index)+1)
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=False, squeeze=True, figsize=(30, 30))
        fig.subplots_adjust(hspace=0.2)

        ax1.plot(list(ordered_data_acc['Accuracy']),data_range, "bo", zorder=1)
        ax1.hlines(y=data_range, xmin=0, xmax=list(ordered_data_acc['Accuracy']), color='blue', zorder=0)
        ax1.set_yticks(data_range)
        ax1.set_yticklabels([i for i in list(ordered_data_acc.index)])
        ax1.set_title('Accuracy')

        ax2.plot(list(ordered_data_mae['mean(MAE)']),data_range, "bo", zorder=1)
        ax2.hlines(y=data_range, xmin=0, xmax=list(ordered_data_mae['mean(MAE)']), color='blue', zorder=0)
        ax2.set_yticks(data_range)
        ax2.set_yticklabels([i for i in list(ordered_data_mae.index)])
        ax2.set_title('mean(MAE)')

        ax3.plot(list(ordered_data_mse['mean(MSE)']),data_range, "bo", zorder=1)
        ax3.hlines(y=data_range, xmin=0, xmax=list(ordered_data_mse['mean(MSE)']), color='blue', zorder=0)
        ax3.set_yticks(data_range)
        ax3.set_yticklabels([i for i in list(ordered_data_mse.index)])
        ax3.set_title('mean(MSE)')

        ax4.plot(list(ordered_data_rmse['mean(RMSE)']),data_range, "bo", zorder=1)
        ax4.hlines(y=data_range, xmin=0, xmax=list(ordered_data_rmse['mean(RMSE)']), color='blue', zorder=0)
        ax4.set_yticks(data_range)
        ax4.set_yticklabels([i for i in list(ordered_data_rmse.index)])
        ax4.set_title('mean(RMSE)')

        ax5.plot(list(ordered_data_r2['mean(R2)']),data_range, "bo", zorder=1)
        ax5.hlines(y=data_range, xmin=0, xmax=list(ordered_data_r2['mean(R2)']), color='blue', zorder=0)
        ax5.set_yticks(data_range)
        ax5.set_yticklabels([i for i in list(ordered_data_r2.index)])
        ax5.set_title('mean(R2)')

        ax6.plot(list(ordered_data_time['time']),data_range, "bo", zorder=1)
        ax6.hlines(y=data_range, xmin=0, xmax=list(ordered_data_time['time']), color='blue', zorder=0)
        ax6.set_yticks(data_range)
        ax6.set_yticklabels([i for i in list(ordered_data_time.index)])
        ax6.set_title('time')
        
        plt.show();

    def plot_performance_boxplot(self,results_mae, results_mse, results_rmse, results_r2):
        data_range=range(1,len(self.mdls_pf)+1)
        fig, axs = plt.subplots(2,2, sharex=False, squeeze=True, figsize=(30, 30))
        fig.subplots_adjust(hspace=0.1)
        
        axs[0,0].boxplot(results_mae)
        axs[0,0].set_title('MAE')
        axs[0,0].set_xticks(data_range)
        axs[0,0].set_xticklabels([i for i in list(self.mdls_pf)])

        axs[0,1].boxplot(results_mse)
        axs[0,1].set_title('MSE')
        axs[0,1].set_xticks(data_range)
        axs[0,1].set_xticklabels([i for i in list(self.mdls_pf)])

        axs[1,0].boxplot(results_rmse)
        axs[1,0].set_title('RMSE')
        axs[1,0].set_xticks(data_range)
        axs[1,0].set_xticklabels([i for i in list(self.mdls_pf)])

        axs[1,1].boxplot(results_r2)
        axs[1,1].set_title('R2')
        axs[1,1].set_xticks(data_range)
        axs[1,1].set_xticklabels([i for i in list(self.mdls_pf)])

        plt.show()

    if plot_perf == 'lolipop-plot':
        plot_performance_lolipop(self, ordered_data_acc, ordered_data_mae, ordered_data_mse,
        ordered_data_rmse, ordered_data_r2, ordered_data_time)
    elif plot_perf == 'box-plot':
        plot_performance_boxplot(self, results_mae, results_mse, results_rmse, results_r2)

    