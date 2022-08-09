from numpy import absolute
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from time import process_time, perf_counter



def perf_calc(self):
    performance_df = pd.DataFrame(
        columns=[
            "Accuracy",
            "mean(MAE)",
            "std(MAE)",
            "mean(MSE)",
            "std(MSE)",
            "mean(RMSE)",
            "std(RMSE)",
            "mean(R2)",
            "std(R2)",
            "time"])

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
            yhat = mdl.predict(self.test_X)
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
            scc_mae = cross_val_score(
                mdl,
                self.X,
                self.y,
                scoring="neg_mean_absolute_error",
                cv=self.kfold,
                n_jobs=-1)
            results_mae.append(absolute(scc_mae))
            performance_df.loc[mm, performance_df.columns[1]
                               ] = absolute(scc_mae.mean())
            performance_df.loc[mm, performance_df.columns[2]
                               ] = absolute(scc_mae.std())
    mae(self)

    def mse(self):
        for mm in self.mdls_pf:
            mdl = self.models[mm]
            scc_mse = cross_val_score(
                mdl,
                self.X,
                self.y,
                scoring="neg_mean_squared_error",
                cv=self.kfold,
                n_jobs=-1)
            results_mse.append(absolute(scc_mse))
            performance_df.loc[mm, performance_df.columns[3]
                               ] = absolute(scc_mse.mean())
            performance_df.loc[mm, performance_df.columns[4]
                               ] = absolute(scc_mse.std())
    mse(self)

    def rmse(self):
        for mm in self.mdls_pf:
            mdl = self.models[mm]
            scc_rmse = cross_val_score(
                mdl,
                self.X,
                self.y,
                scoring="neg_mean_squared_error",
                cv=self.kfold,
                n_jobs=-1)
            results_rmse.append(np.sqrt(absolute(scc_rmse)))
            performance_df.loc[mm, performance_df.columns[5]
                               ] = np.sqrt(absolute(scc_rmse.mean()))
            performance_df.loc[mm, performance_df.columns[6]
                               ] = np.sqrt(absolute(scc_rmse.std()))
    rmse(self)

    def r2(self):
        for mm in self.mdls_pf:
            mdl = self.models[mm]
            scc_r2 = cross_val_score(
                mdl,
                self.X,
                self.y,
                scoring="r2",
                cv=self.kfold,
                n_jobs=-1)
            results_r2.append(absolute(scc_r2))
            performance_df.loc[mm, performance_df.columns[7]
                               ] = absolute(scc_r2.mean())
            performance_df.loc[mm, performance_df.columns[8]
                               ] = absolute(scc_r2.std())
    r2(self)

    def time_mod(self):
        for mm in self.mdls_pf:
            startk = perf_counter()
            mdl = self.models[mm]
            mdl.fit(self.train_X, self.train_y)
            mdl.predict(self.test_X)
            endk = perf_counter()
            tt = ((endk - startk) / 60)
            performance_df.loc[mm, performance_df.columns[9]] = tt
    time_mod(self)

    return performance_df, results_mae, results_mse, results_rmse, results_r2
