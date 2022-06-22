import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def compare_true_pred_fun(self, *mdls_pf, plot_type, df):
    united = pd.DataFrame(self.test_X, columns=['M','S', 'V_0', 'k', 'P_0', 'T_a', 'T_0'])
    united['y_true'] = self.test_y
    mdls_pf = mdls_pf
    plt_typ = plot_type
    self.mdls_pf = mdls_pf
    for mm in self.mdls_pf:
        mdl = self.models[mm]
        mdl.fit(self.train_X, self.train_y)
        pred = mdl.predict(self.test_X)
        for j in range(np.shape(self.test_X)[0]): 	
                united.loc[j, mm] = pred[j]	
    if df == True:
        display(united) 
    sorted_M_df = united.sort_values(by = 'M')
    sorted_S_df = united.sort_values(by = 'S')
    sorted_V0_df = united.sort_values(by = 'V_0')
    sorted_k_df = united.sort_values(by = 'k')
    sorted_P0_df = united.sort_values(by = 'P_0')
    sorted_Ta_df = united.sort_values(by = 'T_a')
    sorted_T0_df = united.sort_values(by = 'T_0')

    if plot_type in ['M','S', 'V0', 'k', 'P0', 'Ta', 'T0']:
        sorted_dfs = {
            'M': [sorted_M_df['M'], sorted_M_df['y_true'], sorted_M_df],
            'S': [sorted_S_df['S'], sorted_S_df['y_true'], sorted_S_df],
            'V0': [sorted_V0_df['V_0'], sorted_V0_df['y_true'], sorted_V0_df],
            'k': [sorted_k_df['k'], sorted_k_df['y_true'], sorted_k_df],
            'P0': [sorted_P0_df['P_0'], sorted_P0_df['y_true'], sorted_P0_df],
            'Ta': [sorted_Ta_df['T_a'], sorted_Ta_df['y_true'], sorted_Ta_df],
            'T0': [sorted_T0_df['T_0'], sorted_T0_df['y_true'], sorted_T0_df]
            }

        def plot_settings():
                plt.rcParams['figure.figsize'] = (20, 10)
                plt.rcParams['font.family'] = 'Times New Roman'
                plt.rcParams['font.size'] = 16
    
        def plot_compare_y_parameter(mdls_pf, sorted_dfs, plt_typ):
            plt.plot(sorted_dfs[plt_typ][0],sorted_dfs[plt_typ][1], label = 'y_true');
            plot_settings();
            ss_df = sorted_dfs[plt_typ][2]
            for mm in mdls_pf:
                plt.plot(sorted_dfs[plt_typ][0], ss_df[mm],'o', label = mm)
            plt.xlabel(plt_typ)
            plt.ylabel('Cycle time')
            plt.legend()
            plt.show();

        plot_compare_y_parameter(mdls_pf, sorted_dfs, plt_typ)

    elif plot_type == 'true-pred':
        def plot_true_pred(mdls_pf):
            axs_size = len(mdls_pf)

            fig, axs = plt.subplots(axs_size, sharex=True, squeeze=True, figsize=(20, 20))
            fig.subplots_adjust(hspace=0.2)
            for i, mm in zip(range(axs_size), mdls_pf):
                axs[i].plot(united['y_true'], united['y_true'], '-', color='red')
                axs[i].plot(united['y_true'], united[mm], 'o')
                axs[i].set_title(mm)
            plt.xlim([0.2,0.8])
            plt.show()
        plot_true_pred(mdls_pf)