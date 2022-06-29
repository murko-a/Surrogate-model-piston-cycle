import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from plot_settings_mod import plot_settings

def compare_true_pred_fun(self, *models, plot, results):
    """Comparison of true and predicted values function.

    Function takes user-defined models from models option list,
    boolean argument plot which returns comparison plot if enabled,
    and boolean argument results which if enabled, returns dataframe of 
    parameter space generated with running the class, with predicted 
    values of defined models and calculated true values.

    Args:
        *models: Surrogate models argument list. 
            Options:   "RFR" - Random Forrest Regression,
                        "LR" - Linear Regression,
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
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    united = pd.DataFrame(self.test_X, columns=['M','S', 'V_0', 'k', 'P_0', 'T_a', 'T_0'])
    united['y_true'] = self.test_y
    mdls_pf = models
    self.mdls_pf = mdls_pf
    for mm in self.mdls_pf:
        mdl = self.models[mm]
        mdl.fit(self.train_X, self.train_y)
        pred = mdl.predict(self.test_X)
        for j in range(np.shape(self.test_X)[0]): 	
                united.loc[j, mm] = pred[j]	
    if results == True:
        display(united) 

    if plot == True:
        def plot_true_pred(mdls_pf):
            """
            Function that takes user-defined models and creates true
            vs. predicted value plot for every model.
            """
            axs_size = len(mdls_pf)
            plot_settings();
            fig, axs = plt.subplots(axs_size, sharex=True, squeeze=True)
            fig.subplots_adjust(hspace=0.2)
            fig.suptitle('Comparison of true and predicted values', fontsize=24)
            for i, mm in zip(range(axs_size), mdls_pf):
                axs[i].plot(united['y_true'], united['y_true'], '-', color='red', label='true values')
                axs[i].plot(united['y_true'], united[mm], 'o', label='predicted values',)
                axs[i].set_title(mm)
            fig.supxlabel('Cycle time [s]')
            fig.supylabel('Cycle time [s]')
            plt.xlim([0.2,0.8])
            plt.show()
        plot_true_pred(mdls_pf)