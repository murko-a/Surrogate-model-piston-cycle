import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from plot_settings_mod import plot_settings
import matplotlib.gridspec as gridspec


def param_true_pred_fun(self, *models, plot_type, results):
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

        plot_type (sting): returns true/predicted values vs. parameter plot.
                            Default is "all". Plot options:
                                ["all","M","S", "V_0", "k", "P_0", "T_a", "T_0"]


        results (bool): returns dataframe of parameters, true and predicted values.
                            Default is True.

    Returns:
        Dataframe of parameter space, calculated true values and predicted values
        of selected models and  true/predicted values vs. parameter plot.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    united = pd.DataFrame(
        self.test_X,
        columns=[
            "M",
            "S",
            "V_0",
            "k",
            "P_0",
            "T_a",
            "T_0"])
    united["y_true"] = self.test_y
    mdls_pf = models
    plt_typ = plot_type
    self.mdls_pf = mdls_pf
    for mm in self.mdls_pf:
        mdl = self.models[mm]
        mdl.fit(self.train_X, self.train_y)
        pred = mdl.predict(self.test_X)
        for j in range(np.shape(self.test_X)[0]):
            united.loc[j, mm] = pred[j]
    if results:
        display(united)
    sorted_M_df = united.sort_values(by="M")
    sorted_S_df = united.sort_values(by="S")
    sorted_V0_df = united.sort_values(by="V_0")
    sorted_k_df = united.sort_values(by="k")
    sorted_P0_df = united.sort_values(by="P_0")
    sorted_Ta_df = united.sort_values(by="T_a")
    sorted_T0_df = united.sort_values(by="T_0")

    sorted_dfs = {
        "M": [sorted_M_df["M"], sorted_M_df["y_true"], sorted_M_df],
        "S": [sorted_S_df["S"], sorted_S_df["y_true"], sorted_S_df],
        "V0": [sorted_V0_df["V_0"], sorted_V0_df["y_true"], sorted_V0_df],
        "k": [sorted_k_df["k"], sorted_k_df["y_true"], sorted_k_df],
        "P0": [sorted_P0_df["P_0"], sorted_P0_df["y_true"], sorted_P0_df],
        "Ta": [sorted_Ta_df["T_a"], sorted_Ta_df["y_true"], sorted_Ta_df],
        "T0": [sorted_T0_df["T_0"], sorted_T0_df["y_true"], sorted_T0_df]
    }

    if plot_type in ["M", "S", "V0", "k", "P0", "Ta", "T0"]:
        def plot_compare_y_parameter(mdls_pf, sorted_dfs, plt_typ):
            """
            Function that creates one plot of true/predicted value vs.
            defined parameter.

            """
            fig = plt.figure()
            fig.suptitle(
                "Dependence of the function on the parameter",
                fontsize=24)
            plt.plot(
                sorted_dfs[plt_typ][0],
                sorted_dfs[plt_typ][1],
                label="y_true")
            plot_settings()
            ss_df = sorted_dfs[plt_typ][2]
            for mm in mdls_pf:
                plt.plot(sorted_dfs[plt_typ][0], ss_df[mm], "o", label=mm)
            plt.xlabel(plt_typ)
            plt.ylabel("Cycle time [s]")
            plt.legend()
            plt.show()

        plot_compare_y_parameter(mdls_pf, sorted_dfs, plt_typ)

    elif plot_type == "all":
        def plot_compare_y_parameter_all(mdls_pf, sorted_dfs):
            """
            Function that creates  panel plot with true/predicted value vs.
            defined parameter subplots for all the parameters defined in
            parameter space.

            """
            fig = plt.figure()
            plot_settings(fig_size=(15, 30))
            gs = gridspec.GridSpec(4, 2)
            gs.update(wspace=0.2, hspace=0.5)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[1, 0])
            ax4 = plt.subplot(gs[1, 1])
            ax5 = plt.subplot(gs[2, 0])
            ax6 = plt.subplot(gs[2, 1])
            ax7 = plt.subplot(gs[3, 0])
            axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
            plt_types = ["M", "S", "V0", "k", "P0", "Ta", "T0"]
            for ax, ptyp in zip(axs, plt_types):
                ax.plot(
                    sorted_dfs[ptyp][0],
                    sorted_dfs[ptyp][1],
                    label="y_true",
                    lw=0.35)
                for mm in mdls_pf:
                    ss_df = sorted_dfs[ptyp][2]
                    ax.plot(
                        sorted_dfs[ptyp][0],
                        ss_df[mm],
                        "o",
                        label=mm,
                        markersize=2)
                ax.set_xlabel(ptyp)
            fig.supylabel("Cycle time [s]")
            fig.suptitle(
                "Dependence of the function on the parameters",
                fontsize=24)
            plt.legend(loc="lower right")
            plt.show()

        plot_compare_y_parameter_all(mdls_pf, sorted_dfs)
