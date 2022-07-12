import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_settings_mod import plot_settings
import matplotlib.gridspec as gridspec


def param_depend_fun(self, plot_type):
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
    plt_typ = plot_type
    sorted_M_df = united.sort_values(by="M")
    sorted_S_df = united.sort_values(by="S")
    sorted_V0_df = united.sort_values(by="V_0")
    sorted_k_df = united.sort_values(by="k")
    sorted_P0_df = united.sort_values(by="P_0")
    sorted_Ta_df = united.sort_values(by="T_a")
    sorted_T0_df = united.sort_values(by="T_0")

    sorted_dfs = {
        "M": [sorted_M_df["M"], sorted_M_df["y_true"]],
        "S": [sorted_S_df["S"], sorted_S_df["y_true"]],
        "V0": [sorted_V0_df["V_0"], sorted_V0_df["y_true"]],
        "k": [sorted_k_df["k"], sorted_k_df["y_true"]],
        "P0": [sorted_P0_df["P_0"], sorted_P0_df["y_true"]],
        "Ta": [sorted_Ta_df["T_a"], sorted_Ta_df["y_true"]],
        "T0": [sorted_T0_df["T_0"], sorted_T0_df["y_true"]]
    }

    if plot_type in ["M", "S", "V0", "k", "P0", "Ta", "T0"]:
        def plot_parameter_depend(sorted_dfs, plt_typ):
            """
            Function that creates one plot of true/predicted value vs.
            defined parameter.

            """
            fig = plt.figure()
            fig.suptitle(
                "Dependence of the function on the parameter",
                fontsize=18)
            plt.plot(
                sorted_dfs[plt_typ][0],
                sorted_dfs[plt_typ][1],
                "o",
                label="y_true")
            m, b = np.polyfit(
                sorted_dfs[plt_typ][0], sorted_dfs[plt_typ][1], 1)
            plt.plot(
                sorted_dfs[plt_typ][0],
                m * sorted_dfs[plt_typ][0] + b,
                lw=1.5)
            plt.xlabel(plot_type)
            plot_settings()
            fig.supylabel("Cycle time [s]")
            plt.gca().legend(("True values", "Trendline"))
            plt.show()

        plot_parameter_depend(sorted_dfs, plt_typ)

    elif plot_type == "all":
        def plot_parameter_depend_all(sorted_dfs):
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
                    "o",
                    label="y_true",
                    markersize=2)
                ma, ba = np.polyfit(
                    sorted_dfs[ptyp][0], sorted_dfs[ptyp][1], 1)
                ax.plot(
                    sorted_dfs[ptyp][0],
                    ma * sorted_dfs[ptyp][0] + ba,
                    lw=1.5)
                ax.set_xlabel(ptyp)
            fig.supylabel("Cycle time [s]")
            fig.suptitle(
                "Dependence of the function on the parameters",
                fontsize=18)
            plt.gca().legend(("True values", "Trendline"))
            plt.show()

        plot_parameter_depend_all(sorted_dfs)
