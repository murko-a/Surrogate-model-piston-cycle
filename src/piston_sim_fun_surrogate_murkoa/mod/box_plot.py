import matplotlib.pyplot as plt
from plot_settings_mod import plot_settings


def plot_boxplot(self, res_mae, res_mse, res_rmse, res_r2):
    plot_settings()
    data_range = range(1, len(self.mdls_pf) + 1)
    fig, axs = plt.subplots(2, 2, sharex=False, squeeze=True)
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle("Surrogate models performance", fontsize=18)

    axs[0, 0].boxplot(res_mae)
    axs[0, 0].set_title("MAE")
    axs[0, 0].set_xticks(data_range)
    axs[0, 0].set_xticklabels([i for i in list(self.mdls_pf)])
    axs[0, 0].set_ylabel("MAE value")

    axs[0, 1].boxplot(res_mse)
    axs[0, 1].set_title("MSE")
    axs[0, 1].set_xticks(data_range)
    axs[0, 1].set_xticklabels([i for i in list(self.mdls_pf)])
    axs[0, 1].set_ylabel("MSE value")

    axs[1, 0].boxplot(res_rmse)
    axs[1, 0].set_title("RMSE")
    axs[1, 0].set_xticks(data_range)
    axs[1, 0].set_xticklabels([i for i in list(self.mdls_pf)])
    axs[1, 0].set_ylabel("RMSE value")

    axs[1, 1].boxplot(res_r2)
    axs[1, 1].set_title("R2")
    axs[1, 1].set_xticks(data_range)
    axs[1, 1].set_xticklabels([i for i in list(self.mdls_pf)])
    axs[1, 1].set_ylabel("R2 value")

    fig.supxlabel("Model")

    plt.show()
