import matplotlib.pyplot as plt
from plot_settings_mod import plot_settings


def plot_lolipop(self, od_acc, od_mae, od_mse, od_rmse, od_r2, od_time):
    data_range = range(1, len(od_acc.index) + 1)
    plot_settings(fig_size=(15, 30))
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
        6, sharex=False, squeeze=True)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Surrogate models performance", fontsize=24)

    ax1.plot(list(od_acc["Accuracy"]), data_range, "bo", zorder=1)
    ax1.hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_acc["Accuracy"]),
        color="blue",
        zorder=0)
    ax1.set_yticks(data_range)
    ax1.set_yticklabels([i for i in list(od_acc.index)])
    ax1.set_title("Accuracy")

    ax2.plot(list(od_mae["mean(MAE)"]), data_range, "bo", zorder=1)
    ax2.hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_mae["mean(MAE)"]),
        color="blue",
        zorder=0)
    ax2.set_yticks(data_range)
    ax2.set_yticklabels([i for i in list(od_mae.index)])
    ax2.set_title("mean(MAE)")

    ax3.plot(list(od_mse["mean(MSE)"]), data_range, "bo", zorder=1)
    ax3.hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_mse["mean(MSE)"]),
        color="blue",
        zorder=0)
    ax3.set_yticks(data_range)
    ax3.set_yticklabels([i for i in list(od_mse.index)])
    ax3.set_title("mean(MSE)")

    ax4.plot(list(od_rmse["mean(RMSE)"]), data_range, "bo", zorder=1)
    ax4.hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_rmse["mean(RMSE)"]),
        color="blue",
        zorder=0)
    ax4.set_yticks(data_range)
    ax4.set_yticklabels([i for i in list(od_rmse.index)])
    ax4.set_title("mean(RMSE)")

    ax5.plot(list(od_r2["mean(R2)"]), data_range, "bo", zorder=1)
    ax5.hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_r2["mean(R2)"]),
        color="blue",
        zorder=0)
    ax5.set_yticks(data_range)
    ax5.set_yticklabels([i for i in list(od_r2.index)])
    ax5.set_title("mean(R2)")

    ax6.plot(list(od_time["time"]), data_range, "bo", zorder=1)
    ax6.hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_time["time"]),
        color="blue",
        zorder=0)
    ax6.set_yticks(data_range)
    ax6.set_yticklabels([i for i in list(od_time.index)])
    ax6.set_title("time")

    fig.supylabel("Model")

    plt.show()
