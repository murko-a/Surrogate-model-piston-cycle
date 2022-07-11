import matplotlib.pyplot as plt
from plot_settings_mod import plot_settings


def plot_lolipop(self, od_acc, od_mae, od_mse, od_rmse, od_r2, od_time):
    data_range = range(1, len(od_acc.index) + 1)
    plot_settings(fig_size=(20, 20))
    fig, ax = plt.subplots(
        2,3, sharex=False)
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle("Surrogate models performance", fontsize=18)

    ax[0][0].plot(list(od_acc["Accuracy"]), data_range, "bo", zorder=1)
    ax[0][0].hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_acc["Accuracy"]),
        color="blue",
        zorder=0)
    ax[0][0].set_yticks(data_range)
    ax[0][0].set_yticklabels([i for i in list(od_acc.index)])
    ax[0][0].set_title("Accuracy")
    

    ax[0][1].plot(list(od_mae["mean(MAE)"]), data_range, "bo", zorder=1)
    ax[0][1].hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_mae["mean(MAE)"]),
        color="blue",
        zorder=0)
    ax[0][1].set_yticks(data_range)
    ax[0][1].set_yticklabels([i for i in list(od_mae.index)])
    ax[0][1].set_title("mean(MAE)")
    

    ax[0][2].plot(list(od_mse["mean(MSE)"]), data_range, "bo", zorder=1)
    ax[0][2].hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_mse["mean(MSE)"]),
        color="blue",
        zorder=0)
    ax[0][2].set_yticks(data_range)
    ax[0][2].set_yticklabels([i for i in list(od_mse.index)])
    ax[0][2].set_title("mean(MSE)")


    ax[1][0].plot(list(od_rmse["mean(RMSE)"]), data_range, "bo", zorder=1)
    ax[1][0].hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_rmse["mean(RMSE)"]),
        color="blue",
        zorder=0)
    ax[1][0].set_yticks(data_range)
    ax[1][0].set_yticklabels([i for i in list(od_rmse.index)])
    ax[1][0].set_title("mean(RMSE)")


    ax[1][1].plot(list(od_r2["mean(R2)"]), data_range, "bo", zorder=1)
    ax[1][1].hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_r2["mean(R2)"]),
        color="blue",
        zorder=0)
    ax[1][1].set_yticks(data_range)
    ax[1][1].set_yticklabels([i for i in list(od_r2.index)])
    ax[1][1].set_title("mean(R2)")


    ax[1][2].plot(list(od_time["time"]), data_range, "bo", zorder=1)
    ax[1][2].hlines(
        y=data_range,
        xmin=0,
        xmax=list(
            od_time["time"]),
        color="blue",
        zorder=0)
    ax[1][2].set_yticks(data_range)
    ax[1][2].set_yticklabels([i for i in list(od_time.index)])
    ax[1][2].set_title("time")


    fig.supylabel("Model")

    plt.show()
