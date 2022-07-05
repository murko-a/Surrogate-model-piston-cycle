import matplotlib.pyplot as plt


def plot_settings(fig_size=(20, 10)):
    """
    Function that defines settings for every plot in package.
    """
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
