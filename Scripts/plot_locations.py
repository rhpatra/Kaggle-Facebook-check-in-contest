import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd


def plot_locations(filepath):
    df_train = pd.read_csv(filepath)
    x = df_train["x"]
    y = df_train["y"]
    bins = 20
    while bins <= 160:
        plt.hist2d(x, y, bins=bins, norm=LogNorm())
        plt.colorbar()
        plt.title("x and y locations histogram with "+ str(bins) + " bins.")
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.savefig("locations_x_y_plit_bins_" + str(bins) + ".jpg")
        plt.show()
        bins *= 2


def main():
    plot_locations(sys.argv[1])


if __name__ == "__main__":
    main()