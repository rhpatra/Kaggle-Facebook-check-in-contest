import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time
import seaborn as sns
import sys


def prelim_test(trainfile_path, testfile_path):
    # df_train = pd.read_csv("../Data/train.csv/train.csv")
    # df_test = pd.read_csv("../Data/test.csv/test.csv")

    print("Reading training file")
    df_train = pd.read_csv(trainfile_path)

    print("Reading test file")
    df_test = pd.read_csv(testfile_path)

    print("Size of training data: " + str(df_train.shape))
    print("Size of test data: " + str(df_test.shape))

    print("\nTrain columns: " + str(df_train.columns.values))
    print("Test columns: " + str(df_test.columns.values))

    print(df_train.describe())

    print("\nNumber of place ids: " + str(len(set(df_train['place_id'].values.tolist()))))


def main():
    prelim_test(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()