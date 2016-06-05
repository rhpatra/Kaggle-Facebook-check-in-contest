import pandas as pd
from sklearn.neighbors import KDTree
import sys
import timeit


def nn(trainfile_path, testfile_path):
    train = pd.read_csv(trainfile_path)
    test = pd.read_csv(testfile_path)
    tree = KDTree(train[['x', 'y']])
    _, ind = tree.query(test[['x', 'y']], k=3)
    temp = [train.iloc[x].place_id.values for x in ind]
    temp = [" ".join((str(y) for y in x)) for x in temp]
    test['place_id'] = temp
    test[['row_id', 'place_id']].to_csv('submission_NN.gz', index=False, compression='gzip')


def main():
    start_time = timeit.default_timer()
    nn(sys.argv[1], sys.argv[2])
    stop_time = timeit.default_timer()
    print("Time taken to run NN " + str.format("{0:.2f}", stop_time - start_time) + "s")


if __name__ == "__main__":
    main()