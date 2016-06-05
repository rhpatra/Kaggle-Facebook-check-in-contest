import pandas as pd
from sklearn.neighbors import KDTree
import sys
import timeit


def nn(trainfile_path, testfile_path):
    train = pd.read_csv(trainfile_path)
    test = pd.read_csv(testfile_path)
    tree = KDTree(train[['x', 'y']])
    _, ind = tree.query(test[['x', 'y']], k=3)
    temp = [y for x in ind for y in x]
    temp_ind = [i for i, x in enumerate(ind) for y in x]
    temp = train.iloc[temp].place_id.values
    preds = [[] for x in range(test.shape[0])]
    for a, b in zip(temp_ind, temp):
        preds[a].append(str(b))
    preds = [" ".join(x) for x in preds]
    test['place_id'] = preds
    test[['row_id', 'place_id']].to_csv('submission_NN.gz', index=False, compression='gzip')


def main():
    start_time = timeit.default_timer()
    nn(sys.argv[1], sys.argv[2])
    stop_time = timeit.default_timer()
    print("Time taken to run NN " + str.format("{0:.2f}", stop_time - start_time) + "s")


if __name__ == "__main__":
    main()