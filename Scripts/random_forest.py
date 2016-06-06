import sys
import pandas as pd
import numpy as np
import datetime
import time
import os
from sklearn.ensemble import RandomForestClassifier


def random_forest(train_file_path, test_file_path):
    print("Reading training data")
    df_train = pd.read_csv(train_file_path)
    print("Reading test data")
    df_test = pd.read_csv(test_file_path)

    size = 10.0
    x_step = 0.2
    y_step = 0.2

    x_ranges = zip(np.arange(0, size, x_step), np.arange(x_step, size + x_step, x_step))
    y_ranges = zip(np.arange(0, size, y_step), np.arange(y_step, size + y_step, y_step))

    print('Calculate hour, weekday, month and year for train and test')
    df_train['hour'] = (df_train['time'] // 60) % 24 + 1  # 1 to 24
    df_train['weekday'] = (df_train['time'] // 1440) % 7 + 1
    df_train['month'] = (df_train['time'] // 43200) % 12 + 1  # rough estimate, month = 30 days
    df_train['year'] = (df_train['time'] // 525600) + 1

    df_test['hour'] = (df_test['time'] // 60) % 24 + 1  # 1 to 24
    df_test['weekday'] = (df_test['time'] // 1440) % 7 + 1
    df_test['month'] = (df_test['time'] // 43200) % 12 + 1  # rough estimate, month = 30 days
    df_test['year'] = (df_test['time'] // 525600) + 1

    preds_total = pd.DataFrame()

    for x_min, x_max in x_ranges:
        start_time_row = time.time()
        for y_min, y_max in y_ranges:
            start_time_cell = time.time()
            print("Ranges:", x_min, x_max, y_min, y_max)

            x_max = round(x_max, 4)
            x_min = round(x_min, 4)
            y_max = round(y_max, 4)
            y_min = round(y_min, 4)

            if x_max == size:
                x_max = x_max + 0.001
            if y_max == size:
                y_max = y_max + 0.001

            train_grid = df_train[(df_train['x'] >= x_min) &
                                  (df_train['x'] < x_max) &
                                  (df_train['y'] >= y_min) &
                                  (df_train['y'] < y_max)]

            test_grid = df_test[(df_test['x'] >= x_min) &
                                  (df_test['x'] < x_max) &
                                  (df_test['y'] >= y_min) &
                                  (df_test['y'] < y_max)]

            if train_grid.shape[0] == 0 or test_grid.shape[0] == 0:
                continue

            X_train_grid = train_grid[['x', 'y', 'accuracy', 'hour', 'weekday', 'month', 'year']]
            Y_train_grid = train_grid[['place_id']].values.ravel()
            X_test_grid = test_grid[['x', 'y', 'accuracy', 'hour', 'weekday', 'month', 'year']]

            clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0)
            clf.fit(X_train_grid, Y_train_grid)

            preds = dict(zip([el for el in clf.classes_], zip(*clf.predict_proba(X_test_grid))))
            preds = pd.DataFrame.from_dict(preds)

            preds['0_'], preds['1_'], preds['2_'] = zip(
                *preds.apply(lambda x: preds.columns[x.argsort()[::-1][:3]].tolist(), axis=1))
            preds = preds[['0_', '1_', '2_']]
            preds['row_id'] = test_grid['row_id'].reset_index(drop=True)
            preds_total = pd.concat([preds_total, preds], axis=0)

            end_time_cell = time.time()
            print("Elapsed time cell: %s seconds" % (time.time() - start_time_cell))
            print("HERE HERE")
        print("Elapsed time row: %s seconds" % (time.time() - start_time_row))

    preds_total['place_id'] = preds_total[['0_', '1_', '2_']].apply(lambda x: ' '.join([str(x1) for x1 in x]), axis=1)
    preds_total.drop('0_', axis=1, inplace=True)
    preds_total.drop('1_', axis=1, inplace=True)
    preds_total.drop('2_', axis=1, inplace=True)

    sub_file = os.path.join('rf_submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    preds_total.to_csv(sub_file, index=False)


def main():
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    start_time = time.time()
    random_forest(train_file_path, test_file_path)
    print("Time taken to run Random Forest: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()