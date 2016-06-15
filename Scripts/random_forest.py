import sys
import pandas as pd
import datetime
import time
import os
from sklearn.ensemble import RandomForestClassifier


def random_forest(train_file_path, test_file_path, output_file_path):
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)

    size = 10.0
    x_step = 0.2
    y_step = 0.2

    print('Calculate hour, weekday, month and year for train and test')
    train['hour'] = (train['time'] // 60) % 24 + 1  # 1 to 24
    train['weekday'] = (train['time'] // 1440) % 7 + 1
    train['month'] = (train['time'] // 43200) % 12 + 1  # rough estimate, month = 30 days
    train['year'] = (train['time'] // 525600) + 1
    print(train.info())

    test['hour'] = (test['time'] // 60) % 24 + 1  # 1 to 24
    test['weekday'] = (test['time'] // 1440) % 7 + 1
    test['month'] = (test['time'] // 43200) % 12 + 1  # rough estimate, month = 30 days
    test['year'] = (test['time'] // 525600) + 1
    print(test.info())

    preds_total = pd.DataFrame()
    for i in range((int)(size / x_step)):
        x_min = x_step * i
        x_max = x_step * (i + 1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        start_time_row = time.time()
        if x_max == size:
            x_max = x_max + 0.001
        for j in range((int)(size / y_step)):
            # start_time_cell = time.time()
            y_min = y_step * j
            y_max = y_step * (j + 1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)
            if y_max == size:
                y_max = y_max + 0.001

            train_grid = train[(train['x'] >= x_min) &
                               (train['x'] < x_max) &
                               (train['y'] >= y_min) &
                               (train['y'] < y_max)]

            test_grid = test[(test['x'] >= x_min) &
                             (test['x'] < x_max) &
                             (test['y'] >= y_min) &
                             (test['y'] < y_max)]

            X_train_grid = train_grid[['x', 'y', 'accuracy', 'hour', 'weekday', 'month', 'year']]
            y_train_grid = train_grid[['place_id']].values.ravel()
            X_test_grid = test_grid[['x', 'y', 'accuracy', 'hour', 'weekday', 'month', 'year']]

            # clf = GradientBoostingClassifier();
            # clf = LogisticRegression(multi_class='multinomial', solver = 'lbfgs');
            # clf = xgb.XGBClassifier(n_estimators=10);
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            clf.fit(X_train_grid, y_train_grid)

            preds = dict(zip([el for el in clf.classes_], zip(*clf.predict_proba(X_test_grid))))
            preds = pd.DataFrame.from_dict(preds)

            preds['0_'], preds['1_'], preds['2_'] = zip(
                *preds.apply(lambda x: preds.columns[x.argsort()[::-1][:3]].tolist(), axis=1))
            preds = preds[['0_', '1_', '2_']]
            preds['row_id'] = test_grid['row_id'].reset_index(drop=True)
            preds['x'] = test_grid['x'].reset_index(drop=True)
            preds['y'] = test_grid['y'].reset_index(drop=True)
            preds_total = pd.concat([preds_total, preds], axis=0)
            # print("Elapsed time cell: %s seconds" % (time.time() - start_time_cell))
        print("Elapsed time row: %s seconds" % (time.time() - start_time_row))

    preds_total = preds_total.sort_values('row_id')
    print(preds_total.info())
    preds_total['place_id'] = preds_total[['0_', '1_', '2_']].apply(lambda x: ' '.join([str(x1) for x1 in x]), axis=1)
    preds_total.drop('0_', axis=1, inplace=True)
    preds_total.drop('1_', axis=1, inplace=True)
    preds_total.drop('2_', axis=1, inplace=True)
    # sub_file = os.path.join('submission_random_forest_abhishek_kadian' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    sub_file = output_file_path
    preds_total.to_csv(sub_file, index=False)


def main():
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    start_time = time.time()
    random_forest(train_file_path, test_file_path, output_file_path)
    print("Time taken to run Random Forest: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()