import sys
import pandas as pd


def main():
    input_path = sys.argv[1]
    count_splits = int(sys.argv[2])
    data = pd.read_csv(input_path)
    n = data.shape[0]
    split_size = (n + count_splits - 1) // count_splits
    for x in range(count_splits):
        line_start = x * split_size
        line_end = min((x + 1) * split_size, n)
        split_data = data.iloc[line_start:line_end]
        split_data.to_csv(input_path + ".split_" + str(x) + ".csv", index=False)


if __name__ == "__main__":
    main()