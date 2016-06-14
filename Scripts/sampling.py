import sys
import pandas as pd


def main():
    input_path = sys.argv[1]
    data = pd.read_csv(input_path)
    sampling = data.sample(n=max(5, int(0.1 * data.shape[0])))
    sampling.to_csv(input_path + ".sample.csv")


if __name__ == "__main__":
    main()