import pandas as pd
from tqdm import tqdm


def _convert_line_from_row(row, columns):
    return f"{row.y} " + " ".join(f"{getattr(row, column)}:1" for column in columns)


def _save(lines, key, dataset):
    with open(f"{dataset}/{key}.libsvm", "w") as f:
        for line in tqdm(lines):
            f.write(f"{line}\n")


def _split(dataset):
    df = pd.read_pickle(f"{dataset}/df.pickle")
    df = df.sample(frac=1).reset_index(drop=True)

    n_train = int(df.shape[0] * 0.8)
    n_valid = int(df.shape[0] * 0.9)
    columns = list(df.columns)[1:]

    train_lines = []
    valid_lines = []
    test_lines = []
    for row_index, row in tqdm(df.iterrows(), total=df.shape[0]):
        line = _convert_line_from_row(row, columns)

        if row_index < n_train:
            train_lines.append(line)
        elif row_index < n_valid:
            valid_lines.append(line)
        else:
            test_lines.append(line)

    _save(train_lines, "train", dataset)
    _save(valid_lines, "valid", dataset)
    _save(test_lines, "test", dataset)


def main():
    for dataset in ["book_crossing", "jester"]:
        print(dataset)
        _split(dataset)


if __name__ == "__main__":
    main()
