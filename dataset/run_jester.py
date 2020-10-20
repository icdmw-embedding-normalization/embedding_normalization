import pandas as pd
from tqdm import tqdm


def _merge_datasets():
    rating_count = 0.0
    dfs = []
    for file_name in [
        "jester-data-1.xls",
        "jester-data-2.xls",
        "jester-data-3.xls",
        "FINAL jester 2006-15.xls",
        "[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx",
    ]:
        df = pd.read_excel(f"jester/{file_name}", header=None).reset_index(drop=True)

        for joke_index in range(101, 159):
            if joke_index not in list(df.columns):
                df[joke_index] = 99

        y = df.pop(0)
        _rating_count = y.sum()
        rating_count += _rating_count

        print(df.shape, _rating_count, rating_count)
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_pickle("jester/full_df.pickle")


def _convert_to_sparse_form():
    # user: 136025
    # joke: 158
    # instance: 6085247
    # density: 6085247 / (136025 * 158) = 0.283140757
    df = pd.read_pickle("jester/full_df.pickle")
    print(df)

    dfs = []
    for column in tqdm(range(1, 159)):
        s = df.pop(column)
        s = s.fillna(99)
        s = s[s != 99]

        s = s.reset_index(drop=False)
        s = s.rename(columns={"index": "userID", column: "rating"})
        s["itemID"] = column
        s["rating"] = s.pop("rating")
        dfs.append(s)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_pickle("jester/sparse_df.pickle")


def main():
    # _merge_datasets()
    # _convert_to_sparse_form()

    df = pd.read_pickle("jester/sparse_df.pickle")
    print(df["userID"].value_counts())
    print(df["itemID"].value_counts())

    df["itemID"] += 136024
    print(df)

    df["rating"][df["rating"] > 0] = 1
    df["rating"][df["rating"] <= 0] = 0
    print(df)
    print(df["rating"].mean())
    print(df["rating"].isnull().sum())

    print(df["rating"].value_counts())

    df.insert(loc=0, column="y", value=df.pop("rating").astype("int"))
    df.to_pickle("jester/df.pickle")


if __name__ == "__main__":
    main()
