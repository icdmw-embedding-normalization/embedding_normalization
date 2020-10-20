import pandas as pd


def convert_values_to_index(df, column):
    df[column] = df[column].fillna("n/a")
    column_df = pd.DataFrame({column: df[column].value_counts().index}).reset_index().set_index(column)
    print(column, column_df.shape[0])
    column_df = column_df.rename(columns={"index": f"{column}-Index"})
    df = df.join(column_df, on=column)
    column_df[f"{column}-Index"] = column_df[f"{column}-Index"].astype(int)
    df.pop(column)
    df = df.rename(columns={f"{column}-Index": column})
    return df


def _load_item_df():
    df = pd.read_csv("book_crossing/BX-Books.csv", sep='";"', encoding="cp437", quoting=2, engine="python")
    df = df.rename(columns={'"ISBN': "ISBN"})
    df = df.drop(columns=["Book-Title", 'Image-URL-S', 'Image-URL-M', 'Image-URL-L"'])
    df["ISBN"] = df["ISBN"].map(lambda x: x[1:])
    return df


def _load_user_df():
    df = pd.read_csv("book_crossing/BX-Users.csv", sep=";", encoding="cp437")
    return df


def main():
    item_df = _load_item_df()
    user_df = _load_user_df()
    df = pd.read_csv("book_crossing/BX-Book-Ratings.csv", sep=";", encoding="cp437")

    df = df.join(user_df.set_index("User-ID"), on="User-ID")
    df = df.join(item_df.set_index("ISBN"), on="ISBN")

    for column in df.columns:
        if column == "Book-Rating":
            continue
        df = convert_values_to_index(df, column)

    max_index = 0
    for column in df.columns:
        _max_index = df[column].max()
        df[column] = df[column].apply(lambda x: x + max_index)
        max_index += (_max_index + 1)

    print("")
    for column in df.columns:
        print(column, df[column].min(), df[column].max())

    y = df.pop("Book-Rating").apply(lambda x: 1 if x > 0 else 0)
    df.insert(loc=0, column="y", value=y)
    print(df)
    input("!")

    df.to_pickle("book_crossing/df.pickle")


if __name__ == "__main__":
    main()
