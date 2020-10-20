from .afn_dataset import load_afn_dataset


def load_book_crossing_dataset():
    return load_afn_dataset(
        "book-crossing",
        url_dict={
            "nas": "book-crossing",
        },
    )
