from .afn_dataset import load_afn_dataset


def load_avazu_dataset():
    return load_afn_dataset(
        "avazu",
        url_dict={
            "nas": "avazu",
            "postfix": "-1.libsvm",
        },
    )
