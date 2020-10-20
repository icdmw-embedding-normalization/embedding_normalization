from .afn_dataset import load_afn_dataset


def load_afn_movielens_dataset(distortion):
    return load_afn_dataset(
        "afn-movielens",
        url_dict={
            "train": "https://github.com/WeiyuCheng/AFN-AAAI-20/blob/master/data/movielens/train.libsvm?raw=true",
            "valid": "https://github.com/WeiyuCheng/AFN-AAAI-20/blob/master/data/movielens/valid.libsvm?raw=true",
            "test": "https://github.com/WeiyuCheng/AFN-AAAI-20/blob/master/data/movielens/test.libsvm?raw=true",
        },
        distortion=distortion,
    )
