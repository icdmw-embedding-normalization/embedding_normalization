from .afn_dataset import load_afn_dataset


def load_jester_dataset(distortion):
    return load_afn_dataset(
        "jester",
        url_dict={
            "nas": "jester",
        },
        distortion=distortion,
    )
