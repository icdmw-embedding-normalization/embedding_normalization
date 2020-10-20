import re
import zipfile
from functools import partial
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import requests
import torch.utils.data
from tqdm import tqdm

N_LABEL_SHUFFLE = 20

NAS_DIR = "."
BASE_DIR = "."


def _get_key(x):
    m = re.search("(?P<key>.*)\:(?P<value>.*)", x)
    assert m
    assert m.group("value") == "1"
    return int(m.group("key"))


def _download(url, file_path):
    r = requests.get(url, stream=True)
    with open(str(file_path), 'wb') as fd:
        for chunk in tqdm(r.iter_content(chunk_size=1024)):
            fd.write(chunk)


def _download_if_not_exists(dataset_name, url_dict):
    dir_path = _get_dir_path(dataset_name)
    dir_path.mkdir(exist_ok=True, parents=True)

    zip_url = url_dict.get("zip", None)
    if zip_url is not None:
        file_path = dir_path / f"all.zip"
        if not file_path.exists():
            _download(zip_url, file_path)

        if sum(1 for key in ["train", "valid", "test"] if not (dir_path / f"{key}.libsvm").exists()) > 0:
            with zipfile.ZipFile(str(file_path), 'r') as zip_file:
                zip_file.extractall(str(dir_path))

    elif url_dict.get("nas", None):
        nas_dir_path = Path(f"{NAS_DIR}/{url_dict['nas']}")
        postfix = url_dict.get("postfix", ".libsvm")

        for key in ["train", "valid", "test"]:
            file_path = dir_path / f"{key}{postfix}"
            if file_path.exists():
                continue

            nas_file_path = nas_dir_path / f"{key}{postfix}"
            copyfile(str(nas_file_path), str(file_path))

    else:
        for key, url in url_dict.items():
            file_path = dir_path / f"{key}.libsvm"
            if file_path.exists():
                continue

            _download(url, file_path)


def _get_dir_path(dataset_name):
    dir_path = f"{BASE_DIR}/{dataset_name}"
    dir_path = Path(dir_path)
    return dir_path


class _AFNDataset(torch.utils.data.Dataset):
    def __init__(self, key, dataset_name, postfix):
        dir_path = _get_dir_path(dataset_name)

        file_path = dir_path / f"{key}{postfix}"
        df = pd.read_csv(file_path, sep=" ", engine="c", header=None)

        self.targets = df.pop(0).to_numpy().astype(np.int)
        for column in df.columns:
            df[column] = df[column].map(_get_key)
        self.items = df.to_numpy().astype(np.int)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


class AFNTrainDataset(_AFNDataset):
    def __init__(self, dataset_name, postfix):
        super().__init__("train", dataset_name, postfix)

        self.feature_groups = None

    def set_feature_groups(self, _get_feature_groups):
        if _get_feature_groups is not None:
            self.feature_groups = _get_feature_groups(self.items)


class AFNValidDataset(_AFNDataset):
    def __init__(self, dataset_name, postfix):
        super().__init__("valid", dataset_name, postfix)


class AFNTestDataset(_AFNDataset):
    def __init__(self, dataset_name, postfix):
        super().__init__("test", dataset_name, postfix)


class _AFNHeadOrTailDataset(torch.utils.data.Dataset):
    def __init__(self, items, targets):
        self.items = items
        self.targets = targets

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]


def _is_index_overlapped_between_fields(start_index, end_index):
    for j in range(start_index.shape[1] - 1):
        if end_index[0][j] >= start_index[0][j + 1]:
            return True
    return False


def _get_feature_groups(items, dataset_name, field_idx=0, distortion="none"):
    assert dataset_name in ["jester", "afn-movielens"]

    if distortion == "none":
        target_ids, counts = np.unique(items[:, field_idx], return_counts=True)
        df = pd.DataFrame({
            "target_id": target_ids,
            "count": counts,
        })

        df = df.sort_values(by=["count"], axis=0, ascending=False)
        df = df.reset_index(drop=True)

        n_tail = (df["count"].cumsum() > int(df["count"].sum() * 0.8)).sum()
        n_head = df.shape[0] - n_tail

        df_head = df.loc[:n_head - 1]
        df_tail = df.loc[n_head:]
        return {
            "head": (torch.from_numpy(df_head["target_id"].to_numpy()), field_idx),
            "tail": (torch.from_numpy(df_tail["target_id"].to_numpy()), field_idx),
        }

    elif distortion == "dummy-field":
        head_target_ids = np.unique(items[:, field_idx])
        tail_target_ids = np.unique(items[:, items.shape[1] - 1])
        return {
            "head": (torch.from_numpy(head_target_ids), field_idx),
            "tail": (torch.from_numpy(tail_target_ids), items.shape[1] - 1),
        }

    elif distortion == "label-shuffle":
        target_ids = np.unique(items[:, field_idx])
        head_target_ids = target_ids[target_ids % N_LABEL_SHUFFLE != 0]
        tail_target_ids = target_ids[target_ids % N_LABEL_SHUFFLE == 0]

        return {
            "head": (torch.from_numpy(head_target_ids), field_idx),
            "tail": (torch.from_numpy(tail_target_ids), field_idx),
        }
    else:
        raise NotImplementedError()


def _get_head_or_tail_dataset(df):
    targets = df.pop("y").to_numpy()
    items = df.to_numpy()
    dataset = _AFNHeadOrTailDataset(items, targets)
    return dataset


def _get_head_and_tail_dataset(dataset, feature_groups):
    head_features, head_index = feature_groups["head"]
    tail_features, tail_index = feature_groups["tail"]
    assert head_index == tail_index

    head_df = pd.DataFrame({"target_id": head_features})
    tail_df = pd.DataFrame({"target_id": tail_features})

    df = pd.DataFrame({"y": dataset.targets})
    for j in range(dataset.items.shape[1]):
        df[j] = dataset.items[:, j]

    head_df = df.join(head_df.set_index("target_id"), on=head_index, how="inner")
    tail_df = df.join(tail_df.set_index("target_id"), on=tail_index, how="inner")

    head_dataset = _get_head_or_tail_dataset(head_df)
    tail_dataset = _get_head_or_tail_dataset(tail_df)
    return head_dataset, tail_dataset


def _distort_with_dummy_field(dataset, field_dims, field_idx=0):
    n_feature_in_dummy_field = field_dims[field_idx]
    field_dims = np.append(field_dims, [n_feature_in_dummy_field], axis=0)

    for _dataset in dataset:
        _dataset.items = np.append(
            _dataset.items,
            np.random.randint(low=0, high=n_feature_in_dummy_field - 1, size=(_dataset.items.shape[0], 1)),
            axis=1
        )
    return dataset, field_dims


def _distort_with_label_shuffle(dataset, field_dims, field_idx=0):
    for _dataset in dataset:
        _where = (_dataset.items[:, field_idx] % N_LABEL_SHUFFLE == 0)
        _dataset.targets[_where] = np.random.randint(2, size=np.sum(_where))
    return dataset, field_dims


def load_afn_dataset(dataset_name, url_dict, distortion="none", field_idx=0):
    assert dataset_name in ["afn-movielens", "avazu", "book-crossing", "jester"]

    _download_if_not_exists(dataset_name, url_dict)

    # init data
    postfix = url_dict.get("postfix", ".libsvm")
    train_dataset = AFNTrainDataset(dataset_name, postfix)
    valid_dataset = AFNValidDataset(dataset_name, postfix)
    test_dataset = AFNTestDataset(dataset_name, postfix)
    datasets = [train_dataset, valid_dataset, test_dataset]

    # rearrange token ids
    start_index = np.min(np.stack([np.min(dataset.items, axis=0) for dataset in datasets]), axis=0, keepdims=True)
    end_index = np.max(np.stack([np.max(dataset.items, axis=0) for dataset in datasets]), axis=0, keepdims=True)
    if _is_index_overlapped_between_fields(start_index, end_index):
        if dataset_name == "afn-movielens":
            pass
        else:
            raise NotImplementedError()

    for dataset in datasets:
        dataset.items -= start_index

    # calculate field_dims
    field_dims = np.max(
        np.stack(
            [np.max(dataset.items, axis=0) + 1 for dataset in datasets],
            axis=0
        ),
        axis=0
    )

    # apply distortion
    if distortion == "none":
        pass
    elif distortion == "dummy-field":
        dataset, field_dims = _distort_with_dummy_field(datasets, field_dims, field_idx=field_idx)
    elif distortion == "label-shuffle":
        dataset, field_dims = _distort_with_label_shuffle(datasets, field_dims, field_idx=field_idx)
    else:
        raise NotImplementedError()

    # set groups
    head_test_dataset = None
    tail_test_dataset = None
    if dataset_name in ["jester", "afn-movielens"]:
        train_dataset.set_feature_groups(
            partial(_get_feature_groups, dataset_name=dataset_name, field_idx=field_idx, distortion=distortion))

        if distortion in ["none", "label-shuffle"]:
            head_test_dataset, tail_test_dataset = _get_head_and_tail_dataset(test_dataset,
                                                                              train_dataset.feature_groups)
        elif distortion == "dummy-field":
            pass
        else:
            raise NotImplementedError()

    return (train_dataset, valid_dataset, test_dataset, head_test_dataset, tail_test_dataset), field_dims
