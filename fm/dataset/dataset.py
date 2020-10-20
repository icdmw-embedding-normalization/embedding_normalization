import torch
from sklearn.metrics import roc_auc_score

from common.utils.data_loader import data_loader_func
from .afn_movielens import load_afn_movielens_dataset
from .avazu import load_avazu_dataset
from .book_crossing import load_book_crossing_dataset
from .jester import load_jester_dataset


def auroc(y, target):
    return roc_auc_score(target.float().detach().cpu().numpy(),
                         y.detach().cpu().numpy())


def _get_data_loaders(dataset, batch_size, device):
    if len(dataset) == 5:
        train_dataset, valid_dataset, test_dataset, head_test_dataset, tail_test_dataset = dataset
    else:
        train_length = int(len(dataset) * 0.8)
        valid_length = int(len(dataset) * 0.1)
        test_length = len(dataset) - train_length - valid_length
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            dataset, (train_length, valid_length, test_length))
        head_test_dataset, tail_test_dataset = None, None

    train_data_loader = data_loader_func(train_dataset, batch_size, device, shuffle=True)
    valid_data_loader = data_loader_func(valid_dataset, batch_size, device, shuffle=False)
    test_data_loader = data_loader_func(test_dataset, batch_size, device, shuffle=False)

    if head_test_dataset is not None:
        head_test_data_loader = data_loader_func(head_test_dataset, batch_size, device, shuffle=False)
    else:
        head_test_data_loader = None

    if tail_test_dataset is not None:
        tail_test_data_loader = data_loader_func(tail_test_dataset, batch_size, device, shuffle=False)
    else:
        tail_test_data_loader = None

    return train_data_loader, valid_data_loader, test_data_loader, head_test_data_loader, tail_test_data_loader


class Dataset(object):
    def __init__(self, name, batch_size=4096, device="cpu", distortion="none"):
        self.name = name

        if name == "avazu":
            assert distortion == "none"
            dataset, self.field_dims = load_avazu_dataset()
            self.feature_groups = None

        elif name == "afn-movielens":
            dataset, self.field_dims = load_afn_movielens_dataset(distortion=distortion)
            self.feature_groups = dataset[0].feature_groups

        elif name == "book-crossing":
            assert distortion == "none"
            dataset, self.field_dims = load_book_crossing_dataset()
            self.feature_groups = None

        elif name == "jester":
            dataset, self.field_dims = load_jester_dataset(distortion=distortion)
            self.feature_groups = dataset[0].feature_groups

        else:
            raise ValueError("unknown dataset name: " + self.name)

        self.train_data_loader, self.valid_data_loader, self.test_data_loader, self.head_test_data_loader, self.tail_test_data_loader = _get_data_loaders(
            dataset, batch_size, device=device)

        self.metrics = self._init_metrics()
        self.criterion = self.metrics["criterion"]

    def _init_metrics(self):
        if self.name in [
            "avazu",
            "afn-movielens",
            "book-crossing",
            "jester",
        ]:
            return {
                "criterion": torch.nn.BCEWithLogitsLoss(),
                "auroc": auroc,
            }

        else:
            raise ValueError("unknown dataset name: " + self.name)
