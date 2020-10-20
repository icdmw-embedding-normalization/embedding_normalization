import math
from datetime import datetime

import torch
import tqdm

from common.utils import saver
from common.utils.early_stopper import EarlyStopper
from common.utils.logger import PerformanceLogger
from common.utils.optim import get_optimizer
from common.utils.seed import set_seed
from .dataset.dataset import Dataset
from .logger import IterLogger
from .model.loader import load_model


def _validate(model, data_loader, device, metrics):
    if data_loader is None:
        return None

    model.eval()
    performance_dict = {}
    cnt = 0
    with torch.no_grad():
        _tqdm = tqdm.tqdm(data_loader, total=len(data_loader))

        for fields, target in _tqdm:
            fields, target = fields.to(device), target.to(device)
            _cnt = fields.shape[0]

            y = model(fields)
            for key, metric in metrics.items():
                _performance = metric(y, target.float())
                if torch.is_tensor(_performance):
                    _performance = _performance.cpu().numpy()
                performance_dict[key] = performance_dict.get(key, 0) + _performance * _cnt
            cnt += _cnt

            _tqdm.set_description(f"{performance_dict['criterion'] / cnt:.6f}")

    for key, _performance in performance_dict.items():
        performance_dict[key] = _performance / cnt
    return performance_dict


def _is_nan(valid_performance_dict, test_performance_dict):
    values = []
    values.extend(valid_performance_dict.values())
    values.extend(test_performance_dict.values())

    for value in values:
        if math.isnan(value):
            return True
    return False


def _get_weight_dict(model, dataset, device):
    if dataset.feature_groups is None:
        return None

    weight_dict = {}
    for key, (x, field_idx) in dataset.feature_groups.items():
        x = x.to(device)
        w = torch.stack(
            [feature_embedding(x, field_idx=field_idx) for feature_embedding in model.get_feature_embeddings()]
        )
        weight_dict[key] = w
    return weight_dict


def _weight_delta_dict(prev_weight_dict, curr_weight_dict):
    delta_dict = {}
    for key in prev_weight_dict.keys():
        prev_weight = prev_weight_dict[key]
        curr_weight = curr_weight_dict[key]

        delta = torch.abs(prev_weight - curr_weight).mean().cpu().data.numpy()
        delta_dict[key] = delta
    return delta_dict


def _is_logged_step(iter_one_step, total_iter, iter_idx_in_epoch, len_train_data_loader):
    if iter_one_step is None or len_train_data_loader <= iter_one_step:
        return iter_idx_in_epoch == (len_train_data_loader - 1)
    return total_iter % iter_one_step == 0


def _train(cfg):
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    dataset = Dataset(cfg.dataset_name, batch_size=cfg.batch_size, device=device, distortion=cfg.dataset_distortion)

    model = load_model(cfg.model_name, dataset, embed_dim=cfg.embed_dim, sparse_embedding=cfg.sparse_embedding).to(
        device)
    print("Load success!!")
    if cfg.multi_gpu:
        model = torch.nn.DataParallel(model)
    print(model)

    criterion = dataset.criterion.to(device)

    optimizer = get_optimizer(cfg.optimizer, model, lr=cfg.lr, weight_decay=cfg.weight_decay,
                              sparse_embedding=cfg.sparse_embedding)

    if cfg.log_performance:
        performance_iter_logger = IterLogger(
            cfg.model_name,
            cfg.dataset_name,
            label="performance",
            sparse_embedding=cfg.sparse_embedding,
            distortion=cfg.dataset_distortion
        )
    else:
        performance_iter_logger = None

    if cfg.log_weight and dataset.feature_groups is not None:
        weight_delta_iter_logger = IterLogger(
            cfg.model_name,
            cfg.dataset_name,
            label="weight_delta",
            sparse_embedding=cfg.sparse_embedding,
            distortion=cfg.dataset_distortion
        )
        weight_norm_iter_logger = IterLogger(
            cfg.model_name,
            cfg.dataset_name,
            label="weight_norm",
            sparse_embedding=cfg.sparse_embedding,
            distortion=cfg.dataset_distortion
        )
    else:
        weight_delta_iter_logger = None
        weight_norm_iter_logger = None

    early_stopper = EarlyStopper(cfg.early_stop_steps)
    performance_logger = PerformanceLogger()

    total_iter = 0
    total_loss = 0.0
    cnt = 0
    early_stopped = False

    model.eval()
    if cfg.log_weight and dataset.feature_groups is not None:
        prev_weight_dict = _get_weight_dict(model, dataset, device)
    else:
        prev_weight_dict = None

    model.train()

    for epoch_idx in range(1, cfg.max_epochs + 1):
        _tqdm = tqdm.tqdm(enumerate(dataset.train_data_loader, start=1), total=len(dataset.train_data_loader))

        for iter_idx_in_epoch, (fields, target) in _tqdm:
            total_iter += 1
            fields, target = fields.to(device), target.to(device)
            y = model(fields)

            loss = criterion(y, target.float())
            total_loss += loss * fields.shape[0]
            cnt += fields.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_description(f"{total_loss / cnt:.6f}")

            if cfg.log_weight and dataset.feature_groups is not None:
                curr_weight_dict = _get_weight_dict(model, dataset, device)
                weight_delta_dict = _weight_delta_dict(prev_weight_dict, curr_weight_dict)

                weight_delta_iter_logger.write(
                    total_iter,
                    [
                        weight_delta_dict['head'],
                        weight_delta_dict['tail'],
                    ],
                    print_std=(total_iter % cfg.iter_one_step == 0),
                )

                weight_norm_iter_logger.write(
                    total_iter,
                    [
                        (curr_weight_dict[key] ** 2).mean().detach().cpu().numpy()
                        for key in ["head", "tail"]
                    ],
                    print_std=(total_iter % cfg.iter_one_step == 0),
                )
                prev_weight_dict = _get_weight_dict(model, dataset, device)

            if _is_logged_step(cfg.iter_one_step, total_iter, iter_idx_in_epoch, len(dataset.train_data_loader)):
                valid_performance_dict = _validate(model, dataset.valid_data_loader, device, dataset.metrics)
                test_performance_dict = _validate(model, dataset.test_data_loader, device, dataset.metrics)

                if _is_nan(valid_performance_dict, test_performance_dict):
                    break

                train_criterion = total_loss / cnt
                total_loss = 0.0
                cnt = 0

                performance_logger.update(
                    total_iter,
                    train_criterion,
                    valid_performance_dict["criterion"],
                    test_performance_dict["criterion"],
                    valid_acc=valid_performance_dict.get("auroc", None),
                    test_acc=test_performance_dict.get("auroc", None),
                )

                if performance_iter_logger is not None:
                    performance_iter_logger.write(
                        total_iter,
                        [
                            valid_performance_dict["criterion"],
                            test_performance_dict["criterion"],
                            valid_performance_dict.get("auroc", None),
                            test_performance_dict.get("auroc", None),
                        ],
                        print_std=True,
                    )

                model.train()

                if early_stopper.step(valid_performance_dict["criterion"]) == EarlyStopper.FINISH:
                    early_stopped = True
                    break

        if early_stopped:
            break

    saver.save(
        (
            f"{datetime.now()}\t{cfg.dataset_name}\t{cfg.dataset_distortion}\t{cfg.model_name}\t{cfg.embed_dim}"
            f"\t{cfg.seed}\t{cfg.lr}\t{cfg.weight_decay}"
            f"\t{epoch_idx}\t{total_iter}"
            f"\t{performance_logger.min_valid_test_loss}\t{performance_logger.min_valid_test_acc}"
        ),
        filename=f"{cfg.saver_filename}.tsv"
    )

    if cfg.log_weight and dataset.feature_groups is not None:
        weight_norm_iter_logger.close()
        weight_delta_iter_logger.close()

    if performance_iter_logger is not None:
        performance_iter_logger.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset-name",
                        default="book-crossing",
                        help="")
    parser.add_argument("--dataset-distortion",
                        default="none",
                        choices=["none", "dummy-field", "label-shuffle"],
                        help="")
    parser.add_argument("--model-name",
                        default="fm",
                        help="")
    parser.add_argument("--optimizer",
                        default="adam",
                        help="")
    parser.add_argument("--max-epochs",
                        type=int,
                        default=100,
                        help="")
    parser.add_argument("--early-stop-steps",
                        type=int,
                        default=30,
                        help="")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="")
    parser.add_argument("--batch-size", type=int,
                        default=4096,
                        help="full batch if 0.")
    parser.add_argument("--weight-decay",
                        type=float,
                        default=0.0,
                        help="")
    parser.add_argument("--embed-dim",
                        type=int,
                        default=10,
                        help="")
    parser.add_argument("--multi-gpu",
                        action="store_true",
                        default=False,
                        help="")
    parser.add_argument("--model-file-name",
                        type=str,
                        default="fm-default",
                        help="")
    parser.add_argument("--iter-one-step",
                        type=int,
                        default=100,
                        help="")
    parser.add_argument("--log-weight",
                        action="store_true",
                        default=False,
                        help="")
    parser.add_argument("--log-performance",
                        action="store_true",
                        default=False,
                        help="")
    parser.add_argument("--saver-filename",
                        default="fm",
                        help="")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.sparse_embedding = True
    print(args)
    _train(args)


if __name__ == "__main__":
    main()
