import torch
from adamp import AdamP


def filter_layer_norm(name):
    return name.endswith("layer_norm.weight") or name.endswith("layer_norm.bias")


def filter_not_layer_norm(name):
    return not filter_layer_norm(name)


def filter_layer_norm_or_bias(name):
    return name.endswith("layer_norm.weight") or name.endswith("layer_norm.bias") or name.endswith("bias_weight")


def filter_not_layer_norm_and_bias(name):
    return not filter_layer_norm_or_bias(name)


class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def get_optimizer(optimizer_name, model, lr, weight_decay=0.0, filter=lambda x: True, sparse_embedding=False):
    parameters = [p for name, p in model.named_parameters() if filter(name)]
    if not parameters:
        return None

    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)

    elif optimizer_name == "sgdm":
        assert not sparse_embedding
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)

    elif optimizer_name == "adam":
        if sparse_embedding:
            sparse_parameters = []
            dense_parameters = []
            for name, p in model.named_parameters():
                if name.endswith("embedding.weight"):
                    sparse_parameters.append(p)
                else:
                    dense_parameters.append(p)
            sparse_adam = torch.optim.SparseAdam(sparse_parameters, lr=lr)
            dense_adam = torch.optim.Adam(dense_parameters, lr=lr, weight_decay=weight_decay)
            optimizer = MultipleOptimizer(sparse_adam, dense_adam)
            return optimizer
        else:
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    elif optimizer_name == "adame":
        assert not sparse_embedding
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, eps=1e-3)

    elif optimizer_name == "adamw":
        assert not sparse_embedding
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    elif optimizer_name == "adamp":
        assert not sparse_embedding
        return AdamP(parameters, lr=lr, weight_decay=weight_decay)

    else:
        raise NotImplementedError()
