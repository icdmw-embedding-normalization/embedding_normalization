from .afi import AutomaticFeatureInteractionModel
from .afm import AttentionalFactorizationMachineModel
from .afn import AdaptiveFactorizationNetwork
from .dfm import DeepFactorizationMachineModel
from .dnn import DeepNeuralNetworkModel
from .ffm import FieldAwareFactorizationMachineModel
from .fm import FactorizationMachineModel
from .fm import FactorizationMachineModelWithoutBias
from .nfm import NeuralFactorizationMachineModel
from .pnn import ProductNeuralNetworkModel
from .wd import WideAndDeepModel
from .xdfm import ExtremeDeepFactorizationMachineModel


def _split_name_and_normalize(model_name):
    tokens = model_name.split("-")
    if len(tokens) == 1:
        return model_name, "none"

    assert len(tokens) == 2
    name, normalize = tokens
    return name, normalize


def _warn_not_proper_embed_dim(embed_dim, target):
    if embed_dim != target:
        print(f"Warning... embed_dim should be {target} (current: {embed_dim}).")


def load_model(model_name, dataset, embed_dim, sparse_embedding=False):
    """
        Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    name, normalize = _split_name_and_normalize(model_name)

    if name == "fm":
        _warn_not_proper_embed_dim(embed_dim, 10)
        return FactorizationMachineModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
        )

    elif name == "fmwb":
        _warn_not_proper_embed_dim(embed_dim, 10)
        return FactorizationMachineModelWithoutBias(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
        )

    elif name == "ffm":
        _warn_not_proper_embed_dim(embed_dim, 10)
        return FieldAwareFactorizationMachineModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
        )

    elif name == "dfm":
        _warn_not_proper_embed_dim(embed_dim, 10)
        return DeepFactorizationMachineModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
            mlp_dims=(400, 400, 400),
        )

    elif name == "xdfm":
        _warn_not_proper_embed_dim(embed_dim, 10)
        return ExtremeDeepFactorizationMachineModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
            cross_layer_sizes=(200, 200),
            split_half=False,
            mlp_dims=(400, 400),
        )

    elif name == "afm":
        _warn_not_proper_embed_dim(embed_dim, 10)
        return AttentionalFactorizationMachineModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
            attn_size=embed_dim,
            dropouts=(0.0, 0.0),
        )

    elif name == "afn":
        _warn_not_proper_embed_dim(embed_dim, 10)
        return AdaptiveFactorizationNetwork(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
            LNN_dim=600,
            mlp_dims=(400, 400, 400),
            dropouts=(0, 0, 0),
        )

    elif name == "pnn":
        _warn_not_proper_embed_dim(embed_dim, 10)
        return ProductNeuralNetworkModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
            mlp_dims=(400,),
            method="inner",
        )

    elif name == "dnn":
        _warn_not_proper_embed_dim(embed_dim, 32)

        return DeepNeuralNetworkModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
            mlp_dims=(1024, 512, 256),
        )

    elif name == "wd":
        _warn_not_proper_embed_dim(embed_dim, 32)

        return WideAndDeepModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
            mlp_dims=(1024, 512, 256),
        )

    elif name == "nfm":
        _warn_not_proper_embed_dim(embed_dim, 64)

        if normalize == "none":
            return NeuralFactorizationMachineModel(
                field_dims,
                embed_dim=embed_dim,
                normalize=normalize,
                sparse_embedding=sparse_embedding,
                mlp_dims=(embed_dim,),
                embedding_dropout=0.5,
                hidden_dropout=0.3,
                embedding_bn=True,
                layer_bn=True,
            )
        else:
            return NeuralFactorizationMachineModel(
                field_dims,
                embed_dim=embed_dim,
                normalize=normalize,
                sparse_embedding=sparse_embedding,
                mlp_dims=(embed_dim,),
                embedding_dropout=0.0,
                hidden_dropout=0.0,
                embedding_bn=False,
                layer_bn=False,
            )

    elif name == "afi":
        _warn_not_proper_embed_dim(embed_dim, 16)

        return AutomaticFeatureInteractionModel(
            field_dims,
            embed_dim=embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
            atten_embed_dim=64,
            num_heads=2,
            num_layers=3,
            attention_dropout=0.0,
        )

    else:
        raise ValueError("unknown model name: " + name)
