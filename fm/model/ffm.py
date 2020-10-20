import torch

from fm.model.layers.layer import FeaturesBias
from fm.model.layers.layer import FieldAwareFactorizationMachine


class FieldAwareFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.

    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim, normalize="none", sparse_embedding=False):
        super().__init__()
        self.features_bias = FeaturesBias(field_dims, sparse_embedding=sparse_embedding)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim, normalize=normalize,
                                                  sparse_embedding=sparse_embedding)

        if normalize in ["none", "ln", "ten", "voln", "sln"]:
            pass

        elif normalize == "bn1d":
            feature_embeddings = self.ffm.feature_embeddings
            feature_embedding = feature_embeddings[0]
            for _feature_embedding in feature_embeddings:
                _feature_embedding.batch_norm.weight = feature_embedding.batch_norm.weight
                _feature_embedding.batch_norm.bias = feature_embedding.batch_norm.weight

        else:
            raise NotImplementedError()

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.features_bias(x) + ffm_term
        return x.squeeze(1)

    def get_feature_embeddings(self):
        return self.ffm.feature_embeddings
