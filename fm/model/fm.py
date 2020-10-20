from fm.model.layers.layer import FactorizationMachine
from fm.model.layers.layer import FeaturesBias
from fm.model.layers.layer import FeaturesEmbedding

from .base import BaseModel


class _FactorizationMachineModel(BaseModel):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, normalize="none", bias=True, sparse_embedding=False):
        super().__init__()
        self.features_embedding = FeaturesEmbedding(field_dims, embed_dim, normalize=normalize,
                                                    sparse_embedding=sparse_embedding)
        if bias:
            self.features_bias = FeaturesBias(field_dims, sparse_embedding=sparse_embedding)
        else:
            self.features_bias = None

        self.factorization_machine = FactorizationMachine(reduce_sum=True)

    def get_feature_embeddings(self):
        return [self.features_embedding]

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :param return_reg_loss: reg loss
        """
        embed = self.features_embedding(x)
        out = self.factorization_machine(embed)
        if self.features_bias is not None:
            out += self.features_bias(x)
        out = out.squeeze(1)
        return out


class FactorizationMachineModel(_FactorizationMachineModel):
    def __init__(self, field_dims, embed_dim, normalize="none", sparse_embedding=False):
        super().__init__(field_dims, embed_dim, normalize=normalize, bias=True, sparse_embedding=sparse_embedding)


class FactorizationMachineModelWithoutBias(_FactorizationMachineModel):
    def __init__(self, field_dims, embed_dim, normalize="none", sparse_embedding=False):
        super().__init__(field_dims, embed_dim, normalize=normalize, bias=False, sparse_embedding=sparse_embedding)
