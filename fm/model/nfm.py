import torch

from fm.model.layers.layer import FactorizationMachine
from fm.model.layers.layer import FeaturesEmbedding
from fm.model.layers.layer import MultiLayerPerceptronV2
from fm.model.layers.layer import FeaturesLinear


class NeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(
            self, field_dims, embed_dim, normalize, sparse_embedding, mlp_dims,
            embedding_dropout=0.5,
            hidden_dropout=0.3,
            embedding_bn=True,
            layer_bn=True,
    ):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, normalize=normalize,
                                           sparse_embedding=sparse_embedding)
        self.linear = FeaturesLinear(field_dims, sparse_embedding=sparse_embedding)

        if embedding_bn:
            self.fm = torch.nn.Sequential(
                FactorizationMachine(reduce_sum=False),
                torch.nn.BatchNorm1d(embed_dim),
                torch.nn.Dropout(embedding_dropout)
            )
        else:
            self.fm = torch.nn.Sequential(
                FactorizationMachine(reduce_sum=False),
                torch.nn.Dropout(embedding_dropout)
            )
        self.mlp = MultiLayerPerceptronV2(
            embed_dim,
            mlp_dims,
            output_layer=True,
            dropout=hidden_dropout,
            normalize="bn" if layer_bn else "none",
        )

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
        return x.squeeze(1)
