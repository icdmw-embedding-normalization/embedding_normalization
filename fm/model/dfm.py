import torch

from fm.model.layers.layer import FactorizationMachine
from fm.model.layers.layer import FeaturesEmbedding
from fm.model.layers.layer import FeaturesLinear
from fm.model.layers.layer import MultiLayerPerceptronV2


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, normalize, sparse_embedding, mlp_dims):
        super().__init__()
        self.linear = FeaturesLinear(field_dims, sparse_embedding=sparse_embedding)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, sparse_embedding=sparse_embedding,
                                           normalize=normalize)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptronV2(self.embed_output_dim, mlp_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1)
