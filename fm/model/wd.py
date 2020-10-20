import torch

from fm.model.layers.layer import FeaturesEmbedding
from fm.model.layers.layer import FeaturesLinear
from fm.model.layers.layer import MultiLayerPerceptronV2


class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, normalize, sparse_embedding, mlp_dims):
        super().__init__()
        self.linear = FeaturesLinear(field_dims, sparse_embedding=sparse_embedding)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, sparse_embedding=sparse_embedding,
                                           normalize=normalize)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptronV2(self.embed_output_dim, mlp_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1)
