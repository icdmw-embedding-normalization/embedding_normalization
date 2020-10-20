import torch

from fm.model.layers.layer import CompressedInteractionNetwork
from fm.model.layers.layer import FeaturesEmbedding
from fm.model.layers.layer import FeaturesLinear
from fm.model.layers.layer import MultiLayerPerceptronV2


class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, normalize, sparse_embedding, mlp_dims, cross_layer_sizes,
                 split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, normalize=normalize,
                                           sparse_embedding=sparse_embedding)
        self.linear = FeaturesLinear(field_dims, sparse_embedding=sparse_embedding)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

        self.mlp = MultiLayerPerceptronV2(self.embed_output_dim, mlp_dims, output_layer=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        y = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return y.squeeze(1)
