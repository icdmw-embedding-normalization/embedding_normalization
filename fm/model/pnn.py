import torch

from fm.model.layers.layer import FeaturesEmbedding
from fm.model.layers.layer import FeaturesLinear
from fm.model.layers.layer import InnerProductNetwork
from fm.model.layers.layer import MultiLayerPerceptronV2
from fm.model.layers.layer import OuterProductNetwork


class ProductNeuralNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of inner/outer Product Neural Network.
    Reference:
        Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2019.
    """

    def __init__(self, field_dims, embed_dim, normalize, sparse_embedding, mlp_dims, method='inner'):
        super().__init__()
        num_fields = len(field_dims)
        if method == 'inner':
            self.pn = InnerProductNetwork()
        elif method == 'outer':
            raise NotImplementedError()
            self.pn = OuterProductNetwork(num_fields, embed_dim)
        else:
            raise ValueError('unknown product type: ' + method)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim, sparse_embedding=sparse_embedding,
                                           normalize=normalize)
        self.linear = FeaturesLinear(field_dims, embed_dim, sparse_embedding=sparse_embedding)

        self.embed_output_dim = num_fields * embed_dim
        self.mlp = MultiLayerPerceptronV2(
            num_fields * (num_fields - 1) // 2 + self.embed_output_dim,
            mlp_dims,
            normalize="none"
        )

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        cross_term = self.pn(embed_x)
        x = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        x = self.mlp(x)
        return x.squeeze(1)
