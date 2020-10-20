import torch

from fm.model.layers.layer import FeaturesEmbedding
from fm.model.layers.layer import MultiLayerPerceptronV2


class DeepNeuralNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of dnn.
    """

    def __init__(self, field_dims, embed_dim, normalize, sparse_embedding, mlp_dims):
        super().__init__()
        self.embedding = FeaturesEmbedding(
            field_dims,
            embed_dim,
            normalize=normalize,
            sparse_embedding=sparse_embedding,
        )
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptronV2(self.embed_output_dim, mlp_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1)
