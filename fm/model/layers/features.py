import math

import numpy as np
import torch

SCALE_GRAD_BY_FREQ = False


def _get_mlp(embed_dim, n_output=1):
    linear_input = torch.nn.Linear(embed_dim, embed_dim)
    linear_output = torch.nn.Linear(embed_dim, n_output)

    return torch.nn.Sequential(linear_input, torch.nn.ReLU(), linear_output)


class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1, sparse_embedding=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), output_dim, sparse=sparse_embedding,
                                            scale_grad_by_freq=SCALE_GRAD_BY_FREQ)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        torch.nn.init.zeros_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.embedding(x), dim=1) + self.bias


class FeaturesBias(FeaturesLinear):
    def __init__(self, field_dims, sparse_embedding=False):
        super().__init__(field_dims, output_dim=1, sparse_embedding=sparse_embedding)


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, normalize="none", sparse_embedding=False):
        super().__init__()

        self.normalize = normalize
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim, sparse=sparse_embedding,
                                            scale_grad_by_freq=SCALE_GRAD_BY_FREQ)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        std = 0.5 / math.sqrt(len(field_dims) * (len(field_dims) - 1) / 2 * embed_dim)
        torch.nn.init.normal_(self.embedding.weight.data, std=std)

        if normalize == "none":
            pass

        elif normalize == "ln":
            self.layer_norm = torch.nn.LayerNorm(embed_dim)
            torch.nn.init.constant_(self.layer_norm.weight, std)

        elif normalize == "en":
            self.layer_norm = torch.nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.layer_norm_weight_embedding = torch.nn.Embedding(sum(field_dims), 1, sparse=sparse_embedding,
                                                                  scale_grad_by_freq=SCALE_GRAD_BY_FREQ)
            self.layer_norm_bias_embedding = torch.nn.Embedding(sum(field_dims), 1, sparse=sparse_embedding,
                                                                scale_grad_by_freq=SCALE_GRAD_BY_FREQ)
            torch.nn.init.constant_(self.layer_norm_weight_embedding.weight, std)
            torch.nn.init.constant_(self.layer_norm_bias_embedding.weight, 0.0)

        elif normalize == "voln":
            pass

        elif normalize == "sln":
            self.layer_norm = torch.nn.LayerNorm(embed_dim, elementwise_affine=False)

        elif normalize == "bn":
            self.batch_norm = torch.nn.BatchNorm1d(embed_dim)
            torch.nn.init.constant_(self.batch_norm.weight, std)

        else:
            print(normalize)
            raise NotImplementedError()

    def _normalize_embed(self, embed, x, field_idx=None):
        if self.normalize == "none":
            return embed

        elif self.normalize == "ln":
            return self.layer_norm(embed)

        elif self.normalize == "en":
            layer_norm_weight = self.layer_norm_weight_embedding(x)
            layer_norm_bias = self.layer_norm_bias_embedding(x)
            return self.layer_norm(embed) * layer_norm_weight + layer_norm_bias

        elif self.normalize == "voln":
            return embed / torch.sqrt(
                ((embed - embed.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True) + 1e-5)

        elif self.normalize == "sln":
            return self.layer_norm(embed)

        elif self.normalize == "bn":
            embed = embed.reshape([-1, self.embed_dim])
            embed = self.batch_norm(embed)
            if field_idx is None:
                embed = embed.reshape([-1, len(self.field_dims), self.embed_dim])

            return embed

        else:
            print(self.normalize)
            raise NotImplementedError()

    def forward(self, x, field_idx=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if field_idx is None:
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
        else:
            x = x + x.new_tensor(self.offsets[field_idx]).unsqueeze(0)

        embed = self.embedding(x)
        embed = self._normalize_embed(embed, x, field_idx=field_idx)
        return embed
