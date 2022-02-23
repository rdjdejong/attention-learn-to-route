import torch


class DeepSetLayer(torch.nn.Module):
    def __init__(self, embed_dim, pool='max'):
        super().__init__()

        assert pool in ('max', 'sum'), "Undefined pooling type"

        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        self.non_linear = torch.nn.ReLU()
        self.pool = {
            'max': torch.max,
            'sum': torch.sum
        }[pool]

    def forward(self, X):
        return self.non_linear(
            self.linear(X - self.pool(X, dim=-1, keepdims=True).values)
        )


class DeepSetEncoder(torch.nn.Module):
    def __init__(self, embed_dim, n_layers, pool='max'):
        super().__init__()

        self.layers = torch.nn.Sequential(*(
            DeepSetLayer(embed_dim, pool) for _ in range(n_layers)
        ))

    def forward(self, X):
        return self.layers(X)
