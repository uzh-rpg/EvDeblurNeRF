import torch
from torch import nn
import torch.nn.functional as F


class ViewEmbedding(nn.Module):
    """
    Simple view embedding, as used in PDRF
    """
    def __init__(self, num_embed, embed_dim, init_params="zero", **kwargs):
        super(ViewEmbedding, self).__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.out_channels = embed_dim

        if init_params == "zero":
            self.register_parameter(
                "img_embed",
                nn.Parameter(torch.zeros(num_embed, embed_dim).type(torch.float32), True))
        elif init_params == "normal":
            self.register_parameter(
                "img_embed",
                nn.Parameter(torch.randn(num_embed, embed_dim).type(torch.float32), True))
        elif init_params == "linspace":
            self.register_parameter(
                "img_embed",
                nn.Parameter(torch.linspace(-1, 1, num_embed)[:, None].repeat(1, embed_dim).type(torch.float32), True))
        else:
            raise ValueError("Unknown init_params: {}".format(init_params))

    def forward(self, x):
        return self.img_embed[x]


class ViewEmbeddingMLP(ViewEmbedding):
    """
    View embedding followed by a MLP, as used in DP-NeRF
    """
    def __init__(self, D, W, skips, num_embed, embed_dim, init_params="zero", **kwargs):
        super(ViewEmbeddingMLP, self).__init__(num_embed, embed_dim, init_params, **kwargs)
        self.D = D
        self.W = W
        self.skips = skips
        self.out_channels = W

        self.view_embed_linears = nn.ModuleList(
            [nn.Linear(self.embed_dim, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.embed_dim, W) for i in range(D - 1)]
        )

    def forward(self, x):
        view_embedded = super().forward(x)

        input_views_embedded = view_embedded
        h = view_embedded
        for i, l in enumerate(self.view_embed_linears):
            h = self.view_embed_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_views_embedded, h], -1)
        view_feature = h
        return view_feature


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)


def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim
