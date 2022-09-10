import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(DenseLayer, self).__init__()

        self.linear_layer = nn.Linear(in_dim, out_dim)

        if activation is None:
            self.activation = nn.ReLU()
        else:
            self.activation = activation

    def forward(self, x):
        out = self.linear_layer(x)
        out = self.activation(out)
        return out


class GeometryDecoder(nn.Module):
    def __init__(self, W=128, D=4, skips=[], input_feat_dim=16, n_freq=-1, weight_norm=False, concat_qp=False):
        super(GeometryDecoder, self).__init__()

        self.embed_fn, input_ch = get_embedder(n_freq, input_dim=input_feat_dim + concat_qp * 3)
        self.W = W
        self.D = D
        self.skips = skips
        layers = []

        for l in range(D+1):
            if l == D:
                out_dim = 1
            elif l + 1 in self.skips:
                out_dim = W - input_ch
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, return_h=False):
        feat = self.embed_fn(feat)
        h = feat
        
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)

        if return_h:  # return feature
            return h[..., :1], h[..., 1:]
        else:
            return h[..., :1]


class RadianceDecoder(nn.Module):
    def __init__(self, W=64, D=4, skips=[], use_view_dirs=False, use_normals=False,
                 input_feat_dim=64, n_freq=4, weight_norm=False, concat_qp=False, use_dot_prod=False):
        super(RadianceDecoder, self).__init__()
        if use_view_dirs or use_normals or use_dot_prod:
            self.embed_fn, input_ch = get_embedder(n_freq, use_view_dirs * 3 + use_normals * 3 + concat_qp * 3 + use_dot_prod * 1)
            input_ch += input_feat_dim
        else:
            self.embed_fn = None
            input_ch = input_feat_dim

        self.use_view_dirs = use_view_dirs
        self.W = W
        self.D = D
        self.skips = skips
        layers = []

        for l in range(D+1):
            if l == D:
                out_dim = 3
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            elif l in self.skips:
                in_dim = input_ch + W
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, radiance_input):
        h = radiance_input
        
        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)

        return h


class NeRFDecoder(nn.Module):
    def __init__(self, geometry_kwargs, radiance_kwargs, sdf_feat_dim, rgb_feat_dim):
        super(NeRFDecoder, self).__init__()
        self.geometry_net = GeometryDecoder(**geometry_kwargs, input_feat_dim=sdf_feat_dim)
        self.radiance_net = RadianceDecoder(**radiance_kwargs, input_feat_dim=rgb_feat_dim)

    def forward(self, feat, view_dirs=None):
        if view_dirs is not None:
            geometry, h = self.geometry_net(feat, return_h=True)
            rgb = self.radiance_net(h, view_dirs=view_dirs)
            return torch.cat([geometry, rgb], dim=-1)
        else:
            return self.geometry_net(feat, return_h=False)
