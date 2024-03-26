import torch
import numpy as np
from tqdm import tqdm
from torch import nn as nn


class CRF(nn.Module):

    def __init__(self, map_type: str, gamma: float = 2.2, init_identity=False, extra_features=0):
        super(CRF, self).__init__()
        assert map_type in ['none', 'gamma', 'learn']
        self.map_type = map_type
        self.gamma = gamma
        self.extra_features = extra_features

        if map_type == 'learn':
            base_features = 1
            self.linear = nn.Sequential(
                nn.Linear(base_features + extra_features, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )
            self.linear_ch_in = base_features + extra_features

            if init_identity:
                self.init_identity()

    def init_identity(self):
        if self.map_type != 'learn':
            return

        batch_size = 64
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        pbar = tqdm(range(3000), desc='CRF init identity')
        gen = torch.Generator(device="cuda").manual_seed(42)
        for _ in pbar:
            x = torch.rand(batch_size, 3, generator=gen)

            ori_shape = x.shape
            x_in = x.reshape(-1, 1)
            if self.extra_features > 0:
                x_feat = torch.zeros([x_in.shape[0], self.extra_features], device=x.device)
                x_feat = torch.cat([x_in, x_feat], dim=-1)
            else:
                x_feat = x_in
            res_x = self.linear(x_feat) * 0.1
            x_out = torch.sigmoid(res_x + x_in)
            y = x_out.reshape(ori_shape)

            loss = torch.mean((y - x) ** 2)
            pbar.set_postfix({'loss': loss.item()})

            optim.zero_grad()
            loss.backward()
            optim.step()
        optim.zero_grad()

    def forward(self, x, x_feat=None, skip_learn=False):
        """
        Perform linear to gamma corrected space
        """

        if self.map_type == 'none':
            return x

        if 'gamma' in self.map_type:
            x = x ** (1. / self.gamma)

        x_feat_in = x_feat
        if not skip_learn and self.map_type == 'learn':
            ori_shape = x.shape
            x_in = x.reshape(-1, 1)
            if x_feat_in is not None and self.extra_features > 0:
                x_feat_in = x_feat_in.to(x_in.dtype)
                if x_feat_in.ndim != 3:
                    x_feat_in = x_feat_in[:, None].repeat(1, 3, 1)
                x_feat_in = x_feat_in.reshape(-1, self.extra_features)
                x_feat_in = torch.cat([x_in, x_feat_in], dim=-1)
            else:
                x_feat_in = x_in

            if x_feat is None and self.extra_features > 0:
                # Zero pad the tensor if no extra features are provided
                x_pad = torch.zeros([*x_feat_in.shape[:-1], self.linear_ch_in-x_feat_in.shape[-1]], device=x.device)
                x_feat_in = torch.cat([x_feat_in, x_pad], dim=-1)

            res_x = self.linear(x_feat_in) * 0.1
            x_out = torch.sigmoid(res_x + x_in)
            x_out = x_out.reshape(ori_shape)
            return x_out
        else:
            return x


class TonemappingTransform(nn.Module):

    def __init__(self, map_type_rgb: str, map_type_event: str, gamma: float = 2.2,
                 luma_standard="rec601", init_learn_identity=False,
                 extra_features_event=0, extra_features_rgb=0):
        super(TonemappingTransform, self).__init__()

        self.tonemapping_rgb = CRF(map_type_rgb, gamma, init_identity=init_learn_identity,
                                   extra_features=extra_features_rgb)
        self.tonemapping_event = CRF(map_type_event, gamma, init_identity=init_learn_identity,
                                     extra_features=extra_features_event)

        self.luma_standard = luma_standard
        assert luma_standard in ["rec601", "rec709", "avg"]

    def encode_rgb(self, x, skip_learn_crf=False, rgb_extra_feat=None, **kwargs):
        """
        Perform linear to gamma corrected space
        """
        assert x.shape[-1] == 3 and isinstance(x, torch.Tensor)

        x = self.tonemapping_rgb(x, skip_learn=skip_learn_crf, x_feat=rgb_extra_feat)
        return x

    def encode_luma(self, x, keep_rgb=False, tonemap_only=False, skip_learn_crf=False, ev_extra_feat=None, **kwargs):
        """
        Permon linear to gamma corrected event space
        :return:
        """

        x = self.tonemapping_event(x, skip_learn=skip_learn_crf, x_feat=ev_extra_feat)
        if not tonemap_only:
            if self.luma_standard == "rec601":
                x = 0.299 * x[..., [0]] + 0.587 * x[..., [1]] + 0.114 * x[..., [2]]
            elif self.luma_standard == "rec709":
                x = 0.2126 * x[..., [0]] + 0.7152 * x[..., [1]] + 0.0722 * x[..., [2]]
            elif self.luma_standard == "avg":
                x = x.mean(axis=-1, keepdims=True)
            else:
                raise ValueError(f"Unknown luma_standard {self.luma_standard}")

            if keep_rgb:
                x = torch.cat([x] * 3, axis=-1)
        return x

    def forward(self, x, mode="encode", chunk=None, **kwargs):
        device = x.device
        chunk = chunk or max(1, x.shape[0])

        x_res = []
        for i in range(0, max(1, x.shape[0]), chunk):
            x_chunk = x[i:i + chunk].cuda()
            if mode == "encode_rgb":
                x_res.append(self.encode_rgb(x_chunk, **kwargs).to(device))
            elif mode == "encode_luma":
                x_res.append(self.encode_luma(x_chunk, **kwargs).to(device))
            else:
                raise RuntimeError(f"mode '{mode}' not recognized")
        return torch.cat(x_res, dim=0)
