from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

        # cls token
        self.fc1_cls = nn.Linear(d_points, d_model)
        self.fc2_cls = nn.Linear(d_model, d_points)
        self.w_qs_cls = nn.Linear(d_model, d_model, bias=False)
        self.w_ks_cls = nn.Linear(d_model, d_model, bias=False)
        self.w_vs_cls = nn.Linear(d_model, d_model, bias=False)

    # xyz: b x (n + 1) x 3, features: b x (n + 1) x f
    def forward(self, xyz, features):
        cls_token_xyz = xyz[..., 0, :].unsqueeze(-2)        # b, 1, 3
        xyz = xyz[..., 1:, :]

        cls_token_feat = features[..., 0, :].unsqueeze(-2)    # b, 1, f
        features = features[..., 1:, :]

        # local points
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre

        # cls token (global attention)
        cls_token_pre = cls_token_feat
        cls_token_x = self.fc1_cls(cls_token_feat)
        q, k, v = self.w_qs_cls(cls_token_x), self.w_ks_cls(x), self.w_vs_cls(x)    # q has shape (b, 1, f), k, v has shape (b, n, f)

        cls_token_attn = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(k.size(-1))     # b x 1 x n
        cls_token_attn = F.softmax(cls_token_attn, dim=-1)  # b x 1 x n

        cls_token_res = torch.einsum('bmn,bnf->bmf', cls_token_attn, v)
        cls_token_res = self.fc2_cls(cls_token_res) + cls_token_pre

        res = torch.cat((cls_token_res, res), dim=-2)

        return res, (attn, cls_token_attn)
