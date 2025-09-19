from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, use_pos_enc=True, use_residual=True) -> None:
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

        self.use_pos_enc = use_pos_enc
        self.use_residual = use_residual

        print("Transformer block")
        print(f"Position Encoding {self.use_pos_enc}")
        print(f"Residual {self.use_residual}")

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        if self.use_pos_enc:
            pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        else:
            pos_enc = torch.zeros_like(k).to(k.device)
        # pos_enc = torch.normal(mean=torch.mean(pos_enc).item(), std=torch.std(pos_enc).item(), size=pos_enc.shape).to(k.device)
        # pos_enc = torch.ones_like(k).to(k.device) * torch.mean(torch.abs(pos_enc))

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        # attn = self.fc_gamma(q[:, :, None] - k)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res)

        if self.use_residual:
            res = res + pre
        return res, attn


class MLPBlock(nn.Module):
    def __init__(self, d_points, d_model, k, use_pos_enc=True, use_residual=True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(2 * d_model, d_points)
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

        self.use_pos_enc = use_pos_enc
        self.use_residual = use_residual

        print("MLP block")
        print(f"Position Encoding {self.use_pos_enc}")
        print(f"Residual {self.use_residual}")

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        if self.use_pos_enc:
            pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        else:
            pos_enc = torch.zeros_like(k).to(k.device)

        diff = self.fc_gamma(q[:, :, None] - k + pos_enc)

        # Concatenate attn and v along the last dimension (-1)
        concat_feat = torch.cat([diff, v + pos_enc], dim=-1)  # b x n x k x (2f)
        B, N, K, C = concat_feat.size()
        concat_feat = concat_feat.view(-1, C)
        concat_feat = self.fc2(concat_feat).view(B, N, K, -1)    # b x n x k x d_points
        res = concat_feat.max(dim=-2, keepdim=False)[0]     # b x n x d_points

        if self.use_residual:
            res = res + pre
        return res, diff
