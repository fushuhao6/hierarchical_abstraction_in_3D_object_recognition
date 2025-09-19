import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction, square_distance, index_points
from .transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

        # cls token
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = channels[0]
        for out_channel in channels[1:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        cls_token_xyz = xyz[..., 0, :].unsqueeze(-2)        # b, 1, 3
        xyz = xyz[..., 1:, :]

        cls_token_feat = points[..., 0, :].unsqueeze(-2)    # b, 1, c
        points = points[..., 1:, :]

        cls_token_feat = torch.cat((cls_token_xyz, cls_token_feat), dim=-1)     # b, 1, c + 3
        cls_token_feat = cls_token_feat.permute(0, 2, 1).unsqueeze(-1)           # b, c + 3, 1, 1
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            cls_token_feat = F.relu(bn(conv(cls_token_feat)))
        cls_token_feat = cls_token_feat.squeeze(-1).permute(0, 2, 1)                  # b, 1, 2c

        xyz, points = self.sa(xyz, points)

        xyz = torch.cat([cls_token_xyz, xyz], dim=-2)
        points = torch.cat([cls_token_feat, points], dim=-2)

        return xyz, points


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points, attn = self.transformer1(xyz, self.fc1(x))

        xyz_and_feats = [(xyz, points)]
        attns = [attn]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points, attn = self.transformers[i](xyz, points)
            xyz_and_feats.append((xyz, points))
            attns.append(attn)
        return points, xyz_and_feats, attns


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("Using ClsToken's model")
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
        self.cls_token = nn.Parameter(torch.randn(1, 1, 3))

    def forward(self, x, return_feat=False):
        b, n, _ = x.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_token, x], dim=-2)
        points, xyz_and_feats, attns = self.backbone(x)
        res = self.fc2(points[:, 0, :])
        if return_feat:
            return res, xyz_and_feats, attns
        return res


