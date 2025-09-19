import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction, square_distance, index_points
from .transformer import TransformerBlock, MLPBlock


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print(cfg.model)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        if cfg.model.use_attention:
            basic_block = TransformerBlock
        else:
            basic_block = MLPBlock
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = basic_block(32, cfg.model.transformer_dim, nneighbor, cfg.model.use_pos_enc, cfg.model.use_residual)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(basic_block(channel, cfg.model.transformer_dim, nneighbor, cfg.model.use_pos_enc, cfg.model.use_residual))
        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        attns = []
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points, attn = self.transformers[i](xyz, points)
            xyz_and_feats.append((xyz, points))
            attns.append(attn)
        return points, xyz_and_feats, attns


class Transition(nn.Module):
    def __init__(self, nsample, in_channel, mlp):
        super(Transition, self).__init__()
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        dists = square_distance(xyz, xyz)  # B x N x N
        idx = dists.argsort()[:, :, :self.nsample]  # B x N x K

        grouped_xyz = index_points(xyz, idx)  # [B, N, nsample, C]
        torch.cuda.empty_cache()
        grouped_xyz_norm = grouped_xyz - xyz.view(B, N, 1, C)
        torch.cuda.empty_cache()

        grouped_points = index_points(points, idx)
        points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, N, nsample, C+D]

        points = points.permute(0, 3, 2, 1) # [B, C+D, nsample, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points =  F.relu(bn(conv(points)))

        points = torch.max(points, 2)[0].transpose(1, 2)
        return points


class BackboneNoDownSampling(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor, cfg.model.use_pos_enc)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                Transition(nneighbor, channel // 2 + 3, [channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor, cfg.model.use_pos_enc))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        attns = []
        for i in range(self.nblocks):
            points = self.transition_downs[i](xyz, points)
            points, attn = self.transformers[i](xyz, points)
            xyz_and_feats.append((xyz, points))
            attns.append(attn)
        return points, xyz_and_feats, attns


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("Using Hengshuang's model")
        if cfg.model.downsample:
            self.backbone = Backbone(cfg)
        else:
            print("Using non-downsampling model")
            self.backbone = BackboneNoDownSampling(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, x, return_feat=False):
        points, xyz_and_feats, attns = self.backbone(x)
        res = self.fc2(points.mean(1))
        if return_feat:
            return res, xyz_and_feats, attns
        return res


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
            
        return self.fc3(points)


    