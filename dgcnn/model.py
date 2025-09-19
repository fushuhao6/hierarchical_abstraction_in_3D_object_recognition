#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from util import PointNetSetAbstraction


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=40, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device(x.device)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx.view(-1), :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx  # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, output_everything=False):
        batch_size = x.size(0)
        x1, knn1 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x1 = self.conv1(x1)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        f1 = x1.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x2, knn2 = get_graph_feature(f1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x2 = self.conv2(x2)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        f2 = x2.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x3, knn3 = get_graph_feature(f2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x3 = self.conv3(x3)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        f3 = x3.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x4, knn4 = get_graph_feature(f3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x4 = self.conv4(x4)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        f4 = x4.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x5 = torch.cat((f1, f2, f3, f4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        f5 = self.conv5(x5)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        f6 = F.adaptive_max_pool1d(f5, 1).view(batch_size,
                                               -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        f7 = F.adaptive_avg_pool1d(f5, 1).view(batch_size,
                                               -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x8 = torch.cat((f6, f7), 1)  # (batch_size, emb_dims*2)

        f8 = F.leaky_relu(self.bn6(self.linear1(x8)),
                          negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x9 = self.dp1(f8)
        f9 = F.leaky_relu(self.bn7(self.linear2(x9)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x10 = self.dp2(f9)
        x = self.linear3(x10)  # (batch_size, 256) -> (batch_size, output_channels)

        if output_everything:
            return x, knn1, knn2, knn3, knn4, f1, f2, f3, f4, f5
        else:
            return x


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)              # B, num_points, 3
        points = points.permute(0, 2, 1)        # B, num_points, C

        xyz, points = self.sa(xyz, points)

        xyz = xyz.permute(0, 2, 1)              # B, 3, k
        points = points.permute(0, 2, 1)        # B, C, k

        return xyz, points


class DGCNN_Downsample(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Downsample, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv0 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(nn.Conv2d(32*2, 32, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.down1 = TransitionDown(256, 16, [32 + 3, 64, 64])
        self.down2 = TransitionDown(64, 16, [64 + 3, 128, 128])
        self.down3 = TransitionDown(16, 16, [128 + 3, 256, 256])
        self.down4 = TransitionDown(4, 16, [256 + 3, 512, 512])

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.use_residual = args.use_residual

    def forward(self, x, output_everything=False):
        xyz = x[..., :3, :]         # B, 3, N
        batch_size = x.size(0)

        x = self.conv0(x)                # (batch_size, 32, num_points)

        x1, knn1 = get_graph_feature(x, k=self.k)  # (batch_size, 32, num_points) -> (batch_size, 32*2, num_points, k)
        x1 = self.conv1(x1)  # (batch_size, 32*2, num_points, k) -> (batch_size, 32, num_points, k)
        f1 = x1.max(dim=-1, keepdim=False)[0]  # (batch_size, 32, num_points, k) -> (batch_size, 32, num_points)
        if self.use_residual:
            f1 = f1 + x
        xyz1, f1 = self.down1(xyz, f1)          # (batch_size, 32, num_points) -> (batch_size, 64, num_points / 4)

        x2, knn2 = get_graph_feature(f1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x2 = self.conv2(x2)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        f2 = x2.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        if self.use_residual:
            f2 = f1 + f2
        xyz2, f2 = self.down2(xyz1, f2)          # (batch_size, 64, num_points) -> (batch_size, 128, num_points / 4)

        x3, knn3 = get_graph_feature(f2, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x3 = self.conv3(x3)  # (batch_size, 128*2, num_points, k) -> (batch_size, 128, num_points, k)
        f3 = x3.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        if self.use_residual:
            f3 = f2 + f3
        xyz3, f3 = self.down3(xyz2, f3)  # (batch_size, 128, num_points/4) -> (batch_size, 256, num_points/16)

        x4, knn4 = get_graph_feature(f3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x4 = self.conv4(x4)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        f4 = x4.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)
        if self.use_residual:
            f4 = f3 + f4
        xyz4, f4 = self.down4(xyz3, f4)  # (batch_size, 256, num_points/16) -> (batch_size, 512, num_points/64)

        # x5 = torch.cat((f1, f2, f3, f4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        f5 = self.conv5(f4)  # (batch_size, 512, num_points) -> (batch_size, emb_dims, num_points)
        f6 = F.adaptive_max_pool1d(f5, 1).view(batch_size,
                                               -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        f7 = F.adaptive_avg_pool1d(f5, 1).view(batch_size,
                                               -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x8 = torch.cat((f6, f7), 1)  # (batch_size, emb_dims*2)

        f8 = F.leaky_relu(self.bn6(self.linear1(x8)),
                          negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x9 = self.dp1(f8)
        f9 = F.leaky_relu(self.bn7(self.linear2(x9)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x10 = self.dp2(f9)
        x = self.linear3(x10)  # (batch_size, 256) -> (batch_size, output_channels)

        if output_everything:
            return x, knn1, knn2, knn3, knn4, (xyz1, f1), (xyz2, f2), (xyz3, f3), (xyz4, f4), f5
        else:
            return x


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x
