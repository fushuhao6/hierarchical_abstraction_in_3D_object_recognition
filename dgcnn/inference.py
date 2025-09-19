#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
from model import DGCNN_Downsample, DGCNN_cls, PointNet
import numpy as np
import pandas as pd


# python inference.py --model dgcnn_downsample --use_residual --k 16 --model_path outputs/cls_downsample_1024_residual_data_augmentation/models/model.t7
# python inference.py --model_path outputs/cls_1024_data_augmentation/models/model.t7


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def test(args, data_path, class_names, selected_classes=None):
    point_clouds = os.listdir(data_path)

    selected_indices = None
    if selected_classes is not None:
        selected_indices = [class_names.index(c) for c in selected_classes]

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'dgcnn_downsample':
        model = DGCNN_Downsample(args).to(device)
    else:
        raise Exception("Not implemented")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    test_true = []
    test_pred = []
    test_prop = []
    correct = []
    features = []
    for data in point_clouds:
        labels = data.replace('.txt', '').split('_')
        if len(labels) == 4:
            label, obj_id, prop = labels[0], labels[1], labels[3]
        elif len(labels) == 3:
            label, obj_id, prop = labels[0], labels[1], labels[2]

        if selected_classes is not None and label not in selected_classes:
            continue

        with open(os.path.join(data_path, data), 'r') as f:
            lines = f.readlines()
        pointcloud = [l.strip().replace(',', '').split(' ') for l in lines]
        pointcloud = np.array(pointcloud, dtype=float)
        pointcloud = pc_normalize(pointcloud)
        pointcloud = torch.FloatTensor(pointcloud).to(device).unsqueeze(0)

        pointcloud = pointcloud.permute(0, 2, 1)
        logits, knn1, knn2, knn3, knn4, f1, f2, f3, f4, f5 = model(pointcloud, output_everything=True)
        if selected_classes is not None:
            selected_logits = logits[:, selected_indices]
            preds = selected_logits.max(dim=1)[1].item()
            predicted_classes = selected_indices[preds]
            pred = class_names[predicted_classes]
        else:
            preds = logits.max(dim=1)[1].item()
            pred = class_names[preds]

        test_true.append(label)
        test_pred.append(pred)
        correct.append(label == pred)
        test_prop.append(prop)
        if isinstance(f5, tuple):
            _, f5 = f5
        f5 = f5[0].permute(1, 0).max(0)[0].detach().cpu().numpy()
        features.append(f5)

    return test_true, test_pred, correct, test_prop, features


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn', 'dgcnn_downsample'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='pretrained/model.cls.1024.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--use_residual', action='store_true', default=False,
                        help='Use residual in model')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open('data/modelnet40_ply_hdf5_2048/shape_names.txt', 'r') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    selected_classes = ["airplane", "bottle", "bowl", "chair", "cup", "lamp", "person", "piano", "stool", "table"]

    for version in ['stimuli', 'stimuli_upsidedown', 'stimuli_lego_like', 'stimuli_scrambled_PT_shapenetpart']:
        data_path = f'../Point-Transformers/data/{version}_txt'

        if 'scrambled' in version:
            selected_classes = ["airplane", "car", "chair", "lamp", "table"]

        label, pred, correct, prop, features = test(args, data_path, class_names=class_names, selected_classes=selected_classes)

        df = pd.DataFrame({'label': label, 'pred': pred, 'acc': correct, 'prop': prop})
        grouped_df = df.groupby(['label', 'prop'])['acc'].mean().reset_index()

        if 'lego' in version:
            tag = 'Voxel size'
            # Filter out rows where voxel_size == 0.40
            grouped_df = grouped_df[grouped_df['prop'] != 0.40]
        elif 'scrambled' in version:
            tag = 'Type'
        else:
            tag = 'Proportion'

        # Renaming columns
        grouped_df = grouped_df.rename(columns={
            'label': 'Class',
            'prop': tag,
            'acc': 'Accuracy'
        })

        print(grouped_df)

        save_path = f'./results/{args.model}_{version}_pred.csv'
        print(f"Saving results to {save_path}")
        df.to_csv(save_path, index=False)

        # Group by Proportion and calculate the mean Accuracy
        avg_accuracy = df.groupby('prop')['acc'].mean().reset_index()
        print('---------------------------------------')
        print(avg_accuracy)
        print('===============================================')

