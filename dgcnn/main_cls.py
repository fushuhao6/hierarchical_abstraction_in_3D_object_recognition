#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import provider
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet40
from model import PointNet, DGCNN_cls, DGCNN_Downsample
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import pandas as pd


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py outputs'+'/'+args.exp_name+'/'+'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

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

    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-08,)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=50, gamma=0.3)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            if args.use_data_aug:
                points = data.numpy()
                points = provider.random_point_dropout(points)
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                data = torch.Tensor(points)

            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:

            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)


def test(args, io, class_names, selected_classes=None):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points, proportion=args.proportion, upsidedown=args.upsidedown),
                             batch_size=1, shuffle=False, drop_last=False)

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

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        target = class_names[label.item()]
        if selected_indices is not None:
            if target not in selected_classes:
                continue
        data = data.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        assert batch_size == 1
        logits = model(data)

        if selected_classes is not None:
            selected_logits = logits[:, selected_indices]
            preds = selected_logits.max(dim=1)[1].item()
            predicted_classes = selected_indices[preds]
            pred = class_names[predicted_classes]
        else:
            preds = logits.max(dim=1)[1].item()
            pred = class_names[preds]

        test_true.append(class_names.index(target))
        test_pred.append(class_names.index(pred))
    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test downsample proportion: {}, upside_down: {} :: test acc: {:.04f}'.format(args.proportion, args.upsidedown, test_acc)
    io.cprint(outstr)
    return test_true, test_pred


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn', 'dgcnn_downsample'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--use_residual', action='store_true', default=False,
                        help='Use residual in model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--use_data_aug', action='store_true', default=False, help='use data_augmentation')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        with open('data/modelnet40_ply_hdf5_2048/shape_names.txt', 'r') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        selected_classes = ["airplane", "bottle", "bowl", "chair", "cup", "lamp", "person", "piano", "stool", "table"]

        prop_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        upside_down = [False, True]
        save_data = {'Prop': [], 'UpsideDown': [], 'Overall': []}
        for c in class_names:
            save_data[c] = []
        for u in upside_down:
            for proportion in prop_list:

                args.proportion = proportion
                args.upsidedown = u
                label, pred = test(args, io, class_names, selected_classes)

                save_data['Prop'].append(proportion)
                save_data['UpsideDown'].append(u)
                save_data['Overall'].append(metrics.accuracy_score(label, pred))

                for i, c in enumerate(class_names):
                    test_acc = metrics.accuracy_score(label[label == i], pred[label == i])
                    save_data[c].append(test_acc)
        save_data = pd.DataFrame(save_data)
        save_data.to_csv(f'data/{args.model}_ModelNet40_pred.csv')
