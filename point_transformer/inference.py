import numpy as np
import os
import torch
import importlib
import omegaconf
import hydra
import pandas as pd
from tqdm import tqdm
from pointnet_util import pc_normalize


def test(model, data_path, class_names, selected_classes=None):
    selected_indices = None
    if selected_classes is not None:
        selected_indices = [class_names.index(c) for c in selected_classes]

    point_clouds = os.listdir(data_path)

    test_true = []
    test_pred = []
    test_prop = []
    correct = []
    features = []
    for data in tqdm(point_clouds):
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
        pointcloud = torch.FloatTensor(pointcloud).cuda().unsqueeze(0)

        # pointcloud = pointcloud.permute(0, 2, 1)
        classifier = model.eval()
        logits, xyz_and_feats, attns = classifier(pointcloud, return_feat=True)
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

        feat = xyz_and_feats[-1][1][0]
        feat = feat.max(0)[0]
        features.append(feat.cpu().detach().numpy())

    return test_true, test_pred, correct, test_prop, features


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_model(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        print("Key not matched, trying to remove keys with 'sa'...")
        # Create a new state_dict with 'sa' removed from the keys
        new_state_dict = {key.replace('sa.', ''): value for key, value in state_dict.items()}

        # Now you can load the new state_dict back into your model
        model.load_state_dict(new_state_dict)
        print("Successfully loaded!")
    print(f"Number of model parameters: {count_parameters(model)}")
    return model

@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    CLASS_NAME_PATH = hydra.utils.to_absolute_path('./data/modelnet40_ply_hdf5_2048/shape_names.txt')
    with open(CLASS_NAME_PATH, 'r') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    selected_classes = ["airplane", "bottle", "bowl", "chair", "cup", "lamp", "person", "piano", "stool", "table"]

    '''MODEL LOADING'''
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()
    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')

    checkpoint = torch.load('best_model.pth')

    try:
        load_model(classifier, checkpoint['model_state_dict'])
    except:
        classifier = torch.nn.DataParallel(classifier)  # Wrap with DataParallel
        load_model(classifier, checkpoint['model_state_dict'])

    for version in ['stimuli', 'stimuli_upsidedown', 'stimuli_lego_like', 'stimuli_scrambled_dgcnn_shapenetpart']:
        data_path = hydra.utils.to_absolute_path(f'data/{version}_txt')

        if 'scrambled' in version:
            selected_classes = ["airplane", "car", "chair", "lamp", "table"]

        label, pred, correct, prop, features = test(classifier.eval(), data_path, class_names, selected_classes)

        df = pd.DataFrame({'label': label, 'pred': pred, 'acc': correct, 'prop': prop})

        save_path = f'PointTransformer_{version}_pred.csv'
        df.to_csv(save_path, index=False)

        grouped_df = df.groupby(['label', 'prop'])['acc'].mean().reset_index()
        print(grouped_df)

        # Group by Proportion and calculate the mean Accuracy
        avg_accuracy = df.groupby('prop')['acc'].mean().reset_index()
        print('---------------------------------------')
        print(avg_accuracy)
        print('===============================================')



if __name__ == '__main__':
    main()