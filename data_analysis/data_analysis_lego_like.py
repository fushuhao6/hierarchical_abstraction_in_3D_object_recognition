import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from utils import plot_confusion_matrix, correlate_confusion_matrices


custom_colors = {
    'Human': '#AFB0D0',
    'DGCNN': '#F7D8B7',
    'Point Transformer': '#E7C1D7',
}

def calculate_confidence_interval(data):
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)  # 95% confidence interval
    return mean, mean - ci, mean + ci


if __name__ == '__main__':
    SHOW_PLOT = True
    result_file = '../data/point_cloud_lego_like_classification_long_format.csv'
    df = pd.read_csv(result_file)

    # Read original data from exp1
    origin_df = pd.read_csv('../data/point_cloud_classification_long_format.csv')
    origin_df = origin_df[origin_df['prop'] == 100]
    origin_df['voxel'] = [0.0] * len(origin_df)
    df = pd.concat([df, origin_df], ignore_index=True)

    # Concatenate the new row
    grouped_df = df.groupby(['voxel', 'id'])['correct'].mean().reset_index()
    new_column_data = ["Human"] * len(grouped_df)
    grouped_df['Type'] = new_column_data

    grouped_df = grouped_df.rename(columns={
        'voxel': 'Voxel size',
        'correct': 'Accuracy'
    })

    # Group by Voxel size and calculate mean and confidence intervals
    results = df.groupby('voxel')['correct'].apply(calculate_confidence_interval).reset_index()
    results[['Mean Accuracy', 'Lower CI', 'Upper CI']] = pd.DataFrame(results['correct'].tolist(), index=results.index)
    results = results.drop(columns=['correct'])

    # Print the results
    print(results)

    #################### Show model performance ###############################
    model_root = '../data/point_cloud_model_results_data_aug'

    def read_model_data(model_data_path, col_name, model_name):
        df = pd.read_csv(model_data_path)

        df = df[df['prop'] < 0.4]
        supp_df = pd.read_csv(model_data_path.replace('_lego_like', ''))
        supp_df = supp_df[supp_df['prop'] == 100]
        supp_df['prop'] = [0.0] * len(supp_df)
        df = pd.concat([df, supp_df], axis=0)

        df.rename(columns={'prop': col_name, 'acc': 'Accuracy'}, inplace=True)

        # Concatenate the new row
        new_column_data = [model_name] * len(df)
        df['Type'] = new_column_data
        return df

    # DGCNN
    dgcnn_pred_file = os.path.join(model_root, 'dgcnn_stimuli_lego_like_pred.csv')  # '../data/point_cloud_prop_stimuli_model_pred.csv'
    dgcnn_df = read_model_data(dgcnn_pred_file, 'Voxel size', 'DGCNN')

    pt_pred_file = os.path.join(model_root, 'PointTransformer_stimuli_lego_like_pred.csv')  # '../data/point_cloud_prop_stimuli_model_pred.csv'
    pt_df = read_model_data(pt_pred_file, 'Voxel size', 'Point Transformer')

    df = pd.concat([grouped_df, dgcnn_df, pt_df], ignore_index=True)

    # Create subplots for each Type
    g = sns.FacetGrid(df, col='Type', col_wrap=3, height=5, aspect=1.2)

    # Map the barplot to each subplot with 95% confidence intervals
    for type_name, ax in zip(custom_colors.keys(), g.axes.flat):
        sns.barplot(
            data=df[df['Type'] == type_name],
            x='Voxel size',
            y='Accuracy',
            color=custom_colors[type_name],
            ci=95,  # Add 95% confidence intervals
            ax=ax,
        )
        ax.set_title(type_name, fontsize=20)  # Set title for each subplot
        ax.grid(axis='y')  # Add gridlines
        ax.set_ylim(0, 1)  # Set consistent y-axis limits
        ax.tick_params(axis='both', labelsize=16)  # Set tick label size

    # Add global labels
    g.set_axis_labels('Voxel Size', 'Accuracy', fontsize=20)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot
    plt.savefig('exp3_subplots.png', transparent=True)
    plt.show()

    print('========================= Error pattern analysis ===========================')
    dgcnn_pred_file = os.path.join(model_root, 'dgcnn_stimuli_lego_like_pred.csv')
    dgcnn_pred_df = pd.read_csv(dgcnn_pred_file)
    dgcnn_pred_df = dgcnn_pred_df[dgcnn_pred_df['prop'] < 0.4]

    point_trans_pred_file = os.path.join(model_root, 'PointTransformer_stimuli_lego_like_pred.csv')
    point_trans_pred_df = pd.read_csv(point_trans_pred_file)
    point_trans_pred_df = point_trans_pred_df[point_trans_pred_df['prop'] < 0.4]

    human_df = pd.read_csv(result_file)

    dgcnn_cm = plot_confusion_matrix(dgcnn_pred_df, label_col='label', pred_col='pred', show_labels=False, save_path='exp_lego_dgcnn_cm.png')
    point_trans_cm = plot_confusion_matrix(point_trans_pred_df, label_col='label', pred_col='pred', show_labels=False, save_path='exp_lego_pt_cm.png')
    human_cm = plot_confusion_matrix(human_df, label_col='cls', pred_col='choice', show_labels=False, save_path='exp_lego_human_cm.png')

    dgcnn_corr, dgcnn_p = correlate_confusion_matrices(dgcnn_cm, human_cm, normalize=False, exclude_diagonal=True)
    print(f"DGCNN & human confusion matrix correlation: {dgcnn_corr:.04f}, p = {dgcnn_p:.04f}")

    point_trans_corr, point_trans_p = correlate_confusion_matrices(point_trans_cm, human_cm, normalize=False, exclude_diagonal=True)
    print(f"Point Transformer & human confusion matrix correlation: {point_trans_corr:.04f}, p = {point_trans_p:.04f}")
