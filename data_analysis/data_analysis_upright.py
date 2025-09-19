import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
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
    options = {
        "cls1": "Airplane",
        "cls2": "Bottle",
        "cls3": "Bowl",
        "cls4": "Chair",
        "cls5": "Cup",
        "cls6": "Lamp",
        "cls7": "Person",
        "cls8": "Piano",
        "cls9": "Stool",
        "cls10": "Table"
    }
    prop_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    result_file_long_format = '../data/point_cloud_classification_long_format.csv'
    df = pd.read_csv(result_file_long_format)

    grouped_df = df.groupby(['prop', 'id'])['correct'].mean().reset_index()

    new_column_data = ["Human"] * len(grouped_df)
    grouped_df['Type'] = new_column_data

    grouped_df = grouped_df.rename(columns={
        'prop': 'Proportion',
        'correct': 'Accuracy'
    })

    # Fit the model
    aovrm = AnovaRM(grouped_df, 'Accuracy', 'id', within=['Proportion'])
    res = aovrm.fit()

    # Print the results
    print(res)

    # Group by Voxel size and calculate mean and confidence intervals
    results = df.groupby('prop')['correct'].apply(calculate_confidence_interval).reset_index()
    results[['Mean Accuracy', 'Lower CI', 'Upper CI']] = pd.DataFrame(results['correct'].tolist(), index=results.index)
    results = results.drop(columns=['correct'])

    # Print the results
    print(results)


    #################### Show model performance ###############################
    model_root = '../data/point_cloud_model_results_data_aug'

    def read_model_data(model_data_path, col_name, model_name):
        df = pd.read_csv(model_data_path)
        df.rename(columns={'prop': col_name, 'acc': 'Accuracy'}, inplace=True)

        # Concatenate the new row
        new_column_data = [model_name] * len(df)
        df['Type'] = new_column_data
        return df

    # DGCNN
    dgcnn_pred_file = os.path.join(model_root, 'dgcnn_stimuli_pred.csv')  # '../data/point_cloud_prop_stimuli_model_pred.csv'
    dgcnn_df = read_model_data(dgcnn_pred_file, 'Proportion', 'DGCNN')

    pt_pred_file = os.path.join(model_root, 'PointTransformer_stimuli_pred.csv')  # '../data/point_cloud_prop_stimuli_model_pred.csv'
    pt_df = read_model_data(pt_pred_file, 'Proportion', 'Point Transformer')

    all_grouped_df = pd.concat([grouped_df, dgcnn_df, pt_df], ignore_index=True)

    all_grouped_df['Proportion'] = all_grouped_df['Proportion'].astype(int)

    # Set the size of the figure and create subplots
    g = sns.FacetGrid(all_grouped_df, col='Type', col_wrap=3, height=5, aspect=1.2)

    # Map the barplot to each subplot and assign colors
    for type_name, ax in zip(custom_colors.keys(), g.axes.flat):
        sorted_data = all_grouped_df[all_grouped_df['Type'] == type_name].sort_values(by='Proportion', ascending=False)

        sns.barplot(
            data=sorted_data,
            x='Proportion',
            y='Accuracy',
            color=custom_colors[type_name],
            ax=ax,
            ci=95,
            order=sorted_data['Proportion']
        )
        ax.set_title(type_name, fontsize=20)  # Set title for each subplot
        ax.grid(axis='y')  # Add gridlines
        ax.set_ylim(0, 1)  # Set consistent y-axis limits
        ax.tick_params(axis='both', labelsize=16)  # Set tick label size

    g.set_axis_labels('Proportion (%)', 'Accuracy', fontsize=20)
    plt.tight_layout()

    plt.savefig('exp1_subplots.png', transparent=True)
    plt.show()

    ####################### Ablation Study ########################
    dgcnn_ds_pred_file = os.path.join(model_root, 'dgcnn_downsample_stimuli_pred.csv')  # '../data/point_cloud_prop_stimuli_model_pred.csv'
    dgcnn_ds_df = read_model_data(dgcnn_ds_pred_file, 'Proportion', 'DGCNN DS')

    dgcnn_ablation_df = pd.concat([dgcnn_df, dgcnn_ds_df], ignore_index=True)
    dgcnn_ablation_df['Proportion'] = dgcnn_ablation_df['Proportion'].astype(int)
    sorted_data = dgcnn_ablation_df.sort_values(by='Proportion', ascending=False)
    sorted_data['Proportion'] = sorted_data['Proportion'].astype(str)

    plt.figure(figsize=(12, 6))
    custom_palette = {
        "DGCNN": "#F49E52",
        "DGCNN DS": "#4080A0",
    }
    sns.barplot(
        x='Proportion',
        y='Accuracy',
        hue='Type',
        hue_order=['DGCNN', 'DGCNN DS'],  # Explicit order: original first
        data=sorted_data,
        palette=custom_palette
    )
    # Set labels with increased font size
    plt.xlabel('Proportion (%)', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)

    # Adjust tick font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.grid(axis='y')

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ['DGCNN (Original)' if lbl == 'DGCNN' else 'DGCNN + Downsampling' for lbl in labels]
    plt.legend(handles=handles, labels=new_labels, loc='lower left', fontsize=18)
    plt.tight_layout()

    # Save and clear the plot
    plt.savefig(f"dgcnn_ablation.png", transparent=True)
    plt.show()

    ##### Plotting for VSS ###########
    plt.figure(figsize=(12, 6))
    custom_palette = {
        "DGCNN": "#F49E52",
        "DGCNN DS": "#4080A0",
    }

    sorted_data = sorted_data[sorted_data['Type'] == 'DGCNN']
    all_proportions = list(sorted_data['Proportion'].unique())
    dummy_rows = pd.DataFrame({
        'Proportion': all_proportions,
        'Type': ['DGCNN DS'] * len(all_proportions),
        'Accuracy': [np.nan] * len(all_proportions),
    })

    sorted_data = pd.concat([sorted_data, dummy_rows], ignore_index=True)
    sns.barplot(
        x='Proportion',
        y='Accuracy',
        hue='Type',
        hue_order=['DGCNN', 'DGCNN DS'],  # Explicit order: original first
        data=sorted_data,
        palette=custom_palette
    )
    # Set labels with increased font size
    plt.xlabel('Proportion (%)', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)

    # Adjust tick font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.grid(axis='y')

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ['DGCNN (Original)' if lbl == 'DGCNN' else 'DGCNN + Downsampling' for lbl in labels]
    plt.legend(handles=handles, labels=new_labels, loc='lower left', fontsize=18)
    plt.tight_layout()

    # Save and clear the plot
    plt.savefig(f"dgcnn_ablation_1bar.png", transparent=True)

    # Error pattern analysis
    print('========================= Error pattern analysis ===========================')
    dgcnn_pred_file = os.path.join(model_root, 'dgcnn_stimuli_pred.csv')
    dgcnn_pred_df = pd.read_csv(dgcnn_pred_file)

    point_trans_pred_file = os.path.join(model_root, 'PointTransformer_stimuli_pred.csv')
    point_trans_pred_df = pd.read_csv(point_trans_pred_file)

    dgcnn_cm = plot_confusion_matrix(dgcnn_pred_df, label_col='label', pred_col='pred')
    point_trans_cm = plot_confusion_matrix(point_trans_pred_df, label_col='label', pred_col='pred')
    human_cm = plot_confusion_matrix(df, label_col='cls', pred_col='choice')

    dgcnn_corr, dgcnn_p = correlate_confusion_matrices(dgcnn_cm, human_cm, normalize=False, exclude_diagonal=True)
    print(f"DGCNN & human confusion matrix correlation: {dgcnn_corr:.04f}, p = {dgcnn_p:.04f}")

    point_trans_corr, point_trans_p = correlate_confusion_matrices(point_trans_cm, human_cm, normalize=False, exclude_diagonal=True)
    print(f"Point Transformer & human confusion matrix correlation: {point_trans_corr:.04f}, p = {point_trans_p:.04f}")

