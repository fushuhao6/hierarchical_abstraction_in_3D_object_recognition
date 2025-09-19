import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from utils import plot_confusion_matrix, correlate_confusion_matrices

custom_colors = {
    'Human': '#AFB0D0',
    'DGCNN': '#F7D8B7',
    'Point Transformer': '#E7C1D7',
}


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

    result_file_long_format = '../data/point_cloud_classification_upside_down_long_format.csv'
    human_df = pd.read_csv(result_file_long_format)

    grouped_df = human_df.groupby(['prop', 'id'])['correct'].mean().reset_index()

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
    dgcnn_pred_file = os.path.join(model_root, 'dgcnn_stimuli_upsidedown_pred.csv')  # '../data/point_cloud_prop_stimuli_model_pred.csv'
    dgcnn_df = read_model_data(dgcnn_pred_file, 'Proportion', 'DGCNN')

    pt_pred_file = os.path.join(model_root, 'PointTransformer_stimuli_upsidedown_pred.csv')  # '../data/point_cloud_prop_stimuli_model_pred.csv'
    pt_df = read_model_data(pt_pred_file, 'Proportion', 'Point Transformer')


    df = pd.concat([grouped_df, dgcnn_df, pt_df], ignore_index=True)

    df['Proportion'] = df['Proportion'].astype(int)

    # Set the size of the figure and create subplots
    g = sns.FacetGrid(df, col='Type', col_wrap=3, height=5, aspect=1.2)

    # Map the barplot to each subplot and assign colors
    for type_name, ax in zip(custom_colors.keys(), g.axes.flat):
        sorted_data = df[df['Type'] == type_name].sort_values(by='Proportion', ascending=False)

        sns.barplot(
            data=sorted_data,
            x='Proportion',
            y='Accuracy',
            color=custom_colors[type_name],
            ax=ax,
            errorbar=('ci', 95),
            order=sorted_data['Proportion']
        )
        ax.set_title(type_name, fontsize=20)  # Set title for each subplot
        ax.grid(axis='y')  # Add gridlines
        ax.set_ylim(0, 1)  # Set consistent y-axis limits
        ax.tick_params(axis='both', labelsize=16)  # Set tick label size

    # Add global labels
    g.set_axis_labels('Proportion (%)', 'Accuracy', fontsize=20)

    # Adjust layout to avoid overlaps
    plt.tight_layout()

    # Save the plot
    plt.savefig('exp2_subplots.png', transparent=True)
    plt.show()

    print('========================= Error pattern analysis ===========================')
    dgcnn_pred_file = os.path.join(model_root, 'dgcnn_stimuli_upsidedown_pred.csv')
    dgcnn_pred_df = pd.read_csv(dgcnn_pred_file)

    point_trans_pred_file = os.path.join(model_root, 'PointTransformer_stimuli_upsidedown_pred.csv')
    point_trans_pred_df = pd.read_csv(point_trans_pred_file)

    dgcnn_cm = plot_confusion_matrix(dgcnn_pred_df, label_col='label', pred_col='pred')
    point_trans_cm = plot_confusion_matrix(point_trans_pred_df, label_col='label', pred_col='pred')
    human_cm = plot_confusion_matrix(human_df, label_col='cls', pred_col='choice')

    dgcnn_corr, dgcnn_p = correlate_confusion_matrices(dgcnn_cm, human_cm, normalize=False, exclude_diagonal=True)
    print(f"DGCNN & human confusion matrix correlation: {dgcnn_corr:.04f}, p = {dgcnn_p:.04f}")

    point_trans_corr, point_trans_p = correlate_confusion_matrices(point_trans_cm, human_cm, normalize=False, exclude_diagonal=True)
    print(f"Point Transformer & human confusion matrix correlation: {point_trans_corr:.04f}, p = {point_trans_p:.04f}")