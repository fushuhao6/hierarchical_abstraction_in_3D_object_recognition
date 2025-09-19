import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


def read_ablation_results(folder_path, tag):
    filename = 'point_cloud_{}_pointTransformer_pred.csv'

    df = pd.DataFrame()
    for version in ['stimuli', 'stimuli_upsidedown', 'stimuli_lego_like', 'stimuli_scrambled']:
        data = pd.read_csv(os.path.join(folder_path, filename.format(version)))
        data['Experiment'] = [tag] * len(data)
        data['Version'] = [version] * len(data)
        df = pd.concat([df, data], ignore_index=True)
    return df

if __name__ == '__main__':
    root_path = '../data/point_cloud_model_results_data_aug'
    files = [
        ['Hengshuang_no_normal_correct_lr', 'Point Transformer (Original)'],
        ['Hengshuang_no_normal_no_attention_correct_lr', 'Remove Attention'],
        ['Hengshuang_no_normal_no_pos_enc_correct_lr', 'Remove Position Encoding'],
        ['Hengshuang_no_normal_no_downsample_correct_lr', 'Remove Downsampling'],
     ]

    custom_palette = {
        "Point Transformer (Original)": "#F49E52",
        "Remove Attention": "#6A9F58",
        "Remove Position Encoding": "#C78283",
        "Remove Downsampling": "#4080A0",
    }

    label_mapper = {'stimuli': 'Proportion',
                    'stimuli_upsidedown': 'Proportion',
                    'stimuli_lego_like': 'Voxel size',
                    'stimuli_scrambled': 'Type'}


    df = pd.DataFrame()
    for filename, tag in files:
        filename = os.path.join(root_path, filename)
        data = read_ablation_results(filename, tag)
        df = pd.concat([df, data], ignore_index=True)

    # Calculate mean accuracy for each label and order them in descending order
    label_order = df.groupby('Proportion')['Accuracy'].mean().sort_values(ascending=False).index

    experiment_order = [e[1] for e in files]
    df['Experiment'] = pd.Categorical(df['Experiment'], categories=experiment_order, ordered=True)

    # Convert 'Label' column to a categorical type with the specified order
    df['Proportion'] = pd.Categorical(df['Proportion'], categories=label_order, ordered=True)

    for version in ['stimuli', 'stimuli_upsidedown', 'stimuli_lego_like']:
        version_df = df[df['Version'] == version]
        version_df = version_df.dropna(axis=1, how='all')

        if version == 'stimuli_lego_like':
            version_df = version_df[version_df[label_mapper[version]] < 0.3]
            figure_width = 8
        elif version in ['stimuli', 'stimuli_upsidedown']:
            version_df['Proportion'] = version_df['Proportion'].astype(int)
            figure_width = 12

            # Convert 'Proportion' to categorical based on the sorted order
            version_df['Proportion'] = pd.Categorical(
                version_df['Proportion'],
                categories=sorted(version_df['Proportion'].unique(), reverse=True),  # Ensure order from large to small
                ordered=True
            )
        else:
            figure_width = 6

        print(version_df)

        plt.figure(figsize=(figure_width, 6))
        sns.barplot(x=label_mapper[version], y='Accuracy', hue='Experiment', data=version_df, palette=custom_palette, ci=95)

        # Set labels with increased font size
        label = label_mapper[version] if version in ['stimuli_lego_like'] else 'Proportion (%)'
        plt.xlabel(label, fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)

        # Adjust tick font size
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.ylim(0, 1)
        plt.grid(axis='y')

        # Move the legend outside the plot
        plt.legend(title='', loc='lower left', fontsize=16, title_fontsize=18)
        plt.tight_layout()

        # Save and clear the plot
        plt.savefig(f"{version}.png", transparent=True)
        # plt.show()
        plt.clf()
