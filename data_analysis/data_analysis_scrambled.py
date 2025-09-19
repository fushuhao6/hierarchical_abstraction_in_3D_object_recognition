import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_analysis_upright import calculate_confidence_interval
from scipy.stats import ttest_rel
from utils import plot_confusion_matrix, correlate_confusion_matrices


CHOICE_MAPPING = {
    "cls1": "airplane",
    "cls2": "car",
    "cls3": "chair",
    "cls4": "lamp",
    "cls5": "table",
}

if __name__ == '__main__':
    result_file_long_format = '../data/point_cloud_scrambled_classification_v2_long_format.csv'

    df = pd.read_csv(result_file_long_format)

    def label_condition(row):
        if row['scrambled'] == 'normal':
            return 'normal'
        elif row['model'] == 'dgcnn':
            return 'dgcnn_scrambled'
        else:
            return 'point_transformer_scrambled'
    df['condition'] = df.apply(label_condition, axis=1)
    # Set desired order for plotting
    condition_order = ['normal', 'dgcnn_scrambled', 'point_transformer_scrambled']
    df['condition'] = pd.Categorical(df['condition'], categories=condition_order, ordered=True)

    # Group by Voxel size and calculate mean and confidence intervals
    results = df.groupby(['category', 'condition'])['correct'].apply(calculate_confidence_interval).reset_index()
    results[['Mean Accuracy', 'Lower CI', 'Upper CI']] = pd.DataFrame(results['correct'].tolist(), index=results.index)
    results = results.drop(columns=['correct'])

    # Compute overall accuracy per condition (across all categories)
    overall_per_condition = df.groupby('condition')['correct'].apply(calculate_confidence_interval).reset_index()
    overall_per_condition[['Mean Accuracy', 'Lower CI', 'Upper CI']] = pd.DataFrame(
        overall_per_condition['correct'].tolist(), index=overall_per_condition.index
    )
    overall_per_condition = overall_per_condition.drop(columns=['correct'])
    overall_per_condition['category'] = 'Overall'

    # Reorder columns to match `results`
    overall_per_condition = overall_per_condition[['category', 'condition', 'Mean Accuracy', 'Lower CI', 'Upper CI']]

    # Combine with original results
    results_with_overall = pd.concat([results, overall_per_condition], ignore_index=True)

    # Print combined results
    print(results_with_overall)

    ################### Human result bar plot - 3 conditions ####################
    # Group by 'id', 'category', and 'scrambled' and calculate the average of 'response'
    grouped_df = df.groupby(['id', 'category', 'condition'])['correct'].mean().reset_index()
    grouped_df['condition'] = pd.Categorical(grouped_df['condition'], categories=condition_order, ordered=True)

    # Add Overall condition by averaging across categories for each id + condition
    overall_df = (
        grouped_df
        .groupby(['id', 'condition'], as_index=False)['correct']
        .mean()
    )
    overall_df['category'] = 'overall'  # Mark category as "Overall"

    # Append to grouped_df for plotting
    plot_df = pd.concat([grouped_df, overall_df], ignore_index=True)
    plot_df['condition'] = pd.Categorical(plot_df['condition'], categories=condition_order, ordered=True)

    # Colors for bar plot (same)
    condition_colors = {
        'normal': '#AFB0D0',
        'dgcnn_scrambled': '#7E92C8',
        'point_transformer_scrambled': '#314272'
    }

    # Plot
    sns.set(style="whitegrid")
    g = sns.catplot(
        data=plot_df,
        kind='bar',
        x='category',
        y='correct',
        hue='condition',
        palette=condition_colors,
        errorbar=('ci', 95),
        height=5,
        aspect=2.5
    )

    # Format axes and legend
    g.set_axis_labels("Category", "Accuracy", fontsize=22)
    g.set_xticklabels(rotation=0, fontsize=20)
    g.set_yticklabels(fontsize=20)
    g.set_titles("")
    g.set(ylim=(0, 1))

    g._legend.remove()  # Remove default legend

    # Custom legend
    ax = g.ax
    handles, labels = ax.get_legend_handles_labels()
    custom_labels = {
        'normal': 'Intact',
        'dgcnn_scrambled': 'DGCNN scrambled',
        'point_transformer_scrambled': 'Point Transformer scrambled'
    }
    labels = [custom_labels[label] for label in labels]

    legend = plt.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        fontsize=18,
        title="",
        title_fontsize=16
    )

    plt.tight_layout()
    plt.savefig('scrambled_v2_3_conditions.png')
    plt.show()

    # T-test
    print('\n\n-------------------- t-test ------------------------')

    # Filter to only include scrambled conditions
    scrambled_df = grouped_df[grouped_df['condition'].isin(['dgcnn_scrambled', 'point_transformer_scrambled'])]

    # Pivot the data so each participant's accuracy for the two conditions are side-by-side
    pivot_df = scrambled_df.pivot_table(index=['id', 'category'], columns='condition', values='correct').reset_index()
    pivot_df = pivot_df.dropna(subset=['dgcnn_scrambled', 'point_transformer_scrambled'])

    # Paired t-test (for each category)
    categories = pivot_df['category'].unique()
    for cat in categories:
        sub_df = pivot_df[pivot_df['category'] == cat]
        t_stat, p_val = ttest_rel(sub_df['dgcnn_scrambled'], sub_df['point_transformer_scrambled'])
        print(f"[{cat}] Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")

    pivot_df_agg = pivot_df.groupby('id')[['dgcnn_scrambled', 'point_transformer_scrambled']].mean().dropna()

    t_stat, p_val = ttest_rel(pivot_df_agg['dgcnn_scrambled'], pivot_df_agg['point_transformer_scrambled'])
    print(f"[Overall] Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")


    ################### Model performance #######################
    dgcnn_pred_df = pd.read_csv('../data/point_cloud_model_results_data_aug/dgcnn_stimuli_scrambled_PT_shapenetpart_pred.csv')
    PT_pred_df = pd.read_csv('../data/point_cloud_model_results_data_aug/PointTransformer_stimuli_scrambled_dgcnn_shapenetpart_pred.csv')

    dgcnn_pred_df.rename(columns={'acc': 'correct', 'prop': 'scrambled', 'label': 'category', 'pred': 'choice'}, inplace=True)
    PT_pred_df.rename(columns={'acc': 'correct', 'prop': 'scrambled', 'label': 'category', 'pred': 'choice'}, inplace=True)

    human_pred_on_dgcnn_df = df[df['model'] == 'dgcnn']     # to be compared with PT_pred_df
    human_pred_on_PT_df = df[df['model'] == 'point_transformer']        # to be compared with dgcnn_pred_df

    human_pred_on_dgcnn_df = human_pred_on_dgcnn_df.groupby(['id', 'category', 'scrambled'])['correct'].mean().reset_index()
    human_pred_on_dgcnn_df['Type'] = 'Human'
    PT_pred_df['Type'] = 'Point Transformer'
    pred_on_dgcnn_df = pd.concat([human_pred_on_dgcnn_df, PT_pred_df])

    human_pred_on_PT_df = human_pred_on_PT_df.groupby(['id', 'category', 'scrambled'])['correct'].mean().reset_index()
    human_pred_on_PT_df['Type'] = 'Human'
    dgcnn_pred_df['Type'] = 'DGCNN'
    pred_on_PT_df = pd.concat([human_pred_on_PT_df, dgcnn_pred_df])

    # 3 panel figure
    model_plot_df = pd.concat([PT_pred_df, dgcnn_pred_df])

    # replace scrambled to dgcnn_scrambled or point_transformer_scrambled
    model_plot_df.loc[(model_plot_df["Type"] == "DGCNN") & (model_plot_df["scrambled"] != "normal"), "scrambled"] = "point_transformer_scrambled"
    model_plot_df.loc[(model_plot_df["Type"] == "Point Transformer") & (model_plot_df["scrambled"] != "normal"), "scrambled"] = "dgcnn_scrambled"

    # Get unique categories
    cats = df["category"].unique()

    # Build the extra rows for their own scrambled. It is correct by definition (how we got the scrambled)
    extra_rows = []
    for c in cats:
        extra_rows.append({
            "category": c,
            "choice": c,
            "correct": True,
            "scrambled": "dgcnn_scrambled",
            "Type": "DGCNN"
        })
        extra_rows.append({
            "category": c,
            "choice": c,
            "correct": True,
            "scrambled": "point_transformer_scrambled",
            "Type": "Point Transformer"
        })

    extra_df = pd.DataFrame(extra_rows)

    # Combine with original df
    model_plot_df = pd.concat([model_plot_df, extra_df], ignore_index=True)

    custom_colors = {
        'DGCNN': {
            'normal': '#F7D8B7',  # original
            'dgcnn_scrambled': '#D7A97D',  # original scrambled
            'point_transformer_scrambled': '#B07C5F'  # darker/warmer variant
        },
        'Point Transformer': {
            'normal': '#E7C1D7',  # original
            'dgcnn_scrambled': '#B689A9',  # original scrambled
            'point_transformer_scrambled': '#926D8A'  # darker/warmer variant
        }
    }

    hue_order = ["normal", "dgcnn_scrambled", "point_transformer_scrambled"]

    sns.set(style="whitegrid")
    category_order = ["airplane", "car", "chair", "lamp", "table", "overall"]
    for type_name in custom_colors.keys():
        data_df = model_plot_df[model_plot_df['Type'] == type_name]
        # Duplicate the data and label as "Overall"
        overall_df = data_df.copy()
        overall_df['category'] = 'overall'

        # Combine original and "Overall" rows
        combined_df = pd.concat([data_df, overall_df], ignore_index=True)

        g = sns.catplot(
            data=combined_df,
            kind='bar',
            x='category',
            y='correct',
            hue='scrambled',
            palette=custom_colors[type_name],
            errorbar=('ci', 95),
            height=5,
            aspect=2.5,
            order=category_order,
            hue_order=hue_order
        )

        # Format axes and legend
        g.set_axis_labels("Category", "Accuracy", fontsize=22)
        g.set_xticklabels(rotation=0, fontsize=20)
        g.set_yticklabels(fontsize=20)
        g.set_titles("")
        g.set(ylim=(0, 1))

        g._legend.remove()  # Remove default legend

        # Custom legend
        ax = g.ax
        handles, labels = ax.get_legend_handles_labels()
        custom_labels = {
            'normal': 'Intact',
            'dgcnn_scrambled': 'DGCNN scrambled',
            'point_transformer_scrambled': 'Point Transformer scrambled'
        }
        labels = [custom_labels[label] for label in labels]

        legend = plt.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            fontsize=18,
            title="",
            title_fontsize=16
        )

        plt.tight_layout()
        plt.savefig(f'scrambled_v2_{type_name}.png')
        plt.show()
