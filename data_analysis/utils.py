import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(df, label_col='label', pred_col='pred', normalize=False, show_labels=True,
                          figsize=(8, 6), cmap='Blues', title='', save_path=None):
    """
    Plot and return a confusion matrix from a DataFrame with true and predicted labels.

    Parameters:
        df (pd.DataFrame): DataFrame with label and prediction columns.
        label_col (str): Column name for true labels.
        pred_col (str): Column name for predicted labels.
        normalize (bool): If True, normalize by row (recall).
        figsize (tuple): Size of the plot.
        cmap (str): Colormap for heatmap.

    Returns:
        pd.DataFrame or None: The confusion matrix DataFrame if return_matrix=True, else None.
    """
    labels = sorted(df[label_col].unique())
    cm = confusion_matrix(df[label_col], df[pred_col], labels=labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=figsize)
    sns.heatmap(cm_df, annot=True, fmt=fmt, cmap=cmap, cbar=False)
    if show_labels:
        plt.xlabel('Prediction', fontsize=22)
        plt.ylabel('True Label', fontsize=22)
    else:
        plt.xlabel('')
        plt.ylabel('')
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

    return cm


def correlate_confusion_matrices(cm1, cm2, normalize=True, exclude_diagonal=True):
    """
    Compute the Pearson correlation and p-value between two confusion matrices.

    Parameters:
        cm1 (np.ndarray or pd.DataFrame): Confusion matrix 1.
        cm2 (np.ndarray or pd.DataFrame): Confusion matrix 2.
        normalize (bool): Whether to row-normalize each matrix before computing correlation.
        exclude_diagonal (bool): Whether to exclude diagonal (correct predictions).

    Returns:
        corr (float): Pearson correlation coefficient.
        p_value (float): Two-tailed p-value.
    """
    cm1 = np.array(cm1)
    cm2 = np.array(cm2)

    if normalize:
        cm1 = cm1 / cm1.sum(axis=1, keepdims=True)
        cm2 = cm2 / cm2.sum(axis=1, keepdims=True)

    flat1 = cm1.flatten()
    flat2 = cm2.flatten()

    if exclude_diagonal:
        mask = ~np.eye(cm1.shape[0], dtype=bool).flatten()
        flat1 = flat1[mask]
        flat2 = flat2[mask]

    corr, p_value = pearsonr(flat1, flat2)
    return corr, p_value