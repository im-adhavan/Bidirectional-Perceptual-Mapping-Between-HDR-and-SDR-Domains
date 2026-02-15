import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_correlation_matrix(df, save_path):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error_distribution(df, error_col, save_path):
    plt.figure(figsize=(6,5))
    sns.histplot(df[error_col], kde=True)
    plt.title(f"Distribution of {error_col}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error_vs_feature(df, feature_col, error_col, save_path):
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=df[feature_col], y=df[error_col])
    plt.title(f"{error_col} vs {feature_col}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()