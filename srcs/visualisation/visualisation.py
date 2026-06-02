import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import DataFrame
from typing import List


TARGET_COLUMN = "diagnosis"
ID_COLUMN = "sample_id"


def classes_distribution(df: DataFrame, show: bool = False, target_column: str = TARGET_COLUMN) -> None:
    """Visualise the distribution of classes in the target column."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df)
    plt.title("Distribution of Classes")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    if show:
        plt.show()
    plt.savefig("visu/distribution/classes_distribution.png", bbox_inches="tight")

def heatmap_correlation(df: DataFrame, show: bool = False, target_column: str = TARGET_COLUMN) -> None:
    """Visualise the correlation matrix of the dataset."""
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[ID_COLUMN], errors="ignore")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    if show:
        plt.show()
    plt.savefig("visu/heatmap/correlation_matrix.png", bbox_inches="tight")

def _rank_features_by_class_separation(df: DataFrame, target_column: str = TARGET_COLUMN) -> List[str]:
    """Rank numeric features by how well they separate the two classes."""
    feature_columns = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in {ID_COLUMN}
    ]
    class_values = df[target_column].dropna().unique().tolist()
    if len(class_values) != 2:
        return feature_columns

    first_class = class_values[0]
    second_class = class_values[1]
    ranked_features = []

    for feature in feature_columns:
        first_values = df.loc[df[target_column] == first_class, feature].dropna()
        second_values = df.loc[df[target_column] == second_class, feature].dropna()
        pooled_std = df[feature].std()
        if pooled_std == 0 or np.isnan(pooled_std):
            score = 0.0
        else:
            score = abs(first_values.mean() - second_values.mean()) / pooled_std
        ranked_features.append((feature, score))

    ranked_features.sort(key=lambda item: item[1], reverse=True)
    return [feature for feature, _ in ranked_features]


def boxplot_features(df: DataFrame, features: List[str], show: bool = False, target_column: str = TARGET_COLUMN) -> None:
    """Visualise the distribution of features with respect to the target column."""
    features = [feature for feature in features if feature in df.columns]
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=target_column, y=feature, data=df)
        plt.title(f"Boxplot of {feature} by {target_column}")
        plt.xlabel(target_column)
        plt.ylabel(feature)
        if show:
            plt.show()
        plt.savefig(f"visu/boxplot/{feature}_by_{target_column}.png", bbox_inches="tight")

def pairplot_reduced_features(df: DataFrame, features: List[str], show: bool = False, target_column: str = TARGET_COLUMN, name: str = "selected_features") -> None:
    """Visualise pairwise relationships between a subset of features and the target column."""
    sns.pairplot(df[features + [target_column]], hue=target_column)
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    if show:
        plt.show()
    plt.savefig(f"visu/pairplot/{name}.png", bbox_inches="tight")

def visualise_dataset(df: DataFrame) -> None:
    """Visualise the dataset using seaborn and matplotlib."""
    target_column = TARGET_COLUMN if TARGET_COLUMN in df.columns else df.columns[-1]
    ranked_features = _rank_features_by_class_separation(df, target_column)
    top_features = ranked_features[:8]
    same_correlation_features = ["feature_1", "feature_3", "feature_4", "feature_21", "feature_23", "feature_24"]
    
    classes_distribution(df, show=False ,target_column=target_column)
    heatmap_correlation(df, show=False, target_column=target_column)
    boxplot_features(df, top_features,show=False, target_column=target_column)
    print(f"Top features by class separation: {top_features}")
    pairplot_reduced_features(df, top_features, show=False, target_column=target_column, name="top_features")
    pairplot_reduced_features(df, same_correlation_features, show=False, target_column=target_column)
    