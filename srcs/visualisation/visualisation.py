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
    plt.close()

def heatmap_correlation(df: DataFrame, show: bool = False, target_column: str = TARGET_COLUMN) -> None:
    """Visualise the correlation matrix of the dataset."""
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[ID_COLUMN], errors="ignore")
    correlation = numeric_df.corr()
    if correlation.empty:
        return
    print(f"Correlation matrix:\n{correlation}")
    heatmap_size = max(10, min(18, 0.45 * len(correlation.columns)))
    annotate = len(correlation.columns) <= 12
    mask = np.triu(np.ones_like(correlation, dtype=bool))

    plt.figure(figsize=(heatmap_size, heatmap_size))
    sns.heatmap(
        correlation,
        mask=mask,
        annot=annotate,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig("visu/heatmap/correlation_matrix.png", bbox_inches="tight")
    plt.close()


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
        plt.close()

def pairplot_reduced_features(df: DataFrame, features: List[str], show: bool = False, target_column: str = TARGET_COLUMN, name: str = "selected_features") -> None:
    """Visualise pairwise relationships between a subset of features and the target column."""
    sns.pairplot(df[features + [target_column]], hue=target_column)
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    if show:
        plt.show()
    plt.savefig(f"visu/pairplot/{name}.png", bbox_inches="tight")
    plt.close("all")

def visualise_dataset(df: DataFrame, features: List[str] = None) -> None:
    """Visualise the dataset using seaborn and matplotlib."""
    if TARGET_COLUMN in df.columns:
        target_column = TARGET_COLUMN
    elif len(df.columns) > 1 and df.iloc[:, 1].dropna().astype(str).str.upper().isin({"M", "B"}).all():
        target_column = df.columns[1]
    else:
        target_column = df.columns[-1]

    if features is None:
        features = [column for column in df.columns if column not in (target_column, ID_COLUMN)]

    features = [feature for feature in features if feature in df.columns and feature != target_column]
    if not features:
        features = [column for column in df.select_dtypes(include=[np.number]).columns if column not in (target_column, ID_COLUMN)]

    same_correlation_features = ["feature_1", "feature_3", "feature_4", "feature_21", "feature_23", "feature_24"]
    
    classes_distribution(df, show=False ,target_column=target_column)
    heatmap_correlation(df, show=False, target_column=target_column)
    boxplot_features(df, features,show=False, target_column=target_column)
    # print(f"Top features by class separation: {features}")
    pairplot_reduced_features(df, features, show=False, target_column=target_column, name="selected_features")
    