import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def get_model_best_score(true: pd.DataFrame, oof: pd.DataFrame, min_th: float = 0.6, max_th: float = 0.65, visible: bool = True) -> float:
    """Search thresholds and return best micro-f1score results.

    Args:
        true (pd.DataFrame): Dataframe with true values.
        oof (pd.DataFrame): Dataframe with oof prediction probablities.
        min_th (float, optional): Lower limit of threshold search range. Defaults to 0.6.
        max_th (float, optional): Upper limit of threshold search range. Defaults to 0.65.
        visible (bool, optional): Whether to plot the results of the threshold search. Defaults to True.

    Returns:
        float: Micro-F1 score.
    """
    scores = []
    thresholds = []
    best_score = 0
    best_threshold = 0

    # Find the best threshold.
    for threshold in np.arange(min_th, max_th, 0.001):
        preds = (oof.values.reshape(-1) > threshold).astype("int")
        m = f1_score(true.values.reshape(-1), preds, average="macro")
        scores.append(m)
        thresholds.append(threshold)
        if m > best_score:
            best_score = m
            best_threshold = threshold

    # Plot threshold vs micro-F1 score.
    if visible:
        plt.figure(figsize=(20, 5))
        plt.plot(thresholds, scores, "-o", color="blue")
        plt.scatter([best_threshold], [best_score], color="blue", s=300, alpha=1)
        plt.xlabel("Threshold", size=14)
        plt.ylabel("Validation F1 Score", size=14)
        plt.title(f"Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3}", size=18)
        plt.show()

        print(f"When using optimal threshold = {best_threshold:.2f}...")

    # Display micro-F1 score for each question
    for q in range(18):
        f1_q = f1_score(true[q].values, (oof[q].values > best_threshold).astype("int"), average="macro")
        print(f"Q{q}: F1 =", f1_q)
    f1_micro = f1_score(true.values.reshape(-1), (oof.values > best_threshold).reshape(-1).astype("int"), average="macro")

    print("==> Overall F1 =", f1_micro)
    return f1_micro

