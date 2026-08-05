"""
Classification Metrics Utility
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


class ClassificationMetrics:

    def __init__(self, class_names=None):
        self.class_names = class_names

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred, average="macro"):
        return precision_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        )

    def recall(self, y_true, y_pred, average="macro"):
        return recall_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        )

    def f1(self, y_true, y_pred, average="macro"):
        return f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        )

    def roc_auc(self, y_true, y_prob):

        try:
            return roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr"
            )
        except Exception:
            return None

    def top_k_accuracy(self, y_prob, y_true, k=5):

        top_k = np.argsort(y_prob, axis=1)[:, -k:]

        correct = 0

        for idx, label in enumerate(y_true):

            if label in top_k[idx]:
                correct += 1

        return correct / len(y_true)

    def confusion(self, y_true, y_pred):

        return confusion_matrix(
            y_true,
            y_pred
        )

    def classification(self, y_true, y_pred):

        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )

    def plot_confusion_matrix(
            self,
            y_true,
            y_pred,
            save_path="outputs/confusion_matrix.png"
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names
        )

        disp.plot(
            cmap="Blues",
            ax=ax,
            xticks_rotation=45
        )

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()

    def print_metrics(self, y_true, y_pred, y_prob=None):

        print("=" * 60)

        print(f"Accuracy  : {self.accuracy(y_true,y_pred):.4f}")

        print(f"Precision : {self.precision(y_true,y_pred):.4f}")

        print(f"Recall    : {self.recall(y_true,y_pred):.4f}")

        print(f"F1 Score  : {self.f1(y_true,y_pred):.4f}")

        if y_prob is not None:

            auc = self.roc_auc(y_true, y_prob)

            if auc is not None:
                print(f"ROC-AUC   : {auc:.4f}")

        print("=" * 60)
