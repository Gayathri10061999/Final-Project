"""
Visualization Utilities

Author : Gayathri
Project : Multi Dataset Image Classification
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

os.makedirs("outputs", exist_ok=True)


class PlotUtils:

    @staticmethod
    def plot_loss(train_loss, val_loss,
                  save_path="outputs/loss_curve.png"):

        plt.figure(figsize=(8,5))

        plt.plot(train_loss,
                 label="Train Loss",
                 linewidth=2)

        plt.plot(val_loss,
                 label="Validation Loss",
                 linewidth=2)

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Training vs Validation Loss")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()


    @staticmethod
    def plot_accuracy(train_acc,
                      val_acc,
                      save_path="outputs/accuracy_curve.png"):

        plt.figure(figsize=(8,5))

        plt.plot(train_acc,
                 label="Train Accuracy",
                 linewidth=2)

        plt.plot(val_acc,
                 label="Validation Accuracy",
                 linewidth=2)

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.title("Training vs Validation Accuracy")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()


    @staticmethod
    def plot_learning_rate(lr_history,
                           save_path="outputs/lr_curve.png"):

        plt.figure(figsize=(8,5))

        plt.plot(lr_history,
                 linewidth=2)

        plt.xlabel("Epoch")

        plt.ylabel("Learning Rate")

        plt.title("Learning Rate Schedule")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()


    @staticmethod
    def plot_confusion_matrix(y_true,
                              y_pred,
                              classes,
                              save_path="outputs/confusion_matrix.png"):

        cm = confusion_matrix(y_true,
                              y_pred)

        fig, ax = plt.subplots(figsize=(10,8))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=classes
        )

        disp.plot(
            cmap="Blues",
            ax=ax,
            xticks_rotation=45
        )

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()


    @staticmethod
    def plot_roc_curve(y_true,
                       y_score,
                       save_path="outputs/roc_curve.png"):

        fpr, tpr, _ = roc_curve(y_true,
                               y_score)

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8,6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.4f}",
            linewidth=2
        )

        plt.plot(
            [0,1],
            [0,1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()


    @staticmethod
    def plot_precision_recall_curve(
            y_true,
            y_score,
            save_path="outputs/pr_curve.png"):

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_score
        )

        plt.figure(figsize=(8,6))

        plt.plot(
            recall,
            precision,
            linewidth=2
        )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision Recall Curve")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()


    @staticmethod
    def plot_class_distribution(
            labels,
            class_names,
            save_path="outputs/class_distribution.png"):

        counts = np.bincount(labels)

        plt.figure(figsize=(10,5))

        plt.bar(class_names, counts)

        plt.xticks(rotation=45)

        plt.xlabel("Classes")

        plt.ylabel("Number of Images")

        plt.title("Dataset Class Distribution")

        plt.tight_layout()

        plt.savefig(save_path)

        plt.close()
