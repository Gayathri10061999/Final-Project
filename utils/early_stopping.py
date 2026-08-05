"""
Early Stopping Utility

Stops training if the validation loss does not improve
for a specified number of epochs.
"""

import torch


class EarlyStopping:
    def __init__(
        self,
        patience=5,
        verbose=True,
        delta=0.0,
        path="checkpoints/best_model.pth"
    ):
        """
        Args:
            patience (int): Number of epochs to wait.
            verbose (bool): Print messages.
            delta (float): Minimum improvement.
            path (str): Path to save best model.
        """

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float("inf")

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:

            self.counter += 1

            if self.verbose:
                print(
                    f"EarlyStopping Counter: "
                    f"{self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        if self.verbose:
            print(
                f"Validation Loss Improved "
                f"({self.best_loss:.6f} --> {val_loss:.6f})"
            )

        torch.save(model.state_dict(), self.path)

        self.best_loss = val_loss
