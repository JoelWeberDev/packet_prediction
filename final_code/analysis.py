"""
@Author: Joel Weber
@Date: 2025-08-19
@Description: Comparison of the results from the various simulations

@Notes:

@TODO:
"""

### Python imports ###
import numpy as np
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

### Local imports ###
from final_code.helper_functions import pkl_read_model, ModelMetrics
from final_code.online_micro_model import *


def compare_loss_fxns():
    results_dir = "results/online_learning"
    result_titles = [
        "cross_ent_loss_2",
        "focus_loss_2",
        "label_smoothing_loss_2",
        "no_train",
        "no_meta_train",
    ]

    for key in ["train", "validation"]:
        # create accuracy and loss figures
        acc_fig, ax1 = plt.subplots(1, 1, figsize=(11, 11))
        loss_fig, ax2 = plt.subplots(1, 1, figsize=(11, 11))

        # Set titles and labels
        ax1.set_title(f"{key} Accuracy Comparison")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")

        ax2.set_title(f"{key} Loss Comparison")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")

        # Define colors and line styles for different models
        # colors = ["blue", "red", "green", "purple"]
        # linestyles = ["-", "--", "-.", ":"]

        # Load and plot all the results on the same axes
        for idx, result_title in enumerate(result_titles):
            path = os.path.join(results_dir, result_title)

            model, metrics, metadata = pkl_read_model(path)

            # Extract epochs for x-axis
            epochs = range(1, len(metrics[key].epoch_avg_accs) + 1)
            packet_lens = metrics[key]

            # Plot accuracy
            ax1.plot(
                epochs,
                metrics[key].epoch_avg_accs,
                label=result_title.replace("_results", ""),
                # color=colors[idx],
                # linestyle=linestyles[idx],
                marker="o",
                markersize=4,
            )

            # Plot loss
            ax2.plot(
                epochs,
                metrics[key].epoch_avg_losses,
                label=result_title.replace("_results", ""),
                # color=colors[idx],
                # linestyle=linestyles[idx],
                marker="o",
                markersize=4,
            )

        # Add legends
        ax1.legend()
        ax2.legend()

        # Add grid for better readability
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save figure
        acc_fig.savefig(
            os.path.join(results_dir, f"{key}_loss_function_comparison_acc.png"),
            dpi=300,
        )

        loss_fig.savefig(
            os.path.join(results_dir, f"{key}_loss_function_comparison_loss.png"),
            dpi=300,
        )

        # Show plot
        plt.show()


if __name__ == "__main__":
    compare_loss_fxns()
