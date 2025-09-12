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
from typing import List

### Local imports ###
from final_code.helper_functions import pkl_read_model, ModelMetrics
from final_code.online_micro_model import *


def compare_results(
    result_titles: List[str], cmp_huristics_title: str = "Loss Functions"
):
    results_dir = "results/online_learning"

    for key in ["train", "validation"]:
        # create accuracy and loss figures
        acc_fig, ax1 = plt.subplots(1, 1, figsize=(11, 11))
        loss_fig, ax2 = plt.subplots(1, 1, figsize=(11, 11))

        # Set titles and labels
        ax1.set_title(f"{key} Accuracy Comparison for {cmp_huristics_title}")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Conversation Length")

        ax2.set_title(f"{key} Loss Comparison for {cmp_huristics_title}")
        ax2.set_xlabel("Conversation Length")
        ax2.set_ylabel("Loss")

        # Define colors and line styles for different models
        colors = ["blue", "red", "green", "purple"]
        line_styles = ["-", "--", "-.", ":"]
        marker_styles = ["o", "s", "^", "x", "d", "+", "v"]

        # Load and plot all the results on the same axes
        for idx, result_title in enumerate(result_titles):
            path = os.path.join(results_dir, result_title)

            model, metrics, metadata = pkl_read_model(path)

            # Extract epochs for x-axis
            epochs = list(range(1, len(metrics[key].epoch_avg_accs) + 1))
            packet_lens = metrics[key]

            # Generate array of number of packets trained on
            conv_lens = [
                int(metadata["O_MAX_CONV_LEN"] * (epoch) / epochs[-1])
                for epoch in epochs
            ]

            # Plot accuracy
            ax1.plot(
                conv_lens,
                metrics[key].epoch_avg_accs,
                label=result_title.replace("_results", ""),
                # color=colors[idx % len(colors)],
                linestyle=line_styles[idx % len(line_styles)],
                marker=marker_styles[idx % len(marker_styles)],
                markersize=4,
            )

            # Plot loss
            ax2.plot(
                conv_lens,
                metrics[key].epoch_avg_losses,
                label=result_title.replace("_results", ""),
                # color=colors[idx % len(colors)],
                linestyle=line_styles[idx % len(line_styles)],
                marker=marker_styles[idx % len(marker_styles)],
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
            os.path.join(
                results_dir,
                f"{key}_{cmp_huristics_title.replace(" ", "_")}_comparison_acc.png",
            ),
            dpi=300,
        )

        loss_fig.savefig(
            os.path.join(
                results_dir,
                f"{key}_{cmp_huristics_title.replace(" ", "_")}_comparison_loss.png",
            ),
            dpi=300,
        )

        # Show plot
        plt.show()


if __name__ == "__main__":

    loss_fxn_result_titles = [
        "cross_ent_loss",
        "focus_loss",
        "label_smoothing_loss",
        "no_train",
        "no_meta_train",
    ]

    compare_results(loss_fxn_result_titles)

    model_result_titles = [
        "gru_model",
        "no_train",
        "rnn_results",
        "lstm_results",
    ]

    compare_results(model_result_titles, cmp_huristics_title="RNN Variants")
