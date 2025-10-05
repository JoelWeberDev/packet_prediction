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
from final_code.helper_functions import (
    pkl_read_model,
    ModelMetrics,
    estimate_model_size,
)
from final_code.online_micro_model import *
from final_code.CONSTANTS import RESULTS_DIR


def compare_results(
    result_titles: List[str], cmp_huristics_title: str = "Loss Functions"
):

    for key in ["train", "validation"]:
        # create accuracy and loss figures
        acc_fig, ax1 = plt.subplots(1, 1, figsize=(11, 11))
        loss_fig, ax2 = plt.subplots(1, 1, figsize=(11, 11))
        runtime_fig, ax3 = plt.subplots(1, 1, figsize=(11, 11))

        # Set titles and labels
        ax1.set_title(f"{key} Accuracy Comparison for {cmp_huristics_title}")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Conversation Length (packets)")

        ax2.set_title(f"{key} Loss Comparison for {cmp_huristics_title}")
        ax2.set_xlabel("Conversation Length (packets)")
        ax2.set_ylabel("Loss")

        ax3.set_title(f"{key} Time Comparison for {cmp_huristics_title}")
        ax3.set_xlabel("Conversation Length (packets)")
        ax3.set_ylabel("Runtime (s)")

        # Define colors and line styles for different models
        colors = ["blue", "red", "green", "purple"]
        line_styles = ["-", "--", "-.", ":"]
        marker_styles = ["o", "s", "^", "x", "d", "+", "v"]

        # Load and plot all the results on the same axes
        for idx, result_title in enumerate(result_titles):
            path = os.path.join(RESULTS_DIR, result_title)

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
                markersize=6,
                linewidth=2.5
            )

            # Plot loss
            ax2.plot(
                conv_lens,
                metrics[key].epoch_avg_losses,
                label=result_title.replace("_results", ""),
                # color=colors[idx % len(colors)],
                linestyle=line_styles[idx % len(line_styles)],
                marker=marker_styles[idx % len(marker_styles)],
                markersize=6,
                linewidth=2.5
            )

            # Get the runtimes
            epoch_runtimes = list()
            for epoch_result in metrics[key].epoch_results:
                conv_runtimes = list(
                    conv_result.tot_time for conv_result in epoch_result.conv_results
                )
                epoch_runtimes.append(np.mean(conv_runtimes))

            # Plot runtimes
            ax3.plot(
                conv_lens,
                epoch_runtimes,
                label=result_title.replace("_results", ""),
                # color=colors[idx % len(colors)],
                linestyle=line_styles[idx % len(line_styles)],
                marker=marker_styles[idx % len(marker_styles)],
                markersize=6,
                linewidth=2.5
            )

        # Add legends
        ax1.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax2.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax3.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)

        # Add grid for better readability
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax3.grid(True, alpha=0.3)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save figure
        acc_fig.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{key}_{cmp_huristics_title.replace(" ", "_")}_comparison_acc.png",
            ),
            dpi=300,
        )

        loss_fig.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{key}_{cmp_huristics_title.replace(" ", "_")}_comparison_loss.png",
            ),
            dpi=300,
        )

        runtime_fig.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{key}_{cmp_huristics_title.replace(" ", "_")}_comparison_runtimes.png",
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
        # "no_meta_train",
    ]

    compare_results(loss_fxn_result_titles)

    model_result_titles = [
        "gru_model",
        "no_train",
        "rnn_results",
        "lstm_results",
    ]

    compare_results(model_result_titles, cmp_huristics_title="RNN Variants")

    model_result_titles = [
        "simple_conv_loss_train",
        "no_conv_train",
        "exp_conv_loss_train",
    ]

    # compare_results(model_result_titles, cmp_huristics_title="Conv Loss Functions")

    # model_dir = "results/online_learning/online_model_12"

    # model, metrics, metadata = pkl_read_model(model_dir)

    # print(model[0])

    # print(f"approx model size: {estimate_model_size(model[0]) / (1024)}")

    # print(f"approx params size: {get_num_model_params(model[0]) }")
