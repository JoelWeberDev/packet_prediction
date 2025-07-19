"""
@Author: Joel Weber
@Date: 2025-07-18
@Description: Implementation of various custom ml model architechures

@Notes:

@TODO:
"""

### Python imports ###
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import random

### Local imports ###
from modules.CONSTANTS import *


class MemoryCentricGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        # Split hidden state into fast and slow components
        self.fast_gru = nn.GRU(input_size, hidden_size // 2, num_layers=num_layers)
        self.slow_gru = nn.GRU(input_size, hidden_size // 2, num_layers=num_layers)

        # Gates for controlling memory flow
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Sigmoid()
        )

        # Read/write controllers
        self.write_gate = nn.Linear(hidden_size, hidden_size // 2)
        self.read_gate = nn.Linear(hidden_size, hidden_size // 2)

        self.fast_hidden = None
        self.slow_hidden = None
        self.hidden_size = hidden_size

    def forward(self, x):
        # Fast-changing memory (packet-level)
        fast_out, fast_h = self.fast_gru(x, self.fast_hidden)

        # Slow-changing memory (conversation-level) - updated less frequently
        if self.training:
            update_prob = 0.3  # Control update frequency
            if random.random() < update_prob:
                slow_out, slow_h = self.slow_gru(x, self.slow_hidden)
                self.slow_hidden = slow_h.detach()  # Detach to prevent full backprop
            else:
                slow_out = torch.zeros_like(fast_out)
                slow_h = self.slow_hidden
        else:
            slow_out, slow_h = self.slow_gru(x, self.slow_hidden)
            self.slow_hidden = slow_h

        # Controlled memory mixing
        gate = self.memory_gate(torch.cat([fast_out, slow_out], dim=-1))
        output = gate * fast_out + (1 - gate) * slow_out

        self.fast_hidden = fast_h
        return output, (fast_h, slow_h)
