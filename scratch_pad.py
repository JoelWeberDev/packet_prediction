import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import higher
from typing import Iterable, List
from dataclasses import dataclass
from modules.CONSTANTS import *
from copy import deepcopy
from modules.helper_functions import (
    conversation_tradjectory_loss,
    progressive_loss,
    conversation_trajectory_loss_simple,
)


class AbsGRU(nn.GRU):
    def __init__(self):
        super().__init__(100, 50, 2)

    def __str__(self):
        return "Test GRU"


criteria = nn.CrossEntropyLoss()

model = AbsGRU()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

t_in = torch.tensor([list(range(100))], dtype=torch.float32)
exp = torch.tensor([list(range(50))], dtype=torch.float32)

model.train()

output, _ = model.forward(t_in)


loss_list = list()
with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
    prior = output
    for i in range(2):
        t_in = torch.tensor([list(range(100))], dtype=torch.float32)

        exp = torch.tensor([np.random.randint(5, 7)], dtype=torch.long)

        output, _ = fmodel.forward(t_in)

        diff = abs(prior - output)

        loss = criteria.forward(output, exp)
        diffopt.step(loss)

        loss_list.append(loss)

        print(f"Output: {output.argmax(-1)}")
        print(torch.linalg.norm(output[0]))
        print(diff.sum(), end="\n\n")
        prior = output

opt.zero_grad()
overall_loss = conversation_trajectory_loss_simple(torch.stack(loss_list))
overall_loss.backward()
opt.step()


print(f"Overall loss: {overall_loss}")
