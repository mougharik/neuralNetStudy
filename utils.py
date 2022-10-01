import os
import torch
import numpy as np
from typing import cast
from torch.utils.data import Dataset
from prettytable import PrettyTable

class USPSDataset(Dataset):
    def __init__(self, x, y):
        x_type = torch.FloatTensor
        y_type = torch.LongTensor   

        self.length = x.shape[0]

        self.x_data = x.type(x_type)
        self.y_data = y.type(y_type)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.length


def buildUSPSDataset(filename: str, model_idx: int) -> USPSDataset:
    with open(os.path.join(os.getcwd(), filename)) as f:
        raw_data = [lines.split() for lines in f.readlines()]
        tmp = [[x for x in data[1:]] for data in raw_data]
        img_real = np.asarray(tmp, dtype=np.float32).reshape((-1,16,16))
        img_int = ((cast(np.ndarray, img_real) + 1) / 2 * 255).astype(dtype=np.uint8)
        targets = [int(d[0][0]) for d in raw_data]
        if model_idx == 1:
            img_real = torch.from_numpy(img_real).reshape(img_real.shape[0], 256)
        elif model_idx == 2:
            img_real = torch.from_numpy(img_real).unsqueeze(-1)
        else:
            img_real = torch.from_numpy(img_real).unsqueeze(1)

    return USPSDataset(img_real, torch.tensor(targets)), img_int 

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params