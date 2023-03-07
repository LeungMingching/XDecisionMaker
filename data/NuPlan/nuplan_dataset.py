import os
import torch
from torch.utils.data import Dataset

class NuPlanDataset(Dataset):
    def __init__(self,
        data_root: str,
        scenario_list: list[str]
    ) -> None:
        
        self.data_root = data_root
        self.scenario_list = scenario_list

        self.observation
        self.look_ahead_pt

        assert len(self.observation) == len(self.look_ahead_pt)

    def __len__(self):
        return len(self.observation)

    def __getitem__(self, idx):
        