import os
import numpy as np
from typing import Callable, List, Optional
from torch.utils.data import Dataset

class NuPlanDataset(Dataset):
    def __init__(self,
        data_root: str,
        scenario_list: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:

        self.transform = transform
        self.target_transform = target_transform
        
        for idx in range(len(scenario_list)):
            observation_path = os.path.join(data_root, scenario_list[idx], 'observation_array.npy')
            look_ahead_pt_path = os.path.join(data_root, scenario_list[idx], 'look_ahead_pt_array.npy')

            with open(observation_path, 'rb') as obs_f:
                observation_array = np.load(obs_f)
            with open(look_ahead_pt_path, 'rb') as la_f:
                look_ahead_pt_array = np.load(la_f)

            if idx == 0:
                self.observation = observation_array
                self.look_ahead_pt = look_ahead_pt_array
            else:
                self.observation = np.concatenate((self.observation, observation_array), axis=0)
                self.look_ahead_pt = np.concatenate((self.look_ahead_pt, look_ahead_pt_array), axis=0)

        assert len(self.observation) == len(self.look_ahead_pt)

    def __len__(self):
        return len(self.observation)

    def __getitem__(self, idx):
        obs = self.observation[idx]
        lap = self.look_ahead_pt[idx]
        
        if self.transform:
            obs = self.transform(self.observation[idx])
        if self.target_transform:
            lap = self.target_transform(self.look_ahead_pt[idx])
        
        return obs, lap
    
if __name__ == '__main__':

    scenarios = [
        'accelerating_at_crosswalk',
        'near_pedestrian_at_pickup_dropoff',
        'traversing_crosswalk'
    ]
    
    dirname = os.path.dirname(__file__)
    data_root = os.path.join(dirname, '../data/NuPlan/converted_dataset')

    dataset = NuPlanDataset(data_root, scenarios)