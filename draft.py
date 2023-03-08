import os
import numpy as np
from transforms.nuplan_transforms import ToBEV

if __name__ == '__main__':
    
    data_root = './data/NuPlan/converted_dataset'
    scenario = 'low_magnitude_speed'
    
    with open(os.path.join(data_root, scenario, 'observation_array.npy'), 'rb') as f:
        observation_array = np.load(f)
    with open(os.path.join(data_root, scenario, 'look_ahead_pt_array.npy'), 'rb') as f:
        look_ahead_pt_array = np.load(f)
    print(f'observation shape: {observation_array.shape}')
    print(f'look_ahead_pt shape: {look_ahead_pt_array.shape}')

    to_bev = ToBEV()
    result = to_bev(observation_array[0])
    print(result)

