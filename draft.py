import os
import numpy as np
from transforms.nuplan_transforms import ToBEV, FilterObjectsByRadius

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
    filter_objects_by_radius = FilterObjectsByRadius(10, 10)
    result1 = to_bev(observation_array[0])
    result = filter_objects_by_radius(result1)
    print(result)
    print(f'result shape: {result.shape}')

