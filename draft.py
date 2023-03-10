import os
import numpy as np
from torchvision.transforms import Compose

from transforms.nuplan_transforms import *
if __name__ == '__main__':
    
    data_root = './data/NuPlan/converted_dataset'
    scenario = 'low_magnitude_speed'
    
    with open(os.path.join(data_root, scenario, 'observation_array.npy'), 'rb') as f:
        observation_array = np.load(f)
    with open(os.path.join(data_root, scenario, 'look_ahead_pt_array.npy'), 'rb') as f:
        look_ahead_pt_array = np.load(f)
    print(f'observation shape: {observation_array.shape}')
    print(f'look_ahead_pt shape: {look_ahead_pt_array.shape}')

    trsfm = Compose([
        ZeroTinyValue(zero_threshold=1.0e-2),
        ToBEV(),
        FilterObjectsByRadius(50, 50, is_shuffle_objects=True),
        NormalizeByAxis(axis=0),
        ToTensor()
    ])

    zero_value = ZeroTinyValue()
    to_bev = ToBEV()
    filter_objects_by_radius = FilterObjectsByRadius(50, 50)
    to_tensor = ToTensor()
    normailzation = NormalizeByAxis(axis=0)

    result0_1 = to_bev(observation_array[0])
    result0 = zero_value(result0_1)
    result1 = filter_objects_by_radius(result0)
    result2 = normailzation(result1)
    result = to_tensor(result2)
    print(f'result0_1: \n {result0_1}')
    print(f'result0: \n {result0}')
    print(f'result1: \n {result1}')
    print(f'result2: \n {result2}')
    print(f'result: \n {result}')

    result = trsfm(observation_array[0])

    print(f'result: \n {result}')
    print(f'result shape: {result.shape}')

