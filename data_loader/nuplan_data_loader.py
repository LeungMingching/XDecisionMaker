from base import BaseDataLoader
from torchvision.transforms import Compose
from transforms.nuplan_transforms import *
from dataset import NuPlanDataset

class NuPlanDataLoader(BaseDataLoader):

    def __init__(self, data_root, scenario_list, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = Compose([
            ZeroTinyValue(zero_threshold=1.0e-2),
            ToBEV(),
            FilterObjectsByRadius(50, 11, is_shuffle_objects=True),
            NormalizeByAxis(axis=0),
            ToTensor()
        ])
        target_trsfm = Compose([
            ToTensor()
        ])
        self.dataset = NuPlanDataset(data_root, scenario_list, transform=trsfm, target_transform=target_trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)