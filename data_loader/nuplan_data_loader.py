from base import BaseDataLoader
from torchvision.transforms import Compose, ToTensor
from transforms.nuplan_transforms import ToBEV, FilterObjectsByRadius, NormalizeByAxis
from dataset import NuPlanDataset

class NuPlanDataLoader(BaseDataLoader):

    def __init__(self, data_root, scenario_list, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = Compose([
            ToBEV(),
            FilterObjectsByRadius(50, 11, is_shuffle_objects=True),
            NormalizeByAxis(), # TODO:
            ToTensor()
        ])
        self.dataset = NuPlanDataset(data_root, scenario_list, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)