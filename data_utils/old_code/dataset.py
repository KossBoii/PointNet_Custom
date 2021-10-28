import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from transform.normalize import Transform
from transform.point_sampler import PointSampler
from transform.to_tensor import ToTensor

def default_transforms():
    return transforms.Compose([
        PointSampler(4096),
        Normalize(),
        ToTensor(),
    ])

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, valid=False, folder='train', transform=default_transforms()):
        super().__init__()
        self.root_dir = root_dir

        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []

    
    def __len__(self):
        return len(self.files)
    