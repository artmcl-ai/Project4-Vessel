from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import RandFlipd, RandRotate90d, EnsureTyped, Compose


class MultiClassDiffDataset(Dataset):
    def __init__(self, config, classes):
        self._config = config
        
        self._data = {}
        self._data_p = []
        self._cls_ids = []
        for path, cls_d in classes.items():
            cls_id, cls_prop = cls_d
            self._data[cls_id] = list(Path(path).resolve().iterdir())
            self._data_p.append(cls_prop)
            self._cls_ids.append(cls_id)

        # init online transforms
        self._transform = get_transforms(config)

    def __len__(self):
        return self._config.DATA.LEN_EPOCH

    def __getitem__(self, idx):
        while True:
            class_id = self._cls_ids[np.random.choice(len(self._data_p), 1, p=self._data_p)[0]]
            data = self._data[class_id]

            idx = (np.random.randint(low=0, high=len(data)) + idx) % len(data)

            # load labels and images
            img = torch.tensor(np.load(Path(data[idx]) / 'img.npy')).float()[None]
            mask = torch.tensor(np.load(Path(data[idx]) / 'mask.npy')).float()[None]

            if self._config.TRANSFORMS.USE_TRANSFORMS:  # additional, lightweight online data augmentation
                transformed_data = self._transform({'x': img, 'y': mask})
                img, mask = transformed_data['x'], transformed_data['y']

            if list(img.shape) == [1, 128, 128, 128]:
                break
            else:
                print(f'{img.shape} in {class_id}')

        #diffusion model requires [-1, 1]
        img = img * 2 - 1
        return img, mask, class_id

 
def get_transforms(config):
    transforms = [
        RandFlipd(
            keys=['x', 'y'], spatial_axis=0, prob=config.TRANSFORMS.P_FLIP[0]
        ),
        RandFlipd(
            keys=['x', 'y'], spatial_axis=1, prob=config.TRANSFORMS.P_FLIP[1]
        ),
        RandFlipd(
            keys=['x', 'y'], spatial_axis=2, prob=config.TRANSFORMS.P_FLIP[2]
        ),
        RandRotate90d(
            keys=['x', 'y'], spatial_axes=(0, 1), prob=config.TRANSFORMS.P_ROTATE[0]
        ),
        RandRotate90d(
            keys=['x', 'y'], spatial_axes=(0, 2), prob=config.TRANSFORMS.P_ROTATE[1]
        ),
        RandRotate90d(
            keys=['x', 'y'], spatial_axes=(1, 2), prob=config.TRANSFORMS.P_ROTATE[2]
        ),
        EnsureTyped(
            keys=['x', 'y'], track_meta=False,
        )
    ]
    return Compose(transforms)

def build_loader(config, classes):
    dataset = MultiClassDiffDataset(config, classes)
    dataloader = DataLoader(
        dataset, batch_size=config.DATA.BATCH_SIZE, 
        shuffle=True, num_workers=config.DATA.NUM_WORKERS, 
        drop_last=True, pin_memory=True
    )
    return dataloader