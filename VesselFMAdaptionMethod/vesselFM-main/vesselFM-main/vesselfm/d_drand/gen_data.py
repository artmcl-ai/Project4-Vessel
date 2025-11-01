import json
import random
from pathlib import Path
from datetime import datetime

import yaml
from monai.utils import set_determinism
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    Transform,
    RandSpatialCrop, 
    RandFlip,
    RandZoom,
    Rand3DElastic,
    RandBiasField,
    RandGaussianNoise,
    RandKSpaceSpikeNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
    RandGibbsNoise,
    RandGaussianSharpen,
    RandRicianNoise,
    RandRotate90,
    RandHistogramShift,
    MedianSmooth,
    Compose,
    EnsureType
)
import numpy as np
import scipy
from scipy.ndimage import binary_erosion
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=3)

def load_config(dict=True):
    with open(f'./vesselfm/d_drand/config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    if dict:
        return config
    else:
        return Config(config)


class DRandGen(Dataset):
    """
    Class used to generate domain radomized dataset vrand_500k. Use soley for offline preprocessing.
    """
    def __init__(self, config):
        self._config = config

        # load paths to foreground data/labels
        data_dirs = list(Path(config.DATA_GEN.DATASET_LABEL).resolve().iterdir())

        self._seg = []
        self._ang = []
        self._rad = []
        for data_dir in data_dirs:
            files = data_dir.iterdir()
            for file in files:
                if 'mask' in str(file):
                    self._seg.append(file)
                elif 'ang' in str(file):
                    self._ang.append(file)
                elif 'rad' in str(file):
                    self._rad.append(file)

        # load paths to background data
        bg_dir = Path(config.DATA_GEN.DATASET_BACKGROUND).resolve().iterdir()

        self._bg = []
        for file in bg_dir:
            self._bg.append(file)

        # init transforms
        self._label_transform = get_label_transforms(config)    # to transform the labels
        self._fg_bg_transform = get_fg_bg_transforms(config)    # to fg and bg before merging
        self._int_transform = get_int_transforms(config)        # to transform the merged image

    def __len__(self):
        return 500000   # generate 500000 images

    def __getitem__(self, idx):
        seg_idx = (np.random.randint(low=0, high=len(self._seg)) + idx) % len(self._seg)
        bg_idx = (np.random.randint(low=0, high=len(self._bg)) + idx) % len(self._bg)

        # load and transform labels
        raw_labels = torch.tensor(np.load(self._seg[seg_idx])).float()[None]
        labels = self._label_transform(raw_labels)
        labels_int_mod = labels.clone()

        # load and add background
        if self._config.DATA_GEN.ADD_BACKGROUND:
            background = torch.tensor(np.load(self._bg[bg_idx])).float()[None]
            labels_int_mod = torch.clamp(add_background(labels_int_mod, background, self._fg_bg_transform, self._config), 0, 1)

        # transform intensities
        if self._config.DATA_GEN.TRANSFORM_INTENSITY:
            labels_int_mod = torch.clamp(self._int_transform(labels_int_mod), 0, 1)

        return labels, labels_int_mod
    

class RandDilation(Transform):
    def __init__(self, prob, num_max_steps):
        self.prob = prob
        self.num_max_steps = num_max_steps

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img = img.unsqueeze(0)
        img_dtype = img.dtype

        steps = random.randint(1, self.num_max_steps)
        for _ in range(steps):
            img = torch.clamp(F.conv3d(img.int(), torch.ones([1, 1, 3, 3, 3]).int(), padding=(1, 1, 1)), 0, 1)
        
        return img.squeeze(0).to(dtype=img_dtype)
    

class RandMedianSmooth(MedianSmooth):
    def __init__(self, radius_range, prob):
        super().__init__(radius_range[0])
        self.radius_range = radius_range
        self.prob = prob

    def __call__(self, img) :
        if np.random.rand() > self.prob:
            return img
        self.radius = np.random.randint(self.radius_range[0], self.radius_range[1])
        return super().__call__(img)


class RandInvert(Transform):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img

        img_dtype = img.dtype
        img = 1 - img
        return img.to(dtype=img_dtype)


class RandDropOut(Transform):
    def __init__(self, prob, prob_drop):
        self.prob = prob
        self.prob_drop = prob_drop

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        
        img_intermediate = img.clone()
        drop = torch.rand(img.shape) < self.prob_drop
        img_intermediate[drop] = 0
        img[img > 0] = img_intermediate[img > 0]
        return img


class RandRoll(Transform):
    def __init__(self, prob, roll_range):
        self.prob = prob
        self.roll_range = roll_range

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        
        roll_img = torch.empty_like(img)
        for i in range(img.size(-1)):
            shift = torch.randint(self.roll_range[0], self.roll_range[1] + 1, (1,)).item()
            roll_img[:, :, i] = torch.roll(img[:, :, i], shifts=shift)
        
        return roll_img

 
class RandHull(Transform):
    def __init__(self, prob, hull_thickness_range):
        self.prob = prob
        self.hull_thickness_range = hull_thickness_range

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        
        hull_thickness = torch.randint(
            self.hull_thickness_range[0], self.hull_thickness_range[1] + 1, (1,)
        ).item()
        
        img_mask = img > 0
        img_er = img_mask.clone().squeeze()
        for _ in range(hull_thickness):
            img_er = binary_erosion(img_er)

        hull = img_mask & ~torch.tensor(img_er)[None]
        img[hull == 0] = 0
        return img


class RandBinarySmooth(Transform):
    def __init__(self, prob, sigma, thresh=0.5):
        self.prob = prob
        self.sigma = sigma
        self.thresh = thresh

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img_dtype = img.dtype
        
        img_smooth = scipy.ndimage.gaussian_filter(img, sigma=self.sigma)
        return torch.tensor((img_smooth > self.thresh)).to(dtype=img_dtype)


class RandGaussianSmoothCommonSigma(Transform):
    def __init__(self, prob, sigma_range, approx="erf"):
        self.prob = prob
        self.sigma_range = sigma_range
        self.approx = approx

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img_dtype = img.dtype

        sigma = (self.sigma_range[1] - self.sigma_range[0]) * torch.rand(1).item() + self.sigma_range[0]
        sigma = torch.as_tensor(sigma, device=img.device)

        gaussian_filter = GaussianFilter(img.ndim - 1, sigma, approx=self.approx)
        img_smooth = gaussian_filter(img.unsqueeze(0)).squeeze(0)

        return img_smooth.to(dtype=img_dtype)


class RandGaussianSmoothSigmaMod():
    """
    Gaussian smoothing either with sigma shared across spatial dimensions or with individual sigma values per spatial dimension.
    """
    def __init__(self, prob, prob_common, sigma_range, sigma_x, sigma_y, sigma_z):
        self.prob = prob
        self.prob_common = prob_common
        self.smooth_common_sigma = RandGaussianSmoothCommonSigma(prob=1, sigma_range=sigma_range)
        self.smooth_spatial_sigma = RandGaussianSmooth(prob=1, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img_dtype = img.dtype

        if np.random.rand() > self.prob_common:
            img_smooth = self.smooth_spatial_sigma(img)
        else:
            img_smooth = self.smooth_common_sigma(img)

        return img_smooth.to(dtype=img_dtype)


def add_background(label, background, fg_bg_transforms, config):
    # transform background
    background = fg_bg_transforms['invert'](background)
    bg_mean_int = background.mean()
    assert background.max() <= 1 and background.min() >= 0

    # determine fg intensity
    delta = config.DATA_GEN.FOREGROUND.DELTA_BG
    while True:
        label_int = torch.rand(1)

        # label_int should not be extremely similar to bg_mean_int
        if not (bg_mean_int - delta <= label_int.item() <= bg_mean_int + delta):
            break
    
    # sample fg transformation and merging mode
    fg_transform = random.choices(
        ['none', 'bias_field', 'gauss_noise', 'gauss_smooth', 'dropout', 'roll', 'hull'],
        config.DATA_GEN.FOREGROUND.P_FG_TRANSFORM
    )[0]
    merging_mode = random.choices(
        ['paint', 'add_sub'],
        config.DATA_GEN.P_MERGING_MODES
    )[0]

    if merging_mode == 'add_sub' or fg_transform == 'gauss_smooth':
        label_int = label_int - bg_mean_int # to still enforce delta

    # transform fg
    fg = label.clone()
    fg[label == 1] = label_int.abs()
    if not fg_transform == 'none':
        fg = fg_bg_transforms[fg_transform](fg)

    # merge fg into bg
    special_merge = ['gauss_smooth', 'dropout', 'roll', 'hull'] # distorted tubular appearance
    mask = label == 1 if fg_transform not in special_merge else fg > 0
    if merging_mode == 'paint' and fg_transform != 'gauss_smooth':
        background[mask] = fg[mask]
    elif merging_mode == 'add_sub' or fg_transform == 'gauss_smooth':
        if label_int < 0:
            background[mask] -= fg[mask]
        else:
            background[mask] += fg[mask]

    return background

def get_label_transforms(config):
    if config.DATA_GEN.TRANSFORM_LABEL:    # stack transformations on labels
        transforms = [
            RandSpatialCrop(
                roi_size=(config.DATA_GEN.IMG_SIZE, config.DATA_GEN.IMG_SIZE, config.DATA_GEN.IMG_SIZE),
                random_center=True, random_size=False
            ),
            # spatial transforms
            RandFlip(
                spatial_axis=0, prob=config.DATA_GEN.LABEL.P_FLIP[0]
            ),
            RandFlip(
                spatial_axis=1, prob=config.DATA_GEN.LABEL.P_FLIP[1]
            ),
            RandFlip(
                spatial_axis=2, prob=config.DATA_GEN.LABEL.P_FLIP[2]
            ),
            RandRotate90(
                spatial_axes=(0, 1), prob=config.DATA_GEN.LABEL.P_ROTATE[0]
            ),
            RandRotate90(
                spatial_axes=(0, 2), prob=config.DATA_GEN.LABEL.P_ROTATE[1]
            ),
            RandRotate90(
                spatial_axes=(1, 2), prob=config.DATA_GEN.LABEL.P_ROTATE[2]
            ),
            RandDilation(
                prob=config.DATA_GEN.LABEL.P_DILATE, num_max_steps=config.DATA_GEN.LABEL.DILATE_MAX_STEPS
            ),
            RandZoom(
                prob=config.DATA_GEN.LABEL.P_ZOOM, min_zoom=config.DATA_GEN.LABEL.MIN_ZOOM, max_zoom=config.DATA_GEN.LABEL.MAX_ZOOM,
                mode='nearest'
            ),
            Rand3DElastic(
                prob=config.DATA_GEN.LABEL.P_ELASTIC, sigma_range=config.DATA_GEN.LABEL.ELASTIC_SIGMA,
                magnitude_range=config.DATA_GEN.LABEL.ELASTIC_MAGNITUDE, padding_mode='reflection', mode='nearest'
            ),
            RandBinarySmooth(
                prob=config.DATA_GEN.LABEL.P_BINARY_SMOOTH, sigma=config.DATA_GEN.LABEL.BINARY_SMOOTH_SIGMA,
                thresh=config.DATA_GEN.LABEL.BINARY_SMOOTH_THRESH
            ),
            EnsureType(
                track_meta=False,
            )
        ]
    else:
        transforms = [
            RandSpatialCrop(
                roi_size=(config.DATA_GEN.IMG_SIZE, config.DATA_GEN.IMG_SIZE, config.DATA_GEN.IMG_SIZE),
                random_center=True, random_size=False
            ),
            RandFlip(
                spatial_axis=0, prob=config.DATA_GEN.LABEL.P_FLIP[0]
            ),
            RandFlip(
                spatial_axis=1, prob=config.DATA_GEN.LABEL.P_FLIP[1]
            ),
            RandFlip(
                spatial_axis=2, prob=config.DATA_GEN.LABEL.P_FLIP[2]
            ),
            EnsureType(
                track_meta=False,
            )
        ]
    return Compose(transforms)

def get_int_transforms(config):
    transforms = [
        # intensity transforms
        RandBiasField(
            prob=config.DATA_GEN.INTENSITY.P_BIAS_FIELD, degree=config.DATA_GEN.INTENSITY.BIAS_FIELD_DEGREE,
            coeff_range=config.DATA_GEN.INTENSITY.BIAS_FIELD_RANGE
        ),
        RandGaussianNoise(
            prob=config.DATA_GEN.INTENSITY.P_GAUSSIAN_NOISE, mean=config.DATA_GEN.INTENSITY.GAUSSIAN_NOISE_MEAN,
            std=config.DATA_GEN.INTENSITY.GAUSSIAN_NOISE_STD, sample_std=True
        ),
        RandKSpaceSpikeNoise(
            prob=config.DATA_GEN.INTENSITY.P_KSPACE_SPIKE
        ),
        RandAdjustContrast(
            prob=config.DATA_GEN.INTENSITY.P_CONTRAST, gamma=config.DATA_GEN.INTENSITY.CONTRAST_GAMMA
        ),
        RandGaussianSmoothSigmaMod(
            prob=config.DATA_GEN.INTENSITY.P_GAUSSIAN_SMOOTH, prob_common=config.DATA_GEN.INTENSITY.P_GAUSSIAN_SMOOTH_COMMON_SIGMA,
            sigma_range=config.DATA_GEN.INTENSITY.GAUSSIAN_SMOOTH_SIGMA_RANGE, sigma_x=config.DATA_GEN.INTENSITY.GAUSSIAN_SMOOTH_SIGMA_X,
            sigma_y=config.DATA_GEN.INTENSITY.GAUSSIAN_SMOOTH_SIGMA_Y, sigma_z=config.DATA_GEN.INTENSITY.GAUSSIAN_SMOOTH_SIGMA_Z
        ),
        RandRicianNoise(
            prob=config.DATA_GEN.INTENSITY.P_RICIAN_NOISE, mean=config.DATA_GEN.INTENSITY.RICIAN_NOISE_MEAN,
            std=config.DATA_GEN.INTENSITY.RICIAN_NOISE_STD, sample_std=True
        ),
        RandGibbsNoise(
            prob=config.DATA_GEN.INTENSITY.P_GIBBS_NOISE, alpha=config.DATA_GEN.INTENSITY.GIBBS_ALPHA,
        ),
        RandGaussianSharpen(
            prob=config.DATA_GEN.INTENSITY.P_GAUSSIAN_SHARPEN, alpha=config.DATA_GEN.INTENSITY.GAUSSIAN_SHARPEN_ALPHA
        ),
        RandHistogramShift(
            prob=config.DATA_GEN.INTENSITY.P_HISTOGRAM_SHIFT, num_control_points=config.DATA_GEN.INTENSITY.HISTOGRAM_SHIFT_CPOINTS
        ),        
        RandMedianSmooth(
            prob=config.DATA_GEN.INTENSITY.P_MEDIAN_SMOOTH, radius_range=config.DATA_GEN.INTENSITY.MEDIAN_SMOOTH_RADIUS
        ),
        EnsureType(
            track_meta=False,
        )
    ]
    return Compose(transforms)

def get_fg_bg_transforms(config):
    transforms = {} # dict of fg and bg transforms

    transforms['bias_field'] = RandBiasField(
        prob=1, degree=config.DATA_GEN.FOREGROUND.BIAS_FIELD_DEGREE,
        coeff_range=config.DATA_GEN.FOREGROUND.BIAS_FIELD_RANGE
    )
    transforms['gauss_noise'] = RandGaussianNoise(
        prob=1, mean=config.DATA_GEN.FOREGROUND.GAUSSIAN_NOISE_MEAN,
        std=config.DATA_GEN.FOREGROUND.GAUSSIAN_NOISE_STD, sample_std=True
    )
    transforms['gauss_smooth'] = RandGaussianSmooth(
        prob=1, sigma_x=[0, 0], sigma_z=[0, 0],
        sigma_y=config.DATA_GEN.FOREGROUND.GAUSSIAN_SMOOTH_SIGMA, 
    )
    transforms['hull'] = RandHull(prob=1, hull_thickness_range=config.DATA_GEN.FOREGROUND.HULL_THICKNESS_RANGE)
    transforms['roll'] = RandRoll(prob=1, roll_range=config.DATA_GEN.FOREGROUND.ROLL_RANGE)
    transforms['dropout'] = RandDropOut(prob=1, prob_drop=config.DATA_GEN.FOREGROUND.DROPOUT)
    transforms['invert'] = RandInvert(prob=config.DATA_GEN.BACKGROUND.P_INVERT)
    return transforms

def build_loader(config):
    dataset = DRandGen(config)
    dataloader = DataLoader(
        dataset, batch_size=config.DATA_GEN.BATCH_SIZE, 
        shuffle=True, num_workers=config.DATA_GEN.NUM_WORKERS, 
        drop_last=True, pin_memory=True
    )
    return dataloader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # seed for reproducible results
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)

    config = load_config(dict=False)
    loader = build_loader(config)
    path_to_store_data = Path(config.DATA_GEN.OUT_DIR)

    labels, labels_int_mod = next(iter(loader))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    # generate example image
    for idx, img in enumerate(labels_int_mod):
        ax = axes[idx // 4, idx % 4]
        ax.imshow(img[:, 50, :, :].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    time = datetime.now().strftime("%H_%M_%S")
    plt.savefig(f'./d_drand_sample_{time}.png')

    # generate samples
    num_samples = 500000
    idx = 0
    for sample in iter(loader):
        labels, labels_int_mod = sample

        for label, label_int_mod in zip(labels, labels_int_mod):

            if idx >= num_samples:
                break
            print(idx)

            path_to_store_data_sample = path_to_store_data / str(idx)
            path_to_store_data_sample.mkdir(parents=True, exist_ok=True)

            np.save(path_to_store_data_sample / 'mask.npy', label.squeeze().numpy().astype(np.bool_))
            np.save(path_to_store_data_sample / 'img.npy', label_int_mod.squeeze().numpy().astype(np.float16))

            idx += 1