import os
import sys
import multiprocessing as mp
from pathlib import Path

import hydra
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from torch.utils.data import Dataset
from monai import transforms

from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.data import generate_transforms


class HelperDataset(Dataset):
    """
    Helper dataset class to extract patches from D_real and apply offline augmentation.
    """
    def __init__(self, dataset_configs, mode):
        super().__init__()
        # init datasets
        dataset_config = next(iter(dataset_configs.values()))
        path = Path(dataset_config['path'])
        self._path_imgs = sorted((path / 'imagesTr').iterdir())
        self._path_labels = sorted((path / 'labelsTr').iterdir())
        self._transforms = generate_transforms(dataset_config.transforms[mode])
        self._reader = determine_reader_writer(dataset_config.file_format)()
        
    def __len__(self):
        return len(self._path_imgs)

    def __getitem__(self, idx):
        path_img = self._path_imgs[idx]
        path_label = self._path_labels[idx]

        img = self._reader.read_images(path_img)[0].astype(np.float32)
        label = self._reader.read_images(path_label)[0].astype(bool)
        
        transformed_data = self._transforms({'Image': img, 'Mask': label})
        img, label = transformed_data['Image'], transformed_data['Mask']
        return img, label


def crop_fg(image, mask):
    trans = transforms.CropForegroundd(
        keys=["image", "mask"], source_key="mask", allow_smaller=True
    )
    res = trans({"image": image, "mask": mask})
    return res["image"], res["mask"]

def get_transforms(config):
    trans = transforms.Compose(
        [
            transforms.RandSpatialCropd(
                ["image", "mask"], config.input_size, random_size=False
            ),
            transforms.RandFlipd(
                ["image", "mask"], spatial_axis=0, prob=config.p_flip[0]
            ),
            transforms.RandFlipd(
                ["image", "mask"], spatial_axis=1, prob=config.p_flip[1]
            ),
            transforms.RandFlipd(
                ["image", "mask"], spatial_axis=2, prob=config.p_flip[2]
            ),
            transforms.RandRotated(
                ["image", "mask"],
                range_x=config.rotation_range[0],
                range_y=config.rotation_range[1],
                range_z=config.rotation_range[2],
                prob=config.p_rotate,
                keep_size=True,
                padding_mode="reflect",
            ),
            transforms.CenterSpatialCropd(["image", "mask"], config.input_size),
            transforms.Rand3DElasticd(
                ["image", "mask"],
                prob=config.p_elastic,
                sigma_range=config.elastic_sigma,
                magnitude_range=config.elastic_mag,
                padding_mode="reflect",
                mode=["bilinear", "nearest"],
            ),
            transforms.RandZoomd(
                ["image", "mask"],
                min_zoom=config.min_zoom,
                max_zoom=config.max_zoom,
                prob=config.p_zoom,
            ),
            (
                transforms.ScaleIntensityRangePercentilesd(
                    keys=["image"],
                    lower=config.lower,
                    upper=config.upper,
                    b_min=0,
                    b_max=1,
                )
                if config.use_percentiles_intensity_scaling
                else transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=1)
            )
        ]
    )
    return trans

def smooth_mask(mask):
    mask_prec = mask.sum() / np.prod(mask.shape)
    mask = ndimage.median_filter(mask.numpy(), size=5)
    mask = mask > 0.5
    post_mask_prec = mask.sum() / np.prod(mask.shape)
    print(f"mask_prec: {mask_prec}, post_mask_prec: {post_mask_prec}")
    return mask

def pad_or_resize(img, is_mask=False, upsampled=False):
    if img.shape[1] < 80:
        factor = min(128 / img.shape[1], 1.5)
        image_size = (
            int(img.shape[1] * factor),
            int(img.shape[2] * factor),
            int(img.shape[3] * factor),
        )
        # Upsample image to 70
        img = transforms.Resize(
            spatial_size=image_size, mode="nearest" if is_mask else "trilinear"
        )(img)
        upsampled = True

    while img.shape[1] < 128:
        padding_z = max(min((140 - img.shape[1]) // 2, img.shape[1]), 1)
        padding_x = max(min((140 - img.shape[2]) // 2, img.shape[2]), 1)
        padding_y = max(min((140 - img.shape[3]) // 2, img.shape[3]), 1)
        img = np.pad(
            img,
            [
                (0, 0),
                (padding_z, padding_z),
                (padding_x, padding_x),
                (padding_y, padding_y),
            ],
            mode="reflect",
        )
        upsampled = True
    return img, upsampled

def create_sample(sample, dataset, path, dataset_config):
    np.random.seed(sample)
    skipped = 0
    transforms = get_transforms(dataset_config.transforms_config)

    while True:
        rand = np.random.randint(0, len(dataset))
        print(f"Creating sample {sample} from id {rand}.")

        image, mask = dataset[rand]
        image, mask = crop_fg(image, mask)

        # skip samples with little annotations
        if (skipped <= 5 and mask.sum() < (0.01 * np.prod(mask.shape))) or (skipped > 5 and mask.sum() < (0.005 * np.prod(mask.shape))):
            print(f"{mask.sum()/ np.prod(mask.shape)} < {0.01} or {mask.sum()/ np.prod(mask.shape)} < {0.005} skipped {skipped}")
            skipped += 1
            continue

        image, upsampled = pad_or_resize(image)
        mask, _ = pad_or_resize(mask, is_mask=True)
        assert mask.shape == image.shape, "shape diff between image and mask"

        res_dict = transforms({"image": image, "mask": mask})
        image, mask = res_dict["image"], res_dict["mask"] > 0.0

        if upsampled or path.endswith("BvEM"):
            mask = smooth_mask(mask)
        image, mask = image.squeeze(), mask.squeeze()

        # store data
        os.makedirs(f"{path}/{sample}", exist_ok=True)

        if not np.isfinite(image).all():
            print("non normal number encountered")

        np.save(f"{path}/{sample}/img.npy", image.astype(np.float16))
        np.save(f"{path}/{sample}/mask.npy", mask.astype(bool))
        # image = nib.Nifti1Image(image.float().cpu().numpy().squeeze(), affine=np.eye(4))
        # nib.save(image, f"{path}/{sample}/img.nii")
        # mask = nib.Nifti1Image(mask.int().cpu().numpy().squeeze(), affine=np.eye(4))
        # nib.save(mask, f"{path}/{sample}/mask.nii")
        return


def create_dataset(dataset, key, config, path):
    num_samples = config.dataset[key].num_samples
    path = f"{path}/{key}"

    print(f"Creating dataset in {path} with {num_samples} samples")
    os.makedirs(path, exist_ok=True)

    with mp.Pool(processes=config.cpus) as pool:
        pool.starmap(
            create_sample,
            [(sample, dataset, path, config.dataset[key]) for sample in range(num_samples)],
        )
    print(f"Done with dataset at {path}", '\n')


def handel_dataset(key, dataset, config):
    print(f"Creating dataset {key}: dataset")
    path = config.path
    dataset = HelperDataset({key: dataset}, mode='train')
    create_dataset(dataset, key, config, path)


@hydra.main(version_base=None, config_path="config", config_name="dataset_creation")
def generate_dataset(config):
    data_cfg = config.dataset.items()
    data_cfg = [(a, b, config) for a, b in data_cfg if a in config.generate_datasets]

    for key, dcfg, cfg in data_cfg:
        handel_dataset(key, dcfg, cfg)


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    generate_dataset()
