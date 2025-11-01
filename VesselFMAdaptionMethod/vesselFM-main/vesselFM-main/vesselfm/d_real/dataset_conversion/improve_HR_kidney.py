import os
import multiprocessing as mp

import numpy as np
from scipy import ndimage
from skimage import morphology


def handle_sample(sample, input_path, output_path):
    print(f"Processing {sample}")
    mask = np.load(os.path.join(input_path, sample, "mask.npy"))

    if np.sum(mask)/np.prod(mask.shape) < 0.05:
        print(f"Skipping {sample}")
        return
    del mask

    image = np.load(os.path.join(input_path, sample, "img.npy"))
    image = image.astype(np.float32)
    filtered_image = ndimage.median_filter(image, size=11)
    mask = (image - filtered_image) > 0.1
    mask = np.logical_or(mask, image > 0.9)
    mask = morphology.binary_closing(mask, footprint=np.ones((3, 3, 3)))
    mask = morphology.remove_small_objects(mask, min_size=100)

    if np.sum(mask)/np.prod(mask.shape) > 0.05:
        print(f"Saving {sample}")
        os.makedirs(os.path.join(output_path, sample), exist_ok=True)
        np.save(os.path.join(output_path, sample, "img.npy"), image.astype(np.float16))
        np.save(os.path.join(output_path, sample, "mask.npy"), mask.astype(bool))

    del image
    del filtered_image
    del mask

def HR_kidney_label_improvement(input_path, output_path, cpus=32):
    os.makedirs(output_path, exist_ok=True)
    samples = os.listdir(input_path)
    args = [(sample, input_path, output_path) for sample in samples]
    with mp.Pool(processes=cpus) as pool:
        pool.starmap(handle_sample, args)


if __name__ == "__main__":
    HR_kidney_label_improvement(
        "/path/to/d_real/HRKidney", # TODO
        "/path/to/d_real/HRKidney_thres" # TODO
    )