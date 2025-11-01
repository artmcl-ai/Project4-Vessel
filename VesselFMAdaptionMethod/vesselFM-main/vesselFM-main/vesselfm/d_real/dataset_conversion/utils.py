import os
import json
import numpy as np
import SimpleITK as sitk
from skimage.filters import gaussian


def calculate_metadata(array: np.ndarray):
    """
        This function will calculate the metadata of the image.
    """
    array = array.astype(np.float64)
    return {"max": array.max(), "min": array.min(), "mean": array.mean(), "std": array.std(), "shape": array.shape, "p95": np.percentile(array, 95), "p5": np.percentile(array, 5)}

def save_array(array: np.ndarray, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    for i in range(array.shape[0]):
        np.save(os.path.join(output_folder, f"{i:04d}.npy"), array[i])

def save_metadata(metadata: dict, output_folder: str):
    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f)

def read_DICOM_series(folder: str):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    print(f"Image type: {image.GetPixelIDTypeAsString()}")
    return image

def convert_sitk_image(image: sitk.Image):
    spacing = image.GetSpacing()
    spacing = [spacing[2], spacing[0], spacing[1]]
    return sitk.GetArrayFromImage(image), {"origin": image.GetOrigin(), "spacing": spacing, "direction": image.GetDirection()}

def resample_sample(image: sitk.Image, label: sitk.Image, factor, default_value=None):
    if isinstance(factor, float) or isinstance(factor, int):
        factor = [factor for _ in range(3)]
    new_size = [int(dim * f) for (dim, f) in zip(image.GetSize(), factor)]

    if default_value is None:
        default_value = np.min(sitk.GetArrayFromImage(image))
        default_value = int(np.floor(default_value))

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(
        [
            spc * sz / nsz
            for (sz, spc, nsz) in zip(image.GetSize(), image.GetSpacing(), new_size)
        ]
    )
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    resampled_image = resampler.Execute(image)

    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_label = resampler.Execute(label)

    return resampled_image, resampled_label

def resample_image(image: sitk.Image, factor, default_value=None):
    if isinstance(factor, float) or isinstance(factor, int):
        factor = [factor for _ in range(3)]
    new_size = [int(dim * f) for (dim, f) in zip(image.GetSize(), factor)]

    if default_value is None:
        default_value = np.min(sitk.GetArrayFromImage(image))
        default_value = int(np.floor(default_value))

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(
        [
            spc * sz / nsz
            for (sz, spc, nsz) in zip(image.GetSize(), image.GetSpacing(), new_size)
        ]
    )
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    resampled_image = resampler.Execute(image)
    return resampled_image

def resample_label(label: sitk.Image, factor):
    if isinstance(factor, float) or isinstance(factor, int):
        factor = [factor for _ in range(3)]
    new_size = [int(dim * f) for (dim, f) in zip(label.GetSize(), factor)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(
        [
            spc * sz / nsz
            for (sz, spc, nsz) in zip(label.GetSize(), label.GetSpacing(), new_size)
        ]
    )
    resampler.SetOutputOrigin(label.GetOrigin())
    resampler.SetOutputDirection(label.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampled_label = resampler.Execute(label)

    return resampled_label

def smooth_label(label: np.ndarray, threshold: int = 0.00001, sigma: int = 0.001):
    median_filter = gaussian(label, sigma=sigma)
    return median_filter > threshold
