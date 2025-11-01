import SimpleITK as sitk
import os
from monai.transforms import CropForegroundd, ScaleIntensityRangePercentilesd
from .utils import save_array, save_metadata, calculate_metadata, convert_sitk_image


def upsample(image, label, factor=4):
    # Upsample image in z direction by factor
    size = [
        label.GetSize()[0],
        label.GetSize()[1],
        int(label.GetSize()[2] * factor),
    ]
    label.SetSpacing([1, 1, factor])
    image.SetSpacing([1, 1, factor])

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing([1, 1, 1])
    resampler.SetOutputOrigin(label.GetOrigin())
    resampler.SetOutputDirection(label.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize(size)
    resampled_label = resampler.Execute(label)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([1, 1, 1])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize(size)

    resampled_image = resampler.Execute(image)

    return resampled_image, resampled_label


def convert_MSD(folder: str, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "imagesTs"), exist_ok=True)

    train_folder = "imagesTr"
    label_folder = "labelsTr"
    test_folder = "imagesTs"

    crop_foreground = CropForegroundd(keys=["image", "label"], source_key="label")
    scale_intensity = ScaleIntensityRangePercentilesd(
        keys=["image"], lower=20, upper=98, b_min=0, b_max=1, clip=True
    )

    for sample in os.listdir(os.path.join(folder, train_folder)):
        if sample.startswith("."):
            continue
        print(f"Converting {sample}...")
        image = sitk.ReadImage(os.path.join(folder, train_folder, sample))
        label = sitk.ReadImage(os.path.join(folder, label_folder, sample))
        image, label = upsample(image, label)
        array_img, metadata_img = convert_sitk_image(image)
        array_label, metadata_label = convert_sitk_image(label)
        array_label[array_label != 1] = 0
        array_img = array_img[None, ...]
        array_label = array_label[None, ...]
        res = crop_foreground({"image": array_img, "label": array_label})
        res = scale_intensity(res)
        array_img = res["image"][0]
        array_label = res["label"][0]
        metadata_img = metadata_img | calculate_metadata(array_img)
        save_array(
            array_img, os.path.join(out_folder, train_folder, sample.split(".")[0])
        )
        save_metadata(
            metadata_img, os.path.join(out_folder, train_folder, sample.split(".")[0])
        )

        save_array(
            array_label, os.path.join(out_folder, label_folder, sample.split(".")[0])
        )
        save_metadata(
            metadata_label, os.path.join(out_folder, label_folder, sample.split(".")[0])
        )

    for sample in os.listdir(os.path.join(folder, test_folder)):
        if sample.startswith("."):
            continue
        image = sitk.ReadImage(os.path.join(folder, test_folder, sample))
        array, metadata = convert_sitk_image(image)
        metadata = metadata | calculate_metadata(array)
        save_array(array, os.path.join(out_folder, test_folder, sample.split(".")[0]))
        save_metadata(
            metadata, os.path.join(out_folder, test_folder, sample.split(".")[0])
        )