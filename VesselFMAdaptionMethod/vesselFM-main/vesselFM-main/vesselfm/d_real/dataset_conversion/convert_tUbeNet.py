import SimpleITK as sitk
import numpy as np
import skimage.filters as filters
from libtiff import TIFFfile
import os
from .utils import (
    save_array,
    save_metadata,
    calculate_metadata,
    convert_sitk_image
)


def convert_tUbeNet(folder, out_folder):
    # Only two samples have good quality; only these will be converted
    # 2Photon
    f_name = out_folder
    out_folder = out_folder + "_2Photon"
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "labelsTr"), exist_ok=True)

    image = "Image data/2Photon_murineOlfactoryBulbLectin_subvolume500x500x79.tif"
    label = "Labels/2Photon_murineOlfactoryBulbLectin_subvolume500x500x79_labels.tif"
    image = os.path.join(folder, image)
    label = os.path.join(folder, label)

    image_arr = TIFFfile(image).get_tiff_array()
    label_arr = TIFFfile(label).get_tiff_array()

    # Resample
    spacing = [1, 1, 4]
    res_size = [500, 500, 356]

    image = sitk.GetImageFromArray(image_arr)
    image.SetSpacing(spacing)
    label = sitk.GetImageFromArray(label_arr)
    label.SetSpacing(spacing)

    resample = sitk.ResampleImageFilter()
    resample.SetSize(res_size)
    resample.SetOutputSpacing([1, 1, 1])
    resample.SetOutputOrigin([0, 0, 0])
    resample.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resample.SetTransform(sitk.Transform())

    image = resample.Execute(image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    label = resample.Execute(label)

    label_arr = sitk.GetArrayFromImage(label)

    # smooth label
    label_arr = filters.gaussian(label_arr, sigma=2)
    label_arr = label_arr > 0.7
    label_arr = label_arr.astype(np.uint8)

    label = sitk.GetImageFromArray(label_arr)

    image_arr, metadata = convert_sitk_image(image)
    label_arr, metadata_l = convert_sitk_image(label)

    metadata = metadata | calculate_metadata(image_arr)
    metadata_l = metadata_l | calculate_metadata(label_arr)

    save_array(image_arr, os.path.join(out_folder, "imagesTr"))
    save_array(label_arr, os.path.join(out_folder, "labelsTr"))
    save_metadata(metadata, os.path.join(out_folder, "imagesTr"))
    save_metadata(metadata_l, os.path.join(out_folder, "labelsTr"))

    # HREM MRI
    out_folder = f_name + "_MRI"
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "labelsTr"), exist_ok=True)

    image = "Image data/MRI_porcineLiver_0.9x0.9x5mm.tif"
    label = "Labels/MRI_porcineLiver_0.9x0.9x5mm_labels.tif"
    image = os.path.join(folder, image)
    label = os.path.join(folder, label)

    image_arr = TIFFfile(image).get_tiff_array()
    label_arr = TIFFfile(label).get_tiff_array()

    image = sitk.GetImageFromArray(image_arr)
    label = sitk.GetImageFromArray(label_arr)

    image_arr, metadata = convert_sitk_image(image)
    label_arr, metadata_l = convert_sitk_image(label)

    metadata = metadata | calculate_metadata(image_arr)
    metadata_l = metadata_l | calculate_metadata(label_arr)

    save_array(image_arr, os.path.join(out_folder, "imagesTr"))
    save_array(label_arr, os.path.join(out_folder, "labelsTr"))
    save_metadata(metadata, os.path.join(out_folder, "imagesTr"))
    save_metadata(metadata_l, os.path.join(out_folder, "labelsTr"))
