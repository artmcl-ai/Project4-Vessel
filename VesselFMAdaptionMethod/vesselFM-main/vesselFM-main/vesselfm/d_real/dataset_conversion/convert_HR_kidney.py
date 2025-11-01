import os
import SimpleITK as sitk
import numpy as np
from tifffile import TiffFile
from .utils import save_metadata, calculate_metadata


def convert_HR_kidney(input_folder, output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "imagesTr", "kidney2"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labelsTr", "kidney2"), exist_ok=True)

    label = input_folder + "/kidney2_D5_blood_vessels.ome.tiff"
    image = input_folder + "/kidney2_D3_inpainted.ome.tiff"

    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(label)
    file_reader.ReadImageInformation()
    spacing = file_reader.GetSpacing()
    spacing = [spacing[2], spacing[1], spacing[0]]
    metadata = {"origin": file_reader.GetOrigin(), "spacing": spacing, "direction": file_reader.GetDirection()}
    print(f"Metadata: {metadata}")
    shape = list(file_reader.GetSize())
    print(f"shape {shape}")

    metadata = []
    with TiffFile(label) as tif:
        for i in range(shape[-1]):
            slice = i
            img_array = tif.pages[i].asarray()
            metadata.append(calculate_metadata(img_array))
            print(f"slice {i}: total {img_array.sum()}")
            np.save(os.path.join(output_path, "labelsTr", "kidney2", f"{i:04d}.npy"), img_array)
    metadata = {"max": np.max([m["max"] for m in metadata]), "min": np.min([m["min"] for m in metadata]), "mean": np.mean([m["mean"] for m in metadata]), "std": np.median([m["std"] for m in metadata]), "shape": [len(metadata), shape[0], shape[1]], "p95": np.median([m["p95"] for m in metadata]), "p5": np.median([m["p5"] for m in metadata])}
    print(f"slices {slice}")
    save_metadata(metadata, os.path.join(output_path, "labelsTr", "kidney2"))

    metadata = []
    with TiffFile(image) as tif:
        for i in range(shape[-1]):
            slice = i
            img_array = tif.pages[i].asarray()
            metadata.append(calculate_metadata(img_array))
            print(f"slice {i}: total {img_array.sum()}")
            np.save(os.path.join(output_path, "imagesTr", "kidney2", f"{i:04d}.npy"), img_array)
    metadata = {"max": np.max([m["max"] for m in metadata]), "min": np.min([m["min"] for m in metadata]), "mean": np.mean([m["mean"] for m in metadata]), "std": np.median([m["std"] for m in metadata]), "shape": [len(metadata), shape[0], shape[1]], "p95": np.median([m["p95"] for m in metadata]), "p5": np.median([m["p5"] for m in metadata])}
    print(f"slices {slice}")
    save_metadata(metadata, os.path.join(output_path, "imagesTr", "kidney2"))
