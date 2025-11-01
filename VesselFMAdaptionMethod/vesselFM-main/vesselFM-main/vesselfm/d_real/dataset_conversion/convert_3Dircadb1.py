import os
import sys
import zipfile
import SimpleITK as sitk
from .utils import save_array, save_metadata, read_DICOM_series, calculate_metadata

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def load_DICOM_and_meta(input_folder: str):
    basename = os.path.basename(input_folder).split(".")[-1]
    # Read the DICOM series
    image = read_DICOM_series(os.path.join(input_folder, "PATIENT_DICOM"))
    # Convert 255 to 1
    image_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    spacing = [spacing[2], spacing[0], spacing[1]]
    metadata = {"origin": image.GetOrigin(), "spacing": spacing, "direction": image.GetDirection()}
    return image_array, metadata, basename

def load_tubular_labels(sample_folder: str):
    labels = ["venoussystem", "artery", "portalvein"]
    label_to_index = {label: i+1 for i, label in enumerate(labels)}
    mask_folder = os.path.join(sample_folder, "MASKS_DICOM")
    print(f"Mask folder: {os.listdir(mask_folder)}")

    merged_labels = None
    for folder in os.listdir(mask_folder):
        is_label = [label in folder for label in labels]
        if not any(is_label):
            continue

        label_name =  labels[([i for i, x in enumerate(is_label) if x][0])]
        image = read_DICOM_series(os.path.join(mask_folder, folder))
        array = sitk.GetArrayFromImage(image)
        array = array > 0

        if merged_labels is None:
            merged_labels = array * label_to_index[label_name]
        else:
            merged_labels += array * label_to_index[label_name]
        
    metadata = label_to_index
    return merged_labels, metadata

def unzip(sample_folder: str):
    folders_to_unzip = ["PATIENT_DICOM", "MASKS_DICOM"]

    for folder in folders_to_unzip:
        if os.path.isdir(os.path.join(sample_folder, f"{folder}")):
            continue
        with zipfile.ZipFile(os.path.join(sample_folder, f"{folder}.zip"), "r") as zip_ref:
            zip_ref.extractall(sample_folder)

def convert_3Dircadb1(input_folder: str, output_folder: str):
    # Set up the new dataset folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labelsTr"), exist_ok=True)
    
    for folder in os.listdir(input_folder):
        print(f"Converting {folder}...")
        # Unzip all important folders in the input folder
        sample_folder = os.path.join(input_folder, folder)
        unzip(sample_folder)
        print("unziped folders")

        image_array, metadata, name = load_DICOM_and_meta(os.path.join(input_folder, folder))
        metadata = metadata | calculate_metadata(image_array)

        os.makedirs(os.path.join(output_folder, "imagesTr", name), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "labelsTr", name), exist_ok=True)

        save_array(image_array, os.path.join(output_folder, "imagesTr", name))
        save_metadata(metadata, os.path.join(output_folder, "imagesTr", name))
        print("converted Image")

        labels, metadata = load_tubular_labels(os.path.join(input_folder, folder))
        save_array(labels, os.path.join(output_folder, "labelsTr", name))
        save_metadata(metadata, os.path.join(output_folder, "labelsTr", name))
