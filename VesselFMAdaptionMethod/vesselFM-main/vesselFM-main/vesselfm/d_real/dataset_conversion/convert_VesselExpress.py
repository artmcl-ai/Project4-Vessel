import os
import numpy as np
from libtiff import TIFFfile
from .utils import save_array, save_metadata, calculate_metadata


def convert_sample(folder, out_folder):
    img_path = os.path.join(folder, "Raw")
    print(f"Out folder: {out_folder}")

    for file in os.listdir(img_path):
        print(f"converting file: {file}...")
        if file.endswith(".json"):
            continue
        img_arr = np.array(TIFFfile(os.path.join(img_path, file)).get_tiff_array())
        img_arr = np.squeeze(img_arr)
        if img_arr.shape[1] > 600:
            metadata = []
            for i in range(0, img_arr.shape[0], 500):
                img = img_arr[i : min(i + 500, img_arr.shape[0])]
                metadata.append(calculate_metadata(img))
            metadata = {
                "max": np.max([m["max"] for m in metadata]),
                "min": np.min([m["min"] for m in metadata]),
                "mean": np.mean([m["mean"] for m in metadata]),
                "std": np.median([m["std"] for m in metadata]),
                "shape": [img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]],
                "p95": np.median([m["p95"] for m in metadata]),
                "p5": np.median([m["p5"] for m in metadata]),
            }
        else:
            metadata = calculate_metadata(img_arr)
        out_name = file.split(".")[0].replace(" ", "_")
        save_array(img_arr, os.path.join(out_folder, "imagesTr", out_name))
        save_metadata(metadata, os.path.join(out_folder, "imagesTr", out_name))

    mask_path = os.path.join(folder, "Binary")
    for file in os.listdir(mask_path):
        if file.endswith(".json"):
            continue
        mask_arr = np.array(TIFFfile(os.path.join(mask_path, file)).get_tiff_array())
        mask_arr = np.squeeze(mask_arr)
        mask_arr = mask_arr > 0
        if mask_arr.shape[1] > 600:
            metadata = []
            for i in range(0, mask_arr.shape[0], 500):
                mask = mask_arr[i : min(i + 500, mask_arr.shape[0])]
                metadata.append(calculate_metadata(mask))
            metadata = {
                "max": np.max([m["max"] for m in metadata]),
                "min": np.min([m["min"] for m in metadata]),
                "mean": np.mean([m["mean"] for m in metadata]),
                "std": np.median([m["std"] for m in metadata]),
                "shape": [mask_arr.shape[0], mask_arr.shape[1], mask_arr.shape[2]],
                "p95": np.median([m["p95"] for m in metadata]),
                "p5": np.median([m["p5"] for m in metadata]),
            }
        else:
            metadata = calculate_metadata(mask_arr)
        out_name = file.split(".")[0].replace("Binary_", "").replace(" ", "_")
        save_array(mask_arr, os.path.join(out_folder, "labelsTr", out_name))
        save_metadata(metadata, os.path.join(out_folder, "labelsTr", out_name))


def convert_VesselExpress(folder, out_folder):
    organs = [
        "Bladder",  # used in our experiments
        "Brain",
        "Ear",
        "Heart",    # used in our experiments
        "Liver",
        "Muscle",
        "Spinalcord",
        "Tongue",
        "Figure2",  # used in our experiments (brain)
        "Figure3"
    ]

    for organ in organs:
        os.makedirs(os.path.join(out_folder + "_" + organ), exist_ok=True)
        os.makedirs(os.path.join(out_folder + "_" + organ, "imagesTr"), exist_ok=True)
        os.makedirs(os.path.join(out_folder + "_" + organ, "labelsTr"), exist_ok=True)

    for figure in ["Figure2C_E", "Figure3", "Figure4"]:
        curr_path = os.path.join(folder, figure)
        if figure == "Figure2C_E":
            sample_name = "VEG_4.12mm^3"    # biggest volume
            convert_sample(os.path.join(curr_path, sample_name), os.path.join(out_folder + "_" + "Figure2"),)
        elif figure == "Figure3":
            for subfolder in os.listdir(curr_path):
                curr_path = os.path.join(folder, figure, subfolder)
                convert_sample(curr_path, os.path.join(out_folder + "_" + "Figure3"))
        else: 
            for subfolder in os.listdir(curr_path):
                curr_path = os.path.join(folder, figure, subfolder)
                convert_sample(curr_path, os.path.join(out_folder + "_" + subfolder))
