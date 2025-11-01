import os
import argparse
import dataset_conversion as preprocessing
import sys 


def main(folder:str, out_folder:str):
    basename = os.path.basename(folder)
    print(f"basename of folder {folder} is {basename}.")
    if basename == "":
        basename = os.path.basename(os.path.dirname(folder))
        print(f"corrected basename of folder {folder} is {basename}.")
    if basename not in out_folder:
        out_folder = os.path.join(out_folder, basename)

    if basename.lower() == "3dircadb1":
        preprocessing.convert_3Dircadb1(folder, out_folder)
    elif basename.lower() == "cerebrovascular_segmentation_dataset":
        preprocessing.convert_CSD(folder, out_folder)
    elif basename.lower() == "minivess":
        preprocessing.convert_MiniVess(folder, out_folder)
    elif basename.lower() == "topcow":
        preprocessing.convert_TopCoW(folder, out_folder)
    elif basename.lower() == "tubetk":
        preprocessing.convert_TubeTK(folder, out_folder)
    elif basename.lower() == "msd_task8":
        preprocessing.convert_MSD(folder, out_folder)
    elif basename.lower() == "octa":
        preprocessing.convert_OCTA(folder, out_folder)
    elif basename.lower() == "smile":
        preprocessing.convert_SMILE(folder, out_folder)
    elif basename.lower() == "hip-ct":
        preprocessing.convert_HiP_CT(folder, out_folder)
    elif basename.lower() == "bvem":
        preprocessing.convert_BvEM(folder, out_folder)
    elif basename.lower() == "hr-kidney":
        preprocessing.convert_HR_kidney(folder, out_folder)
    elif basename.lower() == "deepvesselnet":
        preprocessing.convert_DeepVesselNet(folder, out_folder)
    elif basename.lower() == "deepvess":
        preprocessing.convert_DeepVess(folder, out_folder)
    elif basename.lower() == "vesselexpress":
        preprocessing.convert_VesselExpress(folder, out_folder)
    elif basename.lower() == "vessap_anno":
        preprocessing.convert_VesSAP_anno(folder, out_folder)
    elif basename.lower() == "tubenet":
        preprocessing.convert_tUbeNet(folder, out_folder)
    elif basename.lower() == "lightsheet":
        preprocessing.convert_Lightsheet(folder, out_folder)


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_folder", type=str)
    args = parser.parse_args()
    print(args.input_folder, args.output_folder)
    main(args.input_folder, args.output_folder)
