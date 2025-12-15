## Installation of VesselFM Conda Environment (Not Required If Using Docker)

First, set up a conda environment and install dependencies:

    conda create -n vesselfm python=3.9

    conda activate vesselfm

    pip install -e .


## Files that were created and editted:

- VesselFMAdaptionMethod\vesselFM-main\vesselFM-main\configs

- VesselFMAdaptionMethod\vesselFM-main\vesselFM-main\vesselFM/seg (everything except dataset.py, finetune.py, and train.py)

- VesselFMAdaptionMethod\vesselFM-main\vesselFM-main\vesselfm\seg\utils\evaluation.py

- VesselFMAdaptionMethod\vesselFM-main\vesselFM-main\vesselfm\seg\configs\inference.yaml

- VesselFMAdaptionMethod\vesselFM-main\vesselFM-main\vesselfm\seg\configs\eval_inference.yaml

## Credit

```
@InProceedings{Wittmann_2025_CVPR,
    author    = {Wittmann, Bastian and Wattenberg, Yannick and Amiranashvili, Tamaz and Shit, Suprosanna and Menze, Bjoern},
    title     = {vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20874-20884}
}
```

## License

Original license applies from the VesselFM Repository found at https://github.com/bwittmann/vesselFM/tree/main

Code in this repository is licensed under GNU General Public License v3.0. Model weights are released under Open RAIL++-M License and are restricted to research and non-commercial use only. Model use must comply with potential licenses, regulations, and restrictions arising from the use of named data sets during model training.
