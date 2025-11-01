$\mathcal{D}_\text{real}$ is comprised of 17 individual datasets. Below, we detail how these datasets can be accessed. You may have to explicitly request access for some of them.

| Dataset Name  | Access |
|-----------|-----------|
| SMILE-UHURA | https://www.soumick.com/en/uhura/, https://www.synapse.org/Synapse:syn47164761/wiki/620033 |  # request access
| BvEM | https://huggingface.co/datasets/pytc/BvEM |
| OCTA | https://huggingface.co/datasets/bwittmann/syn-cerebral-octa-seg |
| MSD8 | http://medicaldecathlon.com/#tasks (hepatic vessels, task 8)|
| TubeTK | https://data.kitware.com/#collection/591086ee8d777f16d01e0724/folder/58a372fa8d777f0721a64dfb |
| tUbeNet | https://rdr.ucl.ac.uk/articles/dataset/3D_Microvascular_Image_Data_and_Labels_for_Machine_Learning/25715604?file=45996213 |
| TopCoW | https://topcow23.grand-challenge.org/data |
| VesSAP | https://discotechnologies.org/VesSAP | # request access
| DeepVesselNet | https://github.com/giesekow/deepvesselnet/wiki/Datasets | # request access
| HR-Kidney | https://idr.openmicroscopy.org/webclient/?show=project-2701 |
| 3D-IRCADb-01 | https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01 |
| DeepVess | https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3 |
| CSD | http://xzbai.buaa.edu.cn/datasets.html, https://brain-development.org/ixi-dataset |
| VesselExpress | https://zenodo.org/records/6025935#.YajQ1S9Xb5k |
| MiniVess | https://search.kg.ebrains.eu/instances/bf268b89-1420-476b-b428-b85a913eb523 |
| HiP-CT | https://www.kaggle.com/competitions/blood-vessel-segmentation/data |
<!-- | LS | https://www.cell.com/action/showPdf?pii=S0896-6273%2824%2900057-6 | # request access -->

Note that we do **not** provide licenses for these datasets. Therefore, one must adhere to their specified licensing terms.

## Initial Pre-Processing
To pre-process all above datasets and standardize their format, run:

    python vesselfm/d_real/dataset_conversion.py </path/to/raw_dataset> </path/to/preprocessed_dataset>

Note that the name of the raw dataset folder should correspond to a dataset specified in `dataset_conversion.py` (e.g., `3dircadb1`).
Additional information - if required - is provided in the comments of the respective scripts.

## Patch Extraction & Offline Data Augmentation
To speed up training, we apply data augmentation on patches extracted from the pre-processed datasets offline. 

    python extract_patches.py

Please edit `config/dataset_creation.yaml` to include the datasets of interest and adjust the paths (see `#TODO`).
One can utilize our algorithm for label improvement on HR-Kidney operating on patches for compute efficiency by running `dataset_conversion/improve_HR_kidney.py` to obtain `HRKidney_thresh` used in our experiments.

The final structure of $\mathcal{D}_\text{real}$ must be organized as follows:
```
/path/to/d_real   # D_real
└── 3Dircadb1/  # individual datasets
  └── 0/   # sample 0
    └── img.npy   # pre-processed, extracted, augmented patch of shape 128x128x128
    └── mask.npy  # matching label of shape 128x128x128
  ...
└── DeepVess/
  └── 0/
    └── img.npy
    └── mask.npy
  ...
└── DeepVesselNet_syncrotone/
  └── 0/
    └── img.npy
    └── mask.npy
  ...
...
```