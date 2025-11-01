Config files are stored under [`configs`](./configs). Please adjust the placeholder paths and parameters marked with `#TODO`.

## Inference
Adjust the [inference](./configs/inference.yaml) config file (see `#TODO`) and run:

    python vesselfm/seg/inference.py

Images to be segmented should be placed in `/path/to/image_folder` as `.nii.gz` files. The results will be saved in `/path/to/output_folder`. Although we did not use them in our experiments, test-time augmentations (see `tta`) and post-processing steps (see `post`) can further improve the quality of the predicted segmentation mask. We have, therefore, included these features in the inference script as well. It is further strongly advised to adjust other inference parameters (e.g., `upper` and `lower` percentiles in `transforms_config`) to suit your data.

## Pre-Train on Three Data Sources
Adjust the [training](./configs/train.yaml) and [dataset](./configs/data/real_drand_flow.yaml) config files (see `#TODO`) and run:

    python vesselfm/seg/train.py

## Finetune and Evaluation (*Zero*-, *One*-, and *Few*-Shot)
Adjust the [finetune](./configs/finetune.yaml) and dataset ([BvEM](./configs/data/eval_bvem.yaml), [MSD8](./configs/data/eval_msd8.yaml), [OCTA](./configs/data/eval_octa.yaml), [SMILE-UHURA](./configs/data/eval_smile.yaml)) config files (see `#TODO`) and run:

    python vesselfm/seg/finetune.py data=<eval_smile/eval_octa/eval_msd8/eval_bvem> num_shots=<0/1/3>
