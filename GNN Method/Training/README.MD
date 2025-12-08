
This repository contains everything needed to set up the Graph Neural Network (GNN) method for non-contrast artery and vein vessel segmentation.

## 📁 Binarize GT Masks

The script `gt_to_binary_masks.py` convert the GT (Ground Truth) Masks into a binary mask that does not have a vein or artery class. The reason this is
needed is because to finetune VesselFM the GT data must be binary as VesselFM does segmentation of vessels, but not the classification of arteries and veins.

### Required Path Setup
In `gt_to_binary_masks.py`, update the following variables:

| Variable | Description |
|---------|-------------|
| `in_path  ` | Path to the folder containing your GT masks (multiclass) |
| `out_path   ` | Path to the folder where you want your GT binary masks |

---

## Usage

```bash
python gt_to_binary_masks.py
```
***
***
***

## 📁 Finetune VesselFM

The script `finetune_vesselfm.py` finetunes the base VesselFM which can be found here: (https://huggingface.co/bwittmann/vesselFM).

### Required Path Setup
In `finetune_vesselfm.py`, update the following variables:

| Variable | Description |
|---------|-------------|
| `pretrained_ckpt ` | Path to the base VesselFM model|
| `image_dir ` | Path to the folder containing your CT images |
| `mask_dir  ` | Path to the folder containing your GT binary masks |

---

## Usage

```bash
python finetune_vesselfm.py
```
***
***
***

## 📁 Binary Mask Prediction

The script `generate_binary_segmentation_masks.py` uses the finetune VesselFM model (not included in this repository) to predict binary
segmentation masks for given input CT images

### Required Path Setup
In `generate_binary_segmentation_masks.py`, update the following variables:

| Variable | Description |
|---------|-------------|
| `model_path ` | Path to the finetuned VesselFM model|
| `input_image_path` | Path to the folder containing your CT images |
| `output_mask_path` | Path to the folder where you want to store binary masks |

---

## Usage

```bash
python generate_binary_segmentation_masks.py
```
***
***
***

## 📁 Graph Generation

The script `Graph_Generator.py` handles graph construction from the binary masks produced by a fine-tuned **VesselFM** model.  
The data itself is not included in this repository, so you must configure paths before running.

### Required Path Setup
In `Graph_Generator.py`, update the following variables:

| Variable | Description |
|---------|-------------|
| `input_path` | Path to the folder containing your VesselFM-generated binary masks |
| `output_mask_path` | Folder where the graph outputs will be saved |

---

## Usage

```bash
python Graph_Generator.py
```
***
***
***
## 📁 Train and Define GNN

The script `GNN_training.py` defines the GAT model used to classify edges as either veins or arteries. The model decleration and training is included
in this file

## Usage

```bash
python GNN_training.py
