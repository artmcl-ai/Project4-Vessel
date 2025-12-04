
This repository contains everything needed to set up the Graph Neural Network (GNN) method for non-contrast artery and vein vessel segmentation.


## 📁 Binary Mask Creation

The script `generate_binary_segmentation_masks.py` uses the finetune VesselFM model (not included in this repository) to generate binary
segmentation masks for input CT images

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
