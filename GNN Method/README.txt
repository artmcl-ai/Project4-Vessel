This directory contains everything needed to set-up the GNN method for non-contrast artery and vein vessel segmentation.

Graph_Generator.py is responsible for generating the graphs from the binary masks produced from the finetune VesselFM model. The data is not stored in this repository, the binary masks folder should be set to the 'input_path' variable and you need to set the 'output_mask_path' variable to the folder that will house the graphs.
