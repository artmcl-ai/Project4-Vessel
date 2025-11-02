#Get Dice Loss between VesselFM prediction and binary truth
from monai.losses import DiceLoss
import nibabel as nib
import numpy as np
import torch

true_path = '/content/binary_case_001.nii.gz'
pred_path = '/content/image_001_pred (1).nii.gz'
true_img = nib.load(true_path)
pred_img = nib.load(pred_path)
true_arr = true_img.get_fdata().astype(np.int64)
pred_arr = pred_img.get_fdata().astype(np.float32)
true_t = torch.from_numpy(true_arr).unsqueeze(0).unsqueeze(0).to(torch.float32)
pred_t = torch.from_numpy(pred_arr).unsqueeze(0).unsqueeze(0).to(torch.float32)
dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
dice_loss2 = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
loss = dice_loss(pred_t, true_t)
loss2 = dice_loss2(pred_t, true_t)
print(loss.item())
print(loss2.item())