"""
Adapted from:
https://github.com/mobaidoctor/med-ddpm/blob/main/diffusion_model/trainer.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
"""

import copy
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# small helper modules
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5


class FlowMatching(torch.nn.Module):
    """x0: noise, x1: data"""
    def __init__(
        self,
        model,
        use_lognorm_is=True,  # lognorm time importance sampling from SD3
        lognorm_mu=0.0,
        lognorm_sigma=1.0,
        clip_denoised=False,
        num_steps=100,
        use_cfg=False,
        cfg_weight=3,
        cfg_bg=False,
        cfg_rand=False,
        interpolate=False,
        num_classes=None
    ):
        super().__init__()
        self.model = model
        self.use_lognorm_is = use_lognorm_is
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma
        self.clip_denoised = clip_denoised
        self.num_steps = num_steps
        self.use_cfg = use_cfg
        self.cfg_weight = cfg_weight
        self.cfg_bg = cfg_bg
        self.cfg_rand = cfg_rand
        self.interpolate = interpolate
        self.num_classes = num_classes

    def forward(self, data_samples, condition_tensors, y):
        """Compute loss for training."""
        x_1 = data_samples
        x_0 = torch.randn_like(x_1)
        v_gt = x_1 - x_0  # gt velocity is time-constant given x0 and x1

        if self.use_lognorm_is:
            # Apply lognorm SD3 importance sampling here
            t = torch.randn(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
            t = t * self.lognorm_sigma + self.lognorm_mu
            t = torch.sigmoid(t)
        else:
            t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)

        t_unsq = t.reshape(t.shape[0], *((1,) * (x_1.ndim - 1)))
        x_t = torch.lerp(x_0, x_1, t_unsq)

        v_pred = self.model(torch.cat([x_t, condition_tensors], 1), t, y=y)
        loss_unreduced = F.mse_loss(v_pred, v_gt, reduction="none")
        loss_unreduced = loss_unreduced.flatten(1).mean(1)
        loss = loss_unreduced.mean()
        return loss

    @torch.inference_mode()
    def sample(self, condition_tensors, y, reverse_time=False): 
        """Generation: discretize the ODE via Euler integration."""
        noise = torch.randn_like(condition_tensors)
        num_samples = noise.shape[0]

        x_t = noise
        traj = [x_t.detach().clone()]
        dt = 1.0 / self.num_steps
        times = [i / self.num_steps for i in range(self.num_steps)]

        if reverse_time:
            dt *= -1
            times = [1.0 - t for t in times]

        # estimate cfg weights
        if self.cfg_rand:
            cfg_weight = np.random.uniform(0, self.cfg_weight)
            print(cfg_weight)
        else:
            cfg_weight = self.cfg_weight

        # randomly sample second classes and interpolation factor
        if self.interpolate:
            while True:
                y2 = torch.randint_like(y, 0, self.num_classes)
                if y2 != y:
                    break
            w = torch.rand(1).item()
            print(f"Interpolation mode: y1 {y.item()}; y2 {y2.item()}; w {w}")

        for t in tqdm(times, desc='sampling loop time step', total=len(times)):
            t_tnsr = torch.full([num_samples], t, dtype=x_t.dtype, device=x_t.device)

            if self.use_cfg:    # sample in classifier-free guidance mode
                v_cond = self.model(torch.cat([x_t, condition_tensors], 1), t_tnsr, y=y)
                v_uncond = self.model(torch.cat([x_t, condition_tensors], 1), t_tnsr, y=torch.ones_like(y) * (-1))

                if self.cfg_bg: # cfg solely for bg voxels
                    v = torch.zeros_like(v_cond)
                    v[condition_tensors == 1] = v_cond[condition_tensors == 1]
                    v[condition_tensors == 0] = (1 + cfg_weight) * v_cond[condition_tensors == 0] - cfg_weight * v_uncond[condition_tensors == 0]
                else:
                    v = (1 + cfg_weight) * v_cond - cfg_weight * v_uncond

            elif self.interpolate:  # sample in interpolation mode
                # v_y1 = self.model(torch.cat([x_t, condition_tensors], 1), t_tnsr, y=y)    # v interpolation
                # v_y2 = self.model(torch.cat([x_t, condition_tensors], 1), t_tnsr, y=y2)
                # v = w * v_y1 + (1- w) * v_y2

                v = self.model(torch.cat([x_t, condition_tensors], 1), t_tnsr, y=y, y2=y2, w=w) # embedding interpolation

            else:   # sample regularly
                v = self.model(torch.cat([x_t, condition_tensors], 1), t_tnsr, y=y)

            x_t = x_t + (v * dt)

            if self.clip_denoised:
                x_t.clamp_(-1., 1.)
            traj.append(x_t.detach().clone())

        traj = torch.stack(traj, 1)  # num-shapes x num-timepoints x d
        return unnormalize_to_zero_to_one(x_t.clamp_(-1., 1.)) # [-1, 1] -> [0, 1]


# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataloader,
        ema_decay=0.995,
        train_batch_size=2,
        train_lr=2e-6,
        epochs=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results_diff',
        device=None,
        class_cond=False,
        use_cfg=False,
        cfg_p_drop=0.2
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.epochs = epochs
        self.device = device
        self.class_cond = class_cond
        self.cfg = use_cfg
        self.cfg_p_drop = cfg_p_drop

        self.dl = dataloader
        self.opt = Adam(diffusion_model.parameters(), lr=float(train_lr))
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size

        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.log_dir = self.results_folder
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        start_time = time.time()
        accumulated_loss = []
        for epoch in range(self.epochs):
            for itera, batch in enumerate(self.dl):
                image, mask, class_id = batch
                image, mask, class_id = image.to(device=self.device), mask.to(device=self.device), class_id.to(device=self.device)

                if not self.class_cond:
                    class_id = None
                
                # classifier free guidance
                if self.cfg:
                    class_id[torch.rand(self.batch_size) < self.cfg_p_drop] = -1

                loss = self.model(image, condition_tensors=mask, y=class_id)
                loss.backward()

                print(f'e{epoch}, i{itera}: {loss.item()}')
                accumulated_loss.append(loss.item())

                average_loss = np.mean(accumulated_loss)
                end_time = time.time()
                self.writer.add_scalar("training_loss", average_loss, self.step)

                self.opt.step()
                self.opt.zero_grad()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every

                    # sample an image
                    images = self.ema_model.sample(condition_tensors=mask, y=class_id)

                    for idx, (i, m, ci) in enumerate(zip(images, mask, class_id)):  # TODO: breaks if no class_cond / class_id == None
                        nifti_img = nib.Nifti1Image(i.cpu().numpy().squeeze(), affine=np.eye(4))
                        nib.save(nifti_img, str(self.results_folder / f'sample-{milestone}-i-{idx}-class-{ci}.nii.gz'))
                        nifti_mask = nib.Nifti1Image(m.cpu().numpy().squeeze(), affine=np.eye(4))
                        nib.save(nifti_mask, str(self.results_folder / f'sample-{milestone}-m-{idx}-class-{ci}.nii.gz'))

                    if milestone % 5 == 0:  # save only ckpts for every 5th milestone
                        self.save(milestone)

                self.step += 1

        print('training completed')
        end_time = time.time()
        execution_time = (end_time - start_time)/3600
        self.writer.add_hparams(
            {
                "lr": self.train_lr,
                "batchsize": self.train_batch_size,
                "execution_time (hour)":execution_time
            },
            {"last_loss":average_loss}
        )
        self.writer.close()
