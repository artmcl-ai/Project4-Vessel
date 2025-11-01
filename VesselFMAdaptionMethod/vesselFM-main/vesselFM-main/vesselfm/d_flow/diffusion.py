"""Adapted from https://github.com/mobaidoctor/med-ddpm/blob/main/train.py"""

import json
import sys
import socket
import subprocess
from pathlib import Path

import yaml
import torch

from vesselfm.d_flow.data import build_loader
from vesselfm.d_flow.diffusion_unet import create_model
from vesselfm.d_flow.trainer import FlowMatching, Trainer


def get_meta_data():
    meta_data = {}
    meta_data['git_commit_hash'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    meta_data['python_version'] = sys.version.splitlines()[0]
    meta_data['gcc_version'] = sys.version.splitlines()[1]
    meta_data['pytorch_version'] = torch.__version__
    meta_data['host_name'] = socket.gethostname()
    return meta_data

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=3)

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def load_config(dict=True, name='config'):
    with open(f'./vesselfm/d_flow/{name}.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    if dict:
        return config
    else:
        return Config(config)


def train(config):
    # log config
    results_folder = Path(f'./runs/{config.DATA_GEN.DIFF.TAG}')
    results_folder.mkdir(exist_ok=True, parents=True)
    config_log = load_config(dict=True)
    config_log.update(get_meta_data())
    write_json(config_log, results_folder / 'config.json')

    # load model
    device = torch.device(f"cuda:{config.DATA_GEN.DIFF.GPU_ID}" if torch.cuda.is_available() else "cpu")
    classes = config_log['DATA_GEN']['DIFF']['CLASSES']
    model = create_model(
        image_size=config.DATA.IMG_SIZE, num_channels=config.DATA_GEN.DIFF.NUM_CHANNELS,
        num_res_blocks=config.DATA_GEN.DIFF.LAYERS, in_channels=2, out_channels=1,
        class_cond=config.DATA_GEN.DIFF.CLASS_COND, use_cfg=config.DATA_GEN.DIFF.CFG,
        num_classes=torch.tensor([v[0] for v in classes.values()]).max().item() + 1
    )
    model.to(device=device)
    model.train()

    # generate dataloader
    dataloader = build_loader(config, classes)

    diffusion = FlowMatching(
        model, use_lognorm_is=config.DATA_GEN.DIFF.FLOW_MATCHING.USE_LOGNORM_IS, 
        lognorm_mu=config.DATA_GEN.DIFF.FLOW_MATCHING.LOGNORM_MU, 
        lognorm_sigma=config.DATA_GEN.DIFF.FLOW_MATCHING.LOGNORM_SIGMA,
        num_steps=config.DATA_GEN.DIFF.FLOW_MATCHING.TIMESTEPS
    ).to(device=device)

    trainer = Trainer(
        diffusion, dataloader, train_batch_size=config.DATA.BATCH_SIZE, 
        train_lr=config.DATA_GEN.DIFF.LR, epochs=config.DATA_GEN.DIFF.EPOCHS, ema_decay=0.995,
        save_and_sample_every=config.DATA_GEN.DIFF.SAVE_INTERVAL, device=device, 
        results_folder=results_folder, class_cond=config.DATA_GEN.DIFF.CLASS_COND,
        use_cfg=config.DATA_GEN.DIFF.CFG, cfg_p_drop=config.DATA_GEN.DIFF.P_CLS_DROP
    )
    trainer.train()


if __name__ == '__main__':
    config = load_config(dict=False)
    train(config)