#!/usr/bin/env python3
import sys
sys.path.append('/home/rothermm/brain-diffuser/vdvae')

import torch
import numpy as np
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle

# --- Parse arguments ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
parser.add_argument("-bs", "--bs",  help="Batch Size",     default=30)
args = parser.parse_args()
sub = int(args.sub)
assert sub in [1, 2, 5, 7]
batch_size = int(args.bs)

print('Libs imported')

# --- Model & hyperparameters setup --------------------------------------------
model_dir = '/home/rothermm/brain-diffuser/vdvae/model'

H = {
    'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500,
    'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test',
    'hparam_sets': 'imagenet64',
    'restore_path': f'{model_dir}/imagenet64-iter-1600000-model.th',
    'restore_ema_path': f'{model_dir}/imagenet64-iter-1600000-model-ema.th',
    'restore_log_path': f'{model_dir}/imagenet64-iter-1600000-log.jsonl',
    'restore_optimizer_path': f'{model_dir}/imagenet64-iter-1600000-opt.th',
    'dataset': 'imagenet64', 'ema_rate': 0.999,
    'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
    'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12',
    'zdim': 16, 'width': 512, 'custom_width_str': '',
    'bottleneck_multiple': 0.25, 'no_bias_above': 64,
    'scale_encblock': False, 'test_eval': True,
    'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0,
    'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015,
    'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4,
    'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0,
    'iters_per_ckpt': 25000, 'iters_per_print': 1000,
    'iters_per_save': 10000, 'iters_per_images': 10000,
    'epochs_per_eval': 1, 'epochs_per_probe': None,
    'epochs_per_eval_save': 1, 'num_images_visualize': 8,
    'num_variables_visualize': 6, 'num_temperatures_visualize': 3,
    'mpi_size': 1, 'local_rank': 0, 'rank': 0,
    'logdir': './saved_models/test/log'
}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

# This returns (possibly updated H, preprocess_fn)
H, preprocess_fn = set_up_data(H)

print('Model is Loading')
ema_vae = load_vaes(H)

# --- Dataset class -----------------------------------------------------------
class batch_generator_external_images(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img, (64, 64))
        img = torch.tensor(np.array(img)).float()
        return img

    def __len__(self):
        return len(self.im)

# --- Load test stimuli & collect latents ------------------------------------
image_path = f'/home/rothermm/brain-diffuser/data/processed_data/subj{sub:02d}/nsd_test_stim_sub{sub}.npy'
test_images = batch_generator_external_images(data_path=image_path)
testloader = DataLoader(test_images, batch_size, shuffle=False)

test_latents = []
stats = None
for i, x in enumerate(testloader):
    data_input, target = preprocess_fn(x)
    with torch.no_grad():
        print(i * batch_size)
        activations = ema_vae.encoder(data_input)
        px_z, stats = ema_vae.decoder(activations, get_latents=True)
        batch_latent = []
        for layer_i in range(31):
            # flatten each layers z: shape [batch_size, Ci, Hi, Wi] -> [batch_size, Ci*Hi*Wi]
            layer_flat = stats[layer_i]['z'].cpu().numpy().reshape(len(data_input), -1)
            batch_latent.append(layer_flat)
        test_latents.append(np.hstack(batch_latent))
# After loop, stats refers to the last batchs stats
test_latents = np.concatenate(test_latents)  # shape [N, D_flat]

# --- Load predicted latents & set ref_latent (stats) -----------------------
pred_latents = np.load(
    f'/home/rothermm/brain-diffuser/data/predicted_features/'
    f'subj{sub:02d}/nsd_vdvae_nsdgeneral_pred_sub{sub}_31l_alpha50k.npy'
)
ref_latent = stats  # stats from last batch

# --- Hard-coded latent_transformation as in your original ------------------
def latent_transformation(latents, ref):
    """
    Splits flat latent vectors into 31 hierarchical layers using hard-coded dims.
    ref is the `stats` list from one decoder forward, where each ref[i]['z'].shape = [B, Ci, Hi, Wi].
    """
    layer_dims = np.array([
        2**4,  2**4,
        2**8,  2**8,  2**8,  2**8,
        2**10, 2**10, 2**10, 2**10,
        2**10, 2**10, 2**10, 2**10,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**14
    ])
    transformed_latents = []
    for i in range(31):
        start = layer_dims[:i].sum()
        end = layer_dims[:i+1].sum()
        t_lat = latents[:, start:end]  # shape [N, Ci*Hi*Wi]
        c, h, w = ref[i]['z'].shape[1:]
        transformed_latents.append(t_lat.reshape(len(latents), c, h, w))
    return transformed_latents

# Build the base hierarchical latent list for a = 0 (no shift):
idx = range(len(test_images))
base_hier = latent_transformation(pred_latents[idx], ref_latent)

# --- Wrapper to pick out minibatch layers & send to GPU -------------------
def sample_from_hier_latents(latents, sample_ids):
    """
    Given:
      - latents: list of 31 numpy arrays, each [N, Ci, Hi, Wi]
      - sample_ids: a list of indices to pick from dimension 0
    Returns a list of 31 GPU tensors, each [len(ids), Ci, Hi, Wi].
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_ids = [i for i in sample_ids if i < latents[0].shape[0]]
    layers_num = len(latents)
    sample_latents = []
    for i in range(layers_num):
        subset = latents[i][sample_ids]  # numpy slice
        sample_latents.append(torch.tensor(subset, device=device).float())
    return sample_latents

# --- Load ?-vectors for both assessors --------------------------------------
THETA_DIR = f'/home/rothermm/brain-diffuser/results/thetas/subj{sub:02d}'
theta_emo = np.load(os.path.join(THETA_DIR, f'theta_emonet_nsdgeneral_subject{sub}.npy'))
theta_mem = np.load(os.path.join(THETA_DIR, f'theta_memnet_nsdgeneral_subject{sub}.npy'))

# --- Define a values and loop over assessors & a ----------------------------
alphas = [-4, -3, -2, 0, 2, 3, 4]
for name, theta in [('emonet', theta_emo), ('memnet', theta_mem)]:
    for alpha in alphas:
        # If a = 0, we reuse the base hierarchy exactly; otherwise shift flat and re-transform
        if alpha == 0:
            hier = base_hier
        else:
            mod_flat = pred_latents + alpha * theta  # shape [N, D_flat]
            hier = latent_transformation(mod_flat[idx], ref_latent)

        # Create output dir: .../vdvae/subjXX/{name}/alpha_{alpha}/
        OUT_DIR = f'/home/rothermm/brain-diffuser/results/vdvae/subj{sub:02d}/{name}/alpha_{alpha}'
        os.makedirs(OUT_DIR, exist_ok=True)

        # Minibatch decode
        N = pred_latents.shape[0]
        for i in range(int(np.ceil(N / batch_size))):
            start = i * batch_size
            end = min(start + batch_size, N)
            ids = list(range(start, end))
            print(f"{name} a={alpha}: processing indices {start}{end-1}")
            samp = sample_from_hier_latents(hier, ids)
            with torch.no_grad():
                px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
                sample_imgs = ema_vae.decoder.out_net.sample(px_z)
            # sample_imgs is a list of numpy arrays [H, W, 3] uint8
            for j, im in enumerate(sample_imgs):
                pil = Image.fromarray(im)
                pil = pil.resize((512, 512), resample=Image.BILINEAR)
                fname = f'{start + j}.png'
                pil.save(os.path.join(OUT_DIR, fname))

        print(f'Done reconstructing {name} at a={alpha} for subj{sub}')

print(f'All done for subject {sub}. Output at /home/rothermm/brain-diffuser/results/vdvae/subj{sub:02d}')
