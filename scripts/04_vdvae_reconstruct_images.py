import sys
sys.path.append('/home/rothermm/brain-diffuser/vdvae')

import torch
import numpy as np
import argparse
import os
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger, local_mpi_rank, mpi_size, maybe_download, mpi_rank
from data import mkdir_p
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

# -------------------- Parse arguments --------------------
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
parser.add_argument("-bs", "--bs", help="Batch Size", default=30)
args = parser.parse_args()
sub = int(args.sub)
assert sub in [1, 2, 5, 7]
batch_size = int(args.bs)

print('Libs imported')

# -------------------- Model config --------------------
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
    'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25,
    'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100,
    'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015,
    'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4,
    'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0,
    'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000,
    'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None,
    'epochs_per_eval_save': 1, 'num_images_visualize': 8,
    'num_variables_visualize': 6, 'num_temperatures_visualize': 3,
    'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

H = dotdict(H)
H, preprocess_fn = set_up_data(H)

print('Model is loading...')
ema_vae = load_vaes(H)

# -------------------- Dataset Loader --------------------
class batch_generator_external_images(Dataset):
    def __init__(self, data_path):
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img, (64, 64))
        img = torch.tensor(np.array(img)).float()
        return img

    def __len__(self):
        return len(self.im)

# -------------------- Load Predicted Latents --------------------
pred_latents_path = f"/home/rothermm/brain-diffuser/data/predicted_features/subj{sub:02d}/nsd_vdvae_nsdgeneral_pred_sub{sub}_31l_alpha50k.npy"
pred_latents = np.load(pred_latents_path)
print(f"Loaded predicted latents: {pred_latents.shape}")

# -------------------- Load ref_latent --------------------
ref_latent_path = f"/home/rothermm/brain-diffuser/data/extracted_features/subj{sub:02d}/ref_latents.npz"
ref_latent = np.load(ref_latent_path, allow_pickle=True)["ref_latent"]
print(f"Loaded ref_latent from: {ref_latent_path}")

# -------------------- Latent Transformation --------------------
def latent_transformation(latents, ref):
    layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,
    2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,
    2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
    2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
    2**12,2**14])
    transformed_latents = []
    for i in range(31):
        t_lat = latents[:, layer_dims[:i].sum():layer_dims[:i+1].sum()]
        c, h, w = ref[i]['z'].shape[1:]
        transformed_latents.append(t_lat.reshape(len(latents), c, h, w))
    return transformed_latents

input_latent = latent_transformation(pred_latents, ref_latent)

# -------------------- Sampling Function --------------------
def sample_from_hier_latents(latents, sample_ids):
    sample_ids = [i for i in sample_ids if i < len(latents[0])]
    sample_latents = [torch.tensor(latents[i][sample_ids]).float().cuda() for i in range(len(latents))]
    return sample_latents

# -------------------- Image Generation --------------------
output_dir = f"/home/rothermm/brain-diffuser/results/vdvae/subj{sub:02d}"
os.makedirs(output_dir, exist_ok=True)

num_samples = len(pred_latents)
num_batches = int(np.ceil(num_samples / batch_size))

print(f"Generating {num_samples} images in {num_batches} batches...")
for i in range(num_batches):
    print(f"Batch {i+1}/{num_batches}")
    batch_ids = range(i * batch_size, min((i + 1) * batch_size, num_samples))
    samp = sample_from_hier_latents(input_latent, batch_ids)
    with torch.no_grad():
        px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
        generated = ema_vae.decoder.out_net.sample(px_z)
    for j, img_arr in enumerate(generated):
        img = Image.fromarray(img_arr)
        img = img.resize((512, 512), resample=3)
        img.save(os.path.join(output_dir, f"{i * batch_size + j}.png"))

print(f"Image generation complete. Saved to: {output_dir}")
