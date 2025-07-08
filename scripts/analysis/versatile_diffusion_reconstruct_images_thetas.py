#!/usr/bin/env python3
import sys
sys.path.append('/home/rothermm/brain-diffuser/versatile_diffusion')
import os

# Absolute path to your project root
BASE_DIR = "/home/rothermm/brain-diffuser"
os.chdir(BASE_DIR)

import os.path as osp
import PIL
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset
import argparse

# --- Command-line arguments -------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Versatile Diffusion on all VDVAE-reconstructed PNGs "
                "(for each assessor & a) for a given subject"
)
parser.add_argument(
    "-sub","--sub",
    type=int,
    choices=[1,2,5,7],
    required=True,
    help="Subject number"
)
parser.add_argument(
    "-diff_str","--diff_str",
    type=float,
    default=0.75,
    help="Diffusion strength (0.01.0)"
)
parser.add_argument(
    "-mix_str","--mix_str",
    type=float,
    default=0.4,
    help="Mixing ratio (0.01.0) between vision/text conditioning"
)
args = parser.parse_args()
sub = args.sub
strength = args.diff_str
mixing = args.mix_str

assert 0.0 <= strength <= 1.0, "diff_str must be in [0.0, 1.0]"
assert 0.0 <= mixing <= 1.0, "mix_str must be in [0.0, 1.0]"

# --- Device setup -------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} - "
      f"{torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'}")

# --- Versatile Diffusion model loading ----------------------------------------
def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    # Accepts path, PIL.Image, NumPy array, or Torch tensor
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        # assume already: [3, 512, 512], 01 float
        pass
    else:
        raise AssertionError("Unknown image type")

    assert (x.shape[1] == 512) and (x.shape[2] == 512), "Wrong image size"
    return x

cfgm_name = 'vd_noema'
sampler_class = DDIMSampler_VD
ckpt_path = '/home/rothermm/brain-diffuser/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'

cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(ckpt_path, map_location='cpu')
net.load_state_dict(sd, strict=False)

# Move the CLIP+autoKL subnets to GPU/CPU
net.clip = net.clip.to(device)
net.autokl = net.autokl.to(device)

sampler = sampler_class(net)
batch_size = 1  # process one image at a time

# --- Load predicted CLIP latents once ----------------------------------------
pred_text = np.load(
    f'/home/rothermm/brain-diffuser/data/predicted_features/'
    f'subj{sub:02d}/nsd_cliptext_predtest_nsdgeneral.npy'
)
pred_text = torch.from_numpy(pred_text).half().to(device)

pred_vision = np.load(
    f'/home/rothermm/brain-diffuser/data/predicted_features/'
    f'subj{sub:02d}/nsd_clipvision_predtest_nsdgeneral.npy'
)
pred_vision = torch.from_numpy(pred_vision).half().to(device)

n_samples = 1
ddim_steps = 50
ddim_eta = 0.0
scale = 7.5   # unconditional guidance scale
xtype = 'image'
ctype = 'prompt'
net.autokl.half()

torch.manual_seed(0)  # for reproducibility

# --- Main loop: for each assessor, each a, process all VDVAE images ----------
assessors = ['emonet', 'memnet']
alphas = [-4, -2, 0, 2, 4]

for assessor_name in assessors:
    for alpha in alphas:
        # Input directory of VDVAE-reconstructions:
        in_dir = (f'/home/rothermm/brain-diffuser/results/vdvae/'
                  f'subj{sub:02d}/{assessor_name}/alpha_{alpha}')
        if not os.path.isdir(in_dir):
            print(f"[WARNING] Input directory does not exist: {in_dir}")
            continue

        # Output directory under versatile_diffusion:
        out_dir = (f'/home/rothermm/brain-diffuser/results/versatile_diffusion/'
                   f'subj{sub:02d}/{assessor_name}/alpha_{alpha}')
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nProcessing folder:\n  {in_dir}\n? Outputs to:\n  {out_dir}\n")

        # For each PNG in sorted order:
        png_files = [f for f in os.listdir(in_dir)
                     if f.lower().endswith('.png')]
        png_files.sort()

        for idx, fn in enumerate(png_files):
            img_path = os.path.join(in_dir, fn)
            # 1) Load & preprocess the 512512 image
            zim = Image.open(img_path).convert('RGB')
            zim = regularize_image(zim)            # [3,512,512], float in [0,1]
            zin = (zim * 2.0 - 1.0).unsqueeze(0).to(device).half()

            # 2) Encode to autoKL latent
            with torch.no_grad():
                init_latent = net.autokl_encode(zin)

            # 3) Prepare DDIM schedule & stochastic encode
            sampler.make_schedule(
                ddim_num_steps=ddim_steps,
                ddim_eta=ddim_eta,
                verbose=False
            )
            t_enc = int(strength * ddim_steps)
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))

            # 4) Prepare empty vs actual CLIP conditioning
            dummy_text = ""
            utx = net.clip_encode_text(dummy_text).to(device).half()

            dummy_image = torch.zeros((1, 3, 224, 224), device=device)
            uim = net.clip_encode_vision(dummy_image).to(device).half()

            # 5) Grab the predicted CLIP latents for this index
            cim = pred_vision[idx].unsqueeze(0)  # shape [1, D_vision]
            ctx = pred_text[idx].unsqueeze(0)    # shape [1, D_text]

            # 6) Move noisy latent to correct device
            z_enc = z_enc.to(device)

            # 7) Ensure the diffusion_model itself is on GPU and half
            sampler.model.model.diffusion_model = (
                sampler.model.model.diffusion_model.half().to(device)
            )
            sampler.model.model.diffusion_model.device = device

            # 8) Perform double-conditioning decode
            with torch.no_grad():
                z = sampler.decode_dc(
                    x_latent=z_enc,
                    first_conditioning=[uim, cim],
                    second_conditioning=[utx, ctx],
                    t_start=t_enc,
                    unconditional_guidance_scale=scale,
                    xtype='image',
                    first_ctype='vision',
                    second_ctype='prompt',
                    mixed_ratio=(1 - mixing),
                )

            # 9) Decode back to pixel space
            z = z.to(device).half()
            x = net.autokl_decode(z)

            # 10) Clamp and convert to PIL
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            pil_images = [tvtrans.ToPILImage()(xi) for xi in x]

            # 11) Save (each sample batch_size=1, so just pil_images[0])
            out_path = os.path.join(out_dir, fn)
            pil_images[0].save(out_path)

            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx+1}/{len(png_files)} images.")

        print(f"Finished {assessor_name} @ a={alpha}")

print("All done! Versatile Diffusion outputs under:")
print(f"  /home/rothermm/brain-diffuser/results/versatile_diffusion/subj{sub:02d}/")
