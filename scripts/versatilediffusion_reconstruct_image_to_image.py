#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Diffusion from VDVAE round-trip images + ORIGINAL CLIP embeddings
# - Init latent: AutoKL( image_from /results/vdvae/image_to_image/{i}.png )
# - Conditioning: original CLIP vision/text embeddings (npys you extracted)
# - Output: /home/rothermm/brain-diffuser/results/versatile_diffusion/subjXX/original_latents/

import sys
sys.path.append('/home/rothermm/brain-diffuser/versatile_diffusion')

import os
import os.path as osp
from pathlib import Path
import argparse
import numpy as np
import PIL
from PIL import Image

import torch
import torchvision.transforms as tvtrans

from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust  # not used but kept for parity
from lib.model_zoo.vd import VD

# ---------------- CLI ----------------
def get_args():
    p = argparse.ArgumentParser("VD img->img using VDVAE round-trip inputs + ORIGINAL CLIP cond")
    p.add_argument("-sub", "--sub", type=int, default=1, choices=[1,2,5,7],
                   help="Subject (for output + CLIP file paths)")
    p.add_argument("--i2i-dir", type=str,
                   default="/home/rothermm/brain-diffuser/results/vdvae/image_to_image",
                   help="Directory with VDVAE round-trip PNGs named 0.png,1.png,...")
    p.add_argument("--clip-vision-npy", type=str,
                   default="/home/rothermm/brain-diffuser/data/extracted_features/subj01/nsd_clipvision_test.npy",
                   help="Original CLIP vision embeddings (N, 257, 768)")
    p.add_argument("--clip-text-npy", type=str,
                   default="/home/rothermm/brain-diffuser/data/extracted_features/subj01/nsd_cliptext_test.npy",
                   help="Original CLIP text embeddings (N, 77, 768)")
    p.add_argument("--outdir-root", type=str,
                   default="/home/rothermm/brain-diffuser/results/versatile_diffusion",
                   help="Root output directory")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--scale", type=float, default=20.0, help="CFG scale")
    p.add_argument("--strength", type=float, default=0.50, help="DDIM t_start fraction [0,1]")
    p.add_argument("--mixing", type=float, default=0.20,
                   help="Vision/text mixing; mixed_ratio = (1 - mixing) for vision stream")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="If >0, process only this many indices")
    return p.parse_args()

# ---------------- Utils ----------------
def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, PIL.Image.Image):
        x = x.resize((512, 512), resample=BICUBIC)
        x = tvtrans.ToTensor()(x)  # [3,512,512], 0..1
    else:
        raise ValueError("Expected PIL.Image.Image")
    assert x.shape[-2:] == (512, 512), "Wrong image size after resize"
    return x

# ---------------- Main ----------------
@torch.no_grad()
def main():
    args = get_args()
    torch.manual_seed(args.seed)

    # --------- Load model ---------
    cfgm_name = "vd_noema"
    ckpt = "/home/rothermm/brain-diffuser/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth"
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.clip = net.clip.to(device)
    net.autokl = net.autokl.to(device).half()

    sampler = DDIMSampler_VD(net)
    sampler.model.model.diffusion_model.to(device).half()

    # --------- Load ORIGINAL CLIP inputs ---------
    clip_vision = np.load(args.clip_vision_npy)  # (N, 257, 768)
    clip_text   = np.load(args.clip_text_npy)    # (N, 77, 768)

    N = min(len(clip_vision), len(clip_text))
    if args.limit > 0:
        N = min(N, args.limit)

    # --------- Prepare output ---------
    outdir = osp.join(args.outdir_root, f"subj{args.sub:02d}", "original_latents")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(device.index)}")
    print(f"i2i input dir: {args.i2i-dir if hasattr(args, 'i2i-dir') else args.i2i_dir}")  # guard for IDEs
    print(f"Vision: {args.clip_vision_npy} | Text: {args.clip_text_npy}")
    print(f"Output -> {outdir}")
    print(f"DDIM steps={args.ddim_steps}, strength={args.strength}, scale={args.scale}, mixing={args.mixing}")
    print(f"N (from embeddings) = {N}")

    # --------- Schedule ---------
    ddim_steps = args.ddim_steps
    ddim_eta = 0.0
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    t_enc = int(np.clip(args.strength, 0.0, 1.0) * ddim_steps)

    # Unconditional text/vision (empty prompt, black image)
    utx = net.clip_encode_text("").to(device).half()                              # (1,77,768)
    uim = net.clip_encode_vision(torch.zeros((1,3,224,224), device=device)).half()# (1,257,768)

    # --------- Loop ---------
    for i in range(N):
        # --- init latent from VDVAE round-trip image ---
        img_path = osp.join(args.i2i_dir, f"{i}.png")
        if not osp.exists(img_path):
            raise FileNotFoundError(f"Missing input image: {img_path}")
        zim = Image.open(img_path).convert("RGB")
        xim = regularize_image(zim).to(device)             # [3,512,512], 0..1
        zin = (xim * 2.0 - 1.0).unsqueeze(0).half()        # [1,3,512,512] in [-1,1]
        init_latent = net.autokl_encode(zin)               # [1,4,64,64]

        # --- encode to noisy latent at t_enc ---
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc], device=device))

        # --- ORIGINAL CLIP conditioning for this index ---
        cim = torch.from_numpy(clip_vision[i]).unsqueeze(0).to(device).half()  # [1,257,768]
        ctx = torch.from_numpy(clip_text[i]).unsqueeze(0).to(device).half()    # [1,77,768]

        # --- diffusion decode ---
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, cim],      # vision (uncond, cond)
            second_conditioning=[utx, ctx],     # text   (uncond, cond)
            t_start=t_enc,
            unconditional_guidance_scale=args.scale,
            xtype="image",
            first_ctype="vision",
            second_ctype="prompt",
            mixed_ratio=(1.0 - args.mixing),    # e.g., mixing=0.2 => 0.8 vision / 0.2 text
        ).to(device).half()

        # --- decode back to image, clamp and save ---
        x = net.autokl_decode(z)
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        Image.fromarray(tvtrans.ToPILImage()(x[0].detach().cpu()).convert("RGB")).save(
            osp.join(outdir, f"{i}.png")
        )

    print("Done.")

if __name__ == "__main__":
    main()
