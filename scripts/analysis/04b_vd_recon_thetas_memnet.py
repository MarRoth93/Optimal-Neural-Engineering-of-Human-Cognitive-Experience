print("🚀 Script loaded", flush=True)

def main():
    print("✅ main() function entered", flush=True)

if __name__ == "__main__":
    print("🧪 Running __main__", flush=True)
    main()


import sys
import os
import os.path as osp
from pathlib import Path
import argparse
import PIL
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tvtrans
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize, downscale_local_mean
import matplotlib.pyplot as plt

sys.path.append('/home/rothermm/brain-diffuser/versatile_diffusion')
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh


def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
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
        pass
    else:
        raise TypeError('Unknown image type')

    assert (x.shape[1]==512) and (x.shape[2]==512), 'Wrong image size'
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--sub", type=int, choices=[1, 2, 5, 7], default=1, help="Subject Number")
    args = parser.parse_args()
    sub = args.sub

    print(f"🚀 Starting batch processing for subject {sub}")

    # Absolute path to your project root
    BASE_DIR = "/home/rothermm/brain-diffuser"
    os.chdir(BASE_DIR)
    cfgm_name = 'vd_noema'
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'

    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net.clip.to(device)
    net.autokl.to(device)
    net.autokl.half()
    sampler = DDIMSampler_VD(net)

    #pred_text = np.load(f'data/predicted_features/subj{sub:02d}/nsd_cliptext_predtest_nsdgeneral.npy')
    #pred_vision = np.load(f'data/predicted_features/subj{sub:02d}/nsd_clipvision_predtest_nsdgeneral.npy')


    # Dynamically load predicted features based on subject
    pred_text_path   = f'/home/rothermm/brain-diffuser/data/extracted_features/subj{sub:02d}/nsd_cliptext_test.npy'
    pred_vision_path = f'/home/rothermm/brain-diffuser/data/extracted_features/subj{sub:02d}/nsd_clipvision_test.npy'
    print(f"📂 Loading features for subject {sub}:")
    print(f"    text features:   {pred_text_path}")
    print(f"    vision features: {pred_vision_path}")
    pred_text   = np.load(pred_text_path)
    pred_vision = np.load(pred_vision_path)
    
    try:
        pred_text = torch.tensor(pred_text).half().to(device)
        pred_vision = torch.tensor(pred_vision).half().to(device)
    except RuntimeError as e:
        print(f"Error moving tensors to GPU: {e}")
        device = torch.device("cpu")
        pred_text = torch.tensor(pred_text).half()
        pred_vision = torch.tensor(pred_vision).half()


    mixing_var = 0.2
    strength_var = 0.5
    scale = 20
    ddim_steps = 50

    torch.manual_seed(0)

    # List all the “alpha_*” subfolders under your VD-VAE dir
    base_in = Path(f'/home/rothermm/brain-diffuser/results/vdvae/subj{sub:02d}/memnet')
    base_out = Path(f'/home/rothermm/brain-diffuser/results/versatile_diffusion/subj{sub:02d}/memnet')

    # Automatically grab only those folders starting with “alpha_”
    alphas = sorted(p.name for p in base_in.iterdir() if p.is_dir() and p.name.startswith('alpha_'))

    print("✔️ Found alpha folders:", alphas)

    for alpha in alphas:
        input_dir  = base_in  / alpha
        output_dir = base_out / alpha
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔄 Processing folder '{alpha}'")
        print(f"   Input : {input_dir}")
        print(f"   Output: {output_dir}")

        total_imgs = len(pred_vision)
        print(f"   Number of images to process: {total_imgs}")

        for im_id in range(len(pred_vision)):
            zim = Image.open(input_dir / f'{im_id:06d}.png')
            zim = regularize_image(zim)
            zin = zim * 2 - 1
            zin = zin.unsqueeze(0).to(device).half()

            init_latent = net.autokl_encode(zin)
            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose=False)

            t_enc = int(strength_var * ddim_steps)
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))

            utx = net.clip_encode_text('').to(device).half()
            uim = net.clip_encode_vision(torch.zeros((1, 3, 224, 224)).to(device)).half()

            cim = pred_vision[im_id].unsqueeze(0).to(device)
            ctx = pred_text[im_id].unsqueeze(0).to(device)

            sampler.model.model.diffusion_model.device = device
            sampler.model.model.diffusion_model.half().to(device)

            z = sampler.decode_dc(
                x_latent=z_enc,
                first_conditioning=[uim, cim],
                second_conditioning=[utx, ctx],
                t_start=t_enc,
                unconditional_guidance_scale=scale,
                xtype='image',
                first_ctype='vision',
                second_ctype='prompt',
                mixed_ratio=(1 - mixing_var),
            )

            x = net.autokl_decode(z.to(device).half())
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = [tvtrans.ToPILImage()(xi) for xi in x]

            save_path = output_dir / f'{im_id}.png'
            Path(osp.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            x[0].save(save_path)
            print(f"Saved image: {save_path}")
        
        print(f"✅ Finished folder '{alpha}'")
    print("🎉 All folders processed.")


if __name__ == "__main__":
    main()
