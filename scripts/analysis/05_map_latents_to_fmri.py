import os
import pickle
import numpy as np
import nibabel as nib
from nilearn import image
from sklearn.linear_model import Ridge
from pathlib import Path
import logging
import argparse

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

# --- Configuration ---
alpha_values = [-4, 4]  # Manipulations
theta_types = ["emonet", "memnet"]  # Process both theta types
base_path = Path("/home/rothermm/brain-diffuser/data")
output_base_dir = base_path / "brain_mapping"  # NIfTI and contrasts go here

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Map manipulated latents to fMRI and save as NIfTI")
parser.add_argument("--sub", type=int, required=True, help="Subject number (1, 2, 5, or 7)")
args = parser.parse_args()
sub = args.sub

logging.info(f"Processing Subject {sub:02d}")

# Paths
feat_file = base_path / f"extracted_features/subj{sub:02d}/nsd_vdvae_features_31l.npz"
train_fmri_file = base_path / f"processed_data/subj{sub:02d}/nsd_train_fmriavg_nsdgeneral_sub{sub}.npy"
test_fmri_file = base_path / f"processed_data/subj{sub:02d}/nsd_test_fmriavg_nsdgeneral_sub{sub}.npy"
theta_dir = Path(f"/home/rothermm/brain-diffuser/results/thetas/subj{sub:02d}")
mask_file = base_path / f"nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/nsdgeneral.nii.gz"

pred_dir = base_path / f"predicted_fmri_mapping/subj{sub:02d}"
weights_dir = base_path / f"regression_weights/subj{sub:02d}"
subj_output_dir = output_base_dir / f"subj{sub:02d}"
subj_output_dir.mkdir(parents=True, exist_ok=True)

os.makedirs(pred_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# Load data
nsd_features = np.load(feat_file)
train_latents = nsd_features['train_latents']
test_latents = nsd_features['test_latents']

train_fmri = np.load(train_fmri_file).astype(np.float32) / 300.0
test_fmri = np.load(test_fmri_file).astype(np.float32) / 300.0

# Normalization stats
norm_mean = train_fmri.mean(axis=0)
norm_std = train_fmri.std(axis=0)
train_fmri_norm = (train_fmri - norm_mean) / norm_std

# Train decoder with fixed alpha
alpha = 50000  # Fixed regularization parameter
decoder = Ridge(alpha=alpha, fit_intercept=True, max_iter=5000)
decoder.fit(train_latents, train_fmri_norm)
logging.info(f"Trained decoder for Subject {sub:02d} with fixed α={alpha}")

# Save weights
weights_out = weights_dir / f'vdvae_decoder_weights_sub{sub:02d}.pkl'
with open(weights_out, 'wb') as f:
    pickle.dump({'weight': decoder.coef_, 'bias': decoder.intercept_, 'mean': norm_mean, 'std': norm_std}, f)

# Load mask for NIfTI conversion
mask_img = nib.load(str(mask_file))
mask_data = mask_img.get_fdata() > 0

# Process both theta types
for theta_type in theta_types:
    logging.info(f"Processing theta type: {theta_type}")

    # Process each manipulation (alpha)
    nifti_files = []
    for a in alpha_values:
        theta_file = theta_dir / f"theta_{theta_type}_nsdgeneral_subject{sub}.npy"
        theta = np.load(theta_file)
        manipulated_latents = test_latents + (a * theta)

        # Predict normalized fMRI
        pred_test_norm = decoder.predict(manipulated_latents)
        pred_test = (pred_test_norm * norm_std + norm_mean) * 300.0  # Back to raw units

        # Save predicted fMRI (npy)
        npy_out = pred_dir / f'nsd_vdvae_test_fmri_map_sub{sub:02d}_theta{a:+d}_{theta_type}.npy'
        np.save(npy_out, pred_test)

        # Convert to NIfTI (specific test sample)
        target_latent_index = 755  # <<< your desired index
        fmri_sample = pred_test[target_latent_index]  # Get only the selected sample
        nifti_data = np.zeros(mask_data.shape, dtype=np.float32)
        nifti_data[mask_data] = fmri_sample
        nii_img = nib.Nifti1Image(nifti_data, affine=mask_img.affine, header=mask_img.header)
        nii_file = subj_output_dir / f"recon_fmri_alpha{a:+d}_idx{target_latent_index}_{theta_type}.nii.gz"
        nib.save(nii_img, nii_file)
        nifti_files.append(nii_file)
        logging.info(f"Saved NIfTI for α={a}, index={target_latent_index}, theta={theta_type} → {nii_file}")

    # Create contrast maps
    img_a = image.load_img(str(nifti_files[0]))  # alpha -4
    img_b = image.load_img(str(nifti_files[1]))  # alpha +4

    contrast_a_gt_b = image.math_img("img1 - img2", img1=img_a, img2=img_b)
    contrast_b_gt_a = image.math_img("img2 - img1", img1=img_a, img2=img_b)

    # Save contrasts
    contrast_a_file = subj_output_dir / f"contrast_alpha-4_gt_alpha4_{theta_type}.nii.gz"
    contrast_b_file = subj_output_dir / f"contrast_alpha4_gt_alpha-4_{theta_type}.nii.gz"
    contrast_a_gt_b.to_filename(str(contrast_a_file))
    contrast_b_gt_a.to_filename(str(contrast_b_file))
    logging.info(f"Saved contrast maps ({theta_type}): {contrast_a_file}, {contrast_b_file}")

logging.info(f"✅ Subject {sub:02d} processing complete.")
