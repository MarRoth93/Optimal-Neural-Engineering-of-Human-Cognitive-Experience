import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, image
from pathlib import Path
import logging

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- Configuration ---
sub = 1
latent_source_type = "vdvae"  # since you used VDVAE latents
target_latent_index = 755
alfa_values = [-4, 4]  # Two conditions to contrast

# --- Paths ---
base_path = Path("/home/rothermm/brain-diffuser")
pred_dir = base_path / f"data/predicted_fmri_mapping/subj{sub:02d}"
mask_dir = base_path / f"nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi"
mask_path = mask_dir / "nsdgeneral.nii.gz"
output_base_dir = base_path / f"data/brain_mapping/subj{sub:02d}"
output_base_dir.mkdir(parents=True, exist_ok=True)

# --- Load mask ---
logging.info(f"Loading mask from {mask_path}")
mask_img = nib.load(str(mask_path))
mask_data = mask_img.get_fdata() > 0

# --- Functions ---
def array_to_nifti(fmri_array, mask_img, out_path):
    """Map a 1D voxel array into a NIfTI volume using mask geometry and save."""
    vol_shape = mask_img.shape
    output_image_data = np.zeros(vol_shape, dtype=np.float32)
    output_image_data[mask_data] = fmri_array.flatten()
    out_img = nib.Nifti1Image(output_image_data, affine=mask_img.affine, header=mask_img.header)
    nib.save(out_img, str(out_path))
    logging.info(f"NIfTI saved to: {out_path}")

# --- Process each alfa ---
nifti_paths = {}
for alfa in alfa_values:
    logging.info(f"\nProcessing alfa={alfa}...")
    # Load predicted fMRI (all test samples)
    pred_fmri_path = pred_dir / f"nsd_vdvae_test_fmri_map_sub{sub:02d}_alpha50000.npy"
    pred_fmri_all = np.load(str(pred_fmri_path))  # Shape: (N_test, V)
    logging.info(f"Loaded predicted fMRI: {pred_fmri_path}, shape={pred_fmri_all.shape}")

    # Select specific latent index
    fmri_vector = pred_fmri_all[target_latent_index]
    logging.info(f"Selected latent index {target_latent_index}, shape={fmri_vector.shape}")

    # Save as NIfTI
    nifti_dir = output_base_dir / f"recon_fmri_{latent_source_type}_alfa{alfa}"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = nifti_dir / f"recon_fmri_{latent_source_type}_sub{sub:02d}_alfa{alfa}_idx{target_latent_index}.nii.gz"
    array_to_nifti(fmri_vector, mask_img, nifti_path)
    nifti_paths[alfa] = nifti_path

# --- Compute Normalized Contrast ---
logging.info("\nComputing normalized contrasts...")
img_a = image.load_img(str(nifti_paths[alfa_values[0]]))
img_b = image.load_img(str(nifti_paths[alfa_values[1]]))
data_a = img_a.get_fdata()
data_b = img_b.get_fdata()

# Normalized contrast (A > B)
norm_contrast_a = np.divide(
    data_a - data_b,
    data_a + data_b,
    out=np.zeros_like(data_a),
    where=(data_a + data_b) != 0
)
norm_contrast_a[norm_contrast_a < 0] = 0
contrast_img_a = image.new_img_like(img_a, norm_contrast_a)
contrast_dir = output_base_dir / "difference_maps"
contrast_dir.mkdir(parents=True, exist_ok=True)
contrast_path_a = contrast_dir / f"norm_contrast_{latent_source_type}_alfa{alfa_values[0]}gt{alfa_values[1]}_idx{target_latent_index}.nii.gz"
contrast_img_a.to_filename(str(contrast_path_a))
logging.info(f"Saved normalized contrast (A > B): {contrast_path_a}")

# Normalized contrast (B > A)
norm_contrast_b = np.divide(
    data_b - data_a,
    data_b + data_a,
    out=np.zeros_like(data_b),
    where=(data_b + data_a) != 0
)
norm_contrast_b[norm_contrast_b < 0] = 0
contrast_img_b = image.new_img_like(img_b, norm_contrast_b)
contrast_path_b = contrast_dir / f"norm_contrast_{latent_source_type}_alfa{alfa_values[1]}gt{alfa_values[0]}_idx{target_latent_index}.nii.gz"
contrast_img_b.to_filename(str(contrast_path_b))
logging.info(f"Saved normalized contrast (B > A): {contrast_path_b}")
