# Optimal Neural Engineering of Human Cognitive Experience

The initial scripts and processing pipeline in the `scripts` directory originate from the [**brain-diffuser**](https://github.com/furkanozcelik/brain-diffuser) project by Furkan Ozcelik and Rufin VanRullen. These provide the base utilities for extracting features, running regressions and reconstructing images from brain data.

## Additions

The `scripts/analysis` folder extends the original pipeline with analysis tools and further reconstruction scripts:

- **`brain_mapping.py`** – Loads predicted fMRI volumes, converts them to NIfTI format and computes normalized contrast maps between manipulation conditions.
- **`compute_assessor_scores.py`** – Evaluates generated images with EmoNet and MemNet, gathering scores across different alpha manipulations for both VDVAE and Versatile Diffusion outputs.
- **`compute_theta.py`** – Computes direction vectors ("theta") for EmoNet and MemNet by contrasting the top and bottom responses to VDVAE reconstructions.
- **`map_latents_to_fmri.py`** – Maps manipulated latents to fMRI using ridge regression, saving predicted volumes and difference maps.
- **`map_latents_to_fmri_no_negative.py`** – Variant of the above that also produces contrast maps with negative values set to zero.
- **`vdvae_reconstruct_images_thetas.py`** – Decodes VDVAE latent vectors manipulated by theta directions to create alpha‑level image sets.
- **`vd_recon_thetas.py`** and **`vd_recon_thetas_memnet.py`** – Apply Versatile Diffusion to refine VDVAE images for EmoNet or MemNet‑based manipulations.
- **`versatile_diffusion_reconstruct_images_thetas.py`** – Runs Versatile Diffusion on all VDVAE reconstructions conditioned on predicted CLIP features.
- **`graphs.py`** and **`graphs_test.py`** – Load assessor scores and human behavioral data to produce plots of normalized scores, slope distributions and rate‑of‑change comparisons across subjects.
- Jupyter notebooks such as `analyze_assessor_scores.ipynb` and `human_data_detrending.ipynb` provide additional exploratory analysis.

These additions facilitate computing transformation directions, scoring and visualizing reconstructed images, and mapping manipulated latents back to brain space.
