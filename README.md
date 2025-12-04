
<img width="985" height="378" alt="Screenshot from 2025-07-18 15-10-59" src="https://github.com/user-attachments/assets/cbe55d09-9bef-4bee-9d2d-2b46ffe2e84c" />

<img width="2816" height="1536" alt="Gemini_Generated_Image_efvzgbefvzgbefvz" src="https://github.com/user-attachments/assets/576579ae-32c6-4288-be91-976c52e502b1" />


# Optimal Neural Engineering of Human Cognitive Experience

The initial scripts and processing pipeline in the `scripts` directory originate from the [**brain-diffuser**](https://github.com/furkanozcelik/brain-diffuser) project by Furkan Ozcelik and Rufin VanRullen. These provide the base utilities for extracting features, running regressions and reconstructing images from brain data. Downloading the NSD-Dataset is explained there.

## Data preparation

The data preparation pipeline processes the Natural Scenes Dataset (NSD) fMRI data and reconstructs images from brain activity using a hierarchical approach combining VDVAE latent representations with Versatile Diffusion models.

### Pipeline Overview

1. **Preprocessing** (`01_prep_data_job.sh`)
   - Preprocesses fMRI data, stimuli images, and CLIP embeddings (vision and text)
   - Prepares data structures for all subjects (1, 2, 5, 7)

2. **Feature Extraction** (`02_extract_latents.sh`)
   - Extracts VDVAE latent features from stimuli images
   - Creates hierarchical latent representations for reconstruction

3. **Brain-to-Latent Regression** (`03_train_regressor.sh`)
   - Trains Ridge regression models mapping fMRI activity to VDVAE latent space
   - Saves regression weights for each subject

4. **Initial Reconstruction** (`04_vdvae_generate_images.sh` & `04b_vdvae_image_to_image.sh`)
   - Generates reconstructed images from predicted VDVAE latents
   - Validates encoder-decoder pipeline with roundtrip tests

5. **CLIP Feature Processing** (`05-08_cliptext/vision_extraction/regression.sh`)
   - Extracts CLIP text and vision embeddings from images
   - Trains fMRI-to-CLIP regression models for guidance signals

6. **Final Reconstruction** (`09_versatilediff_reconstruct.sh` & `09b`)
   - Uses Versatile Diffusion conditioned on predicted latents
   - Refines reconstructions with CLIP vision and text guidance
   - Produces high-quality final reconstructed images



## Data Analysis

The analysis pipeline investigates how latent space manipulations affect reconstruction quality, memorability, and emotional content through theta-based optimization.

### Analysis Overview

1. **Theta Computation** (`01_compute_theta.sh`)
   - Computes optimization direction vectors from EmoNet (emotion) and MemNet (memorability) assessors
   - Identifies latent space directions that maximize memorability and emotional valence

2. **Theta-Based Reconstruction** (`02_vdvae_reconstruct_theta.sh` & `04_vd_reconstruct_theta.sh`)
   - Generates reconstructions with theta modifications at multiple alpha levels (α = -4, -2, 0, +2, +4)
   - Creates images optimized for high/low memorability and emotion

3. **Quality Assessment** (`03_assessor_scores.sh`, `07_ssim.sh`, `10_pixel_corr.sh`)
   - Computes EmoNet and MemNet scores for all reconstructions
   - Measures Structural Similarity Index (SSIM) and pixel-wise correlations
   - Quantifies reconstruction fidelity and perceptual quality

4. **Brain Mapping** (`05_map_latents_to_fmri.sh`)
   - Reverse maps manipulated latents back to predicted fMRI patterns
   - Generates NIfTI brain images showing how latent manipulations affect neural activity
   - Creates contrast maps comparing brain responses to different theta manipulations

5. **Visualization** (`06_graphs_final.sh`)
   - Generates publication-quality graphs and figures
   - Visualizes assessor scores, reconstruction metrics, and theta effects

6. **Statistical Analysis** (`08_statistics.sh`)
   - Performs significance testing on reconstruction quality differences
   - Validates theta optimization effects across subjects and conditions

7. **Human Behavioral Validation**
   - **`human_data_detrending.ipynb`**: Preprocesses human ratings data, removes order effects
   - **`analyze_human_data.ipynb`**: A priori statistical analysis justifying alpha level exclusion decisions based on linearity, monotonicity, and image quality degradation





<img width="1009" height="552" alt="Screenshot from 2025-07-18 15-10-47" src="https://github.com/user-attachments/assets/c98749a8-a2b0-4109-bc33-38349c5df071" />

