
<img width="985" height="378" alt="Screenshot from 2025-07-18 15-10-59" src="https://github.com/user-attachments/assets/cbe55d09-9bef-4bee-9d2d-2b46ffe2e84c" />

<img width="2816" height="1536" alt="Gemini_Generated_Image_efvzgbefvzgbefvz" src="https://github.com/user-attachments/assets/576579ae-32c6-4288-be91-976c52e502b1" />


# Optimal Neural Engineering of Human Cognitive Experience

This project reconstructs visual images from fMRI brain activity using a two-stage pipeline: VDVAE for latent space encoding and Versatile Diffusion for high-quality image generation. It extends the original [**brain-diffuser**](https://github.com/ozcelikfu/brain-diffuser) project by Furkan Ozcelik and Rufin VanRullen with **theta-based optimization** for manipulating memorability (MemNet) and emotional valence (EmoNet) in reconstructed images.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Data Acquisition](#data-acquisition)
- [Pretrained Models](#pretrained-models)
- [Data Preparation Pipeline](#data-preparation-pipeline)
- [Data Analysis Pipeline](#data-analysis-pipeline)
- [Expected Runtimes](#expected-runtimes)
- [References](#references)

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 1× NVIDIA GPU with 12GB VRAM | 2× NVIDIA GPUs with 24GB+ VRAM each |
| **RAM** | 64GB | 128GB |
| **Storage** | 500GB free space | 1TB+ SSD |
| **CPU** | 8 cores | 16+ cores |

> **Note:** The Versatile Diffusion reconstruction step (`09_versatilediffusion_reconstruct_images.py`) is designed for dual-GPU setups but can be modified for single-GPU configurations with reduced batch sizes.

### Software Requirements

| Software | Version |
|----------|---------|
| **Operating System** | Linux (Ubuntu 18.04+, CentOS 7+) |
| **CUDA** | 11.3+ |
| **cuDNN** | 8.2+ |
| **Python** | 3.8.x |
| **Conda** | 4.10+ (Miniconda or Anaconda) |
| **AWS CLI** | 2.0+ (for NSD data download) |

---

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/brain-diffuser.git
cd brain-diffuser
```

### Step 2: Create Conda Environment

**Option A: Using the provided environment file (recommended)**

```bash
conda env create -f environment.yml
conda activate brain-diffuser
```

**Option B: Manual installation with core dependencies**

```bash
conda create -n brain-diffuser python=3.8
conda activate brain-diffuser

# Install PyTorch with CUDA support
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install core scientific packages
pip install numpy==1.23.4 scipy==1.9.3 pandas==1.5.1 scikit-learn==1.1.3 scikit-image==0.17.2

# Install deep learning utilities
pip install transformers==4.19.2 tokenizers==0.12.1 einops==0.3.0 kornia==0.6.8

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install image processing
pip install pillow==9.2.0 opencv-python==4.5.1.48 imageio==2.22.4

# Install neuroimaging tools
pip install nibabel==4.0.2 h5py==3.7.0

# Install metrics and visualization
pip install matplotlib==3.4.2 lpips==0.1.3 ssim-pil==1.0.14 tqdm==4.60.0

# Install configuration utilities
pip install omegaconf==2.1.1 easydict==1.9 pyyaml==5.4.1

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import clip; print('CLIP installed successfully')"
python -c "import nibabel; print('NiBabel installed successfully')"
```

### Typical Installation Time

| Step | Time (Desktop with SSD) |
|------|-------------------------|
| Clone repository | < 1 minute |
| Create conda environment | 10–20 minutes |
| Download pretrained models | 15–30 minutes |
| **Total** | **~30–50 minutes** |

---

## Data Acquisition

### Natural Scenes Dataset (NSD)

The project uses the [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/), a large-scale 7T fMRI dataset of human brain responses to natural images.

#### Step 1: Install AWS CLI

```bash
# Install AWS CLI (if not already installed)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure for anonymous access (NSD is public)
aws configure set aws_access_key_id ""
aws configure set aws_secret_access_key ""
```

#### Step 2: Download NSD Data

```bash
cd data

# Download experiment info, stimuli, betas, and ROIs for subjects 1, 2, 5, 7
python download_nsddata.py
```

This script downloads:
- **Experiment design**: `nsd_expdesign.mat`, `nsd_stim_info_merged.pkl`
- **Stimuli**: `nsd_stimuli.hdf5` (~28GB)
- **fMRI betas**: 37 sessions per subject (~200GB per subject)
- **ROI masks**: Functional ROI definitions

#### Step 3: Download COCO Annotations

Download `COCO_73k_annots_curated.npy` from [HuggingFace NSD Dataset](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main) and place it in the `data/` directory.

```bash
# Using wget (update URL if needed)
wget -O data/COCO_73k_annots_curated.npy "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/COCO_73k_annots_curated.npy"
```

#### Data Download Summary

| Data Component | Approximate Size | Download Time (100 Mbps) |
|----------------|------------------|--------------------------|
| Experiment info | ~50 MB | < 1 minute |
| Stimuli (HDF5) | ~28 GB | ~40 minutes |
| fMRI betas (4 subjects) | ~800 GB | ~18 hours |
| ROI masks | ~500 MB | ~1 minute |
| COCO annotations | ~200 MB | < 1 minute |
| **Total** | **~830 GB** | **~19 hours** |

---

## Pretrained Models

### VDVAE Model (First Stage)

Download the pretrained ImageNet64 VDVAE model:

```bash
mkdir -p vdvae/model
cd vdvae/model

wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th

cd ../..
```

### Versatile Diffusion Model (Second Stage)

Download from [HuggingFace Versatile Diffusion](https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth):

```bash
mkdir -p versatile_diffusion/pretrained
cd versatile_diffusion/pretrained

# Download these files manually or via huggingface-cli:
# - vd-four-flow-v1-0-fp16-deprecated.pth (~5.5GB)
# - kl-f8.pth (~335MB)  
# - optimus-vae.pth (~850MB)

cd ../..
```

### Assessor Networks (Project Extension)

The assessor weights should be placed in the `assessors/` directory:

| Model | File | Description |
|-------|------|-------------|
| **EmoNet** | `EmoNet_valence_moments_resnet50_5_best.pth.tar` | Emotional valence prediction (ResNet50) |
| **MemNet** | `memnet_state_dict.p` | Memorability prediction |
| **Image Mean** | `image_mean.npy` | BGR mean for MemNet preprocessing |

---

## Data Preparation Pipeline

The data preparation pipeline processes the Natural Scenes Dataset (NSD) fMRI data and reconstructs images from brain activity using a hierarchical approach combining VDVAE latent representations with Versatile Diffusion models.

### Pipeline Overview

| Step | Script | Description | GPU Required |
|------|--------|-------------|--------------|
| 1 | `01_prep_data_job.sh` | Preprocess fMRI data, stimuli images, and CLIP embeddings | No |
| 2 | `02_extract_latents.sh` | Extract VDVAE latent features from stimuli images | Yes |
| 3 | `03_train_regressor.sh` | Train Ridge regression models (fMRI → VDVAE latent space) | No |
| 4 | `04_vdvae_generate_images.sh` | Generate initial reconstructed images from predicted latents | Yes |
| 5-6 | `05-06_cliptext_*.sh` | Extract CLIP text embeddings and train regression | No |
| 7-8 | `07-08_clipvision_*.sh` | Extract CLIP vision embeddings and train regression | No |
| 9 | `09_versatilediff_reconstruct.sh` | Final Versatile Diffusion reconstructions | Yes (2× GPU) |

### Running the Pipeline

```bash
# Activate environment
module load miniconda  # HPC systems
source activate brain-diffuser

# Step 1: Prepare NSD data for all subjects
cd data
python 01_prepare_nsddata.py -sub 1
python 01_prepare_nsddata.py -sub 2
python 01_prepare_nsddata.py -sub 5
python 01_prepare_nsddata.py -sub 7

# Step 2: Extract VDVAE features
python scripts/02_vdvae_extract_features.py -sub 1

# Step 3: Train regression models
python scripts/03_vdvae_regression.py -sub 1

# Step 4: Reconstruct with VDVAE
python scripts/04_vdvae_reconstruct_images.py -sub 1

# Steps 5-8: CLIP feature extraction and regression
python scripts/05_cliptext_extract_features.py -sub 1
python scripts/06_cliptext_regression.py -sub 1
python scripts/07_clipvision_extract_features.py -sub 1
python scripts/08_clipvision_regression.py -sub 1

# Step 9: Final Versatile Diffusion reconstruction
python scripts/09_versatilediffusion_reconstruct_images.py -sub 1
```

### SLURM Job Submission (HPC)

```bash
cd slurm_scripts

# Submit jobs sequentially (wait for each to complete)
sbatch 01_prep_data_job.sh
sbatch 02_extract_latents.sh
sbatch 03_train_regressor.sh
sbatch 04_vdvae_generate_images.sh
sbatch 05_cliptext_extraction.sh
sbatch 06_cliptext_regression.sh
sbatch 07_clipvision_extract.sh
sbatch 08_clipvision_regression.sh
sbatch 09_versatilediff_reconstruct.sh
```



---

## Data Analysis Pipeline (Project Extension)

The analysis pipeline extends the original brain-diffuser with **theta-based optimization** to investigate how latent space manipulations affect reconstruction quality, memorability, and emotional content.

### Theta-Based Optimization

Theta vectors represent directions in latent space that maximize or minimize specific cognitive properties:

```
θ = mean(latents[top_15%_scores]) - mean(latents[bottom_15%_scores])
```

Manipulated latents are computed as:
```
latent_manipulated = latent_original + α × θ
```

Where α ∈ {-4, -2, 0, +2, +4} controls manipulation strength.

### Analysis Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_compute_theta.sh` | Compute optimization direction vectors from EmoNet/MemNet |
| 2 | `02_vdvae_reconstruct_theta.sh` | Generate VDVAE reconstructions with theta modifications |
| 3 | `03_assessor_scores.sh` | Compute EmoNet and MemNet scores for all reconstructions |
| 4 | `04_vd_reconstruct_theta.sh` | Versatile Diffusion reconstructions with theta |
| 5 | `05_map_latents_to_fmri.sh` | Reverse map latents to predicted fMRI patterns |
| 6 | `06_graphs_final.sh` | Generate publication-quality figures |
| 7 | `07_ssim.sh` | Compute Structural Similarity Index metrics |
| 8 | `08_statistics.sh` | Perform significance testing |
| 9 | `10_pixel_corr.sh` | Compute pixel-wise correlations |

### Assessor Networks

| Assessor | Architecture | Output | Preprocessing |
|----------|--------------|--------|---------------|
| **EmoNet** | ResNet50 | Valence score [-1, 1] | ImageNet normalization |
| **MemNet** | Custom CNN | Memorability score [0, 1] | BGR conversion, mean subtraction |

### Human Behavioral Validation

- **`human_data_detrending.ipynb`**: Preprocesses human ratings data, removes order effects
- **`analyze_human_data.ipynb`**: A priori statistical analysis justifying alpha level exclusion decisions based on linearity, monotonicity, and image quality degradation

---

## Expected Runtimes

### Data Preparation Pipeline

| Step | Script | Runtime per Subject | Hardware |
|------|--------|---------------------|----------|
| Data preprocessing | `01_prepare_nsddata.py` | 30–60 minutes | CPU (16 cores) |
| VDVAE feature extraction | `02_vdvae_extract_features.py` | 2–3 hours | 1× GPU (12GB) |
| Ridge regression training | `03_vdvae_regression.py` | 15–30 minutes | CPU (128GB RAM) |
| VDVAE reconstruction | `04_vdvae_reconstruct_images.py` | 1–2 hours | 1× GPU (12GB) |
| CLIP text extraction | `05_cliptext_extract_features.py` | 30–45 minutes | CPU |
| CLIP text regression | `06_cliptext_regression.py` | 10–15 minutes | CPU |
| CLIP vision extraction | `07_clipvision_extract_features.py` | 45–60 minutes | 1× GPU |
| CLIP vision regression | `08_clipvision_regression.py` | 10–15 minutes | CPU |
| Versatile Diffusion | `09_versatilediffusion_reconstruct.py` | 6–10 hours | 2× GPU (24GB each) |

**Total runtime per subject: ~12–18 hours**  
**Total for all 4 subjects: ~2–3 days**

### Analysis Pipeline

| Step | Script | Runtime | Hardware |
|------|--------|---------|----------|
| Theta computation | `01_compute_theta.py` | 5–10 minutes | CPU |
| VDVAE theta reconstruction | `02_vdvae_reconstruct_theta.py` | 2–4 hours | 1× GPU |
| Assessor scoring | `03_assessor_scores.py` | 30–60 minutes | 1× GPU |
| VD theta reconstruction | `04_vd_reconstruct_theta.py` | 8–12 hours | 2× GPU |
| Brain mapping | `05_map_latents_to_fmri.py` | 1–2 hours | CPU |
| Graph generation | `06_graphs_final.py` | 10–15 minutes | CPU |
| SSIM computation | `07_ssim.py` | 15–30 minutes | CPU |
| Statistical analysis | `08_statistics.py` | 5–10 minutes | CPU |

**Total analysis runtime: ~15–24 hours**

### Benchmarks (Reference Hardware)

| Configuration | Total Pipeline Time |
|---------------|---------------------|
| 2× NVIDIA A100 (80GB), 128GB RAM, 32 cores | ~1.5 days (all subjects) |
| 2× NVIDIA RTX 3090 (24GB), 128GB RAM, 16 cores | ~2.5 days (all subjects) |
| 1× NVIDIA RTX 3080 (10GB), 64GB RAM, 8 cores | ~5–7 days (all subjects) |

---

## Output Directory Structure

```
results/
├── vdvae/
│   └── subj{01,02,05,07}/          # VDVAE reconstructions
├── versatile_diffusion/
│   └── subj{01,02,05,07}/          # Final VD reconstructions
├── thetas/
│   └── subj{01,02,05,07}/
│       ├── theta_emonet_*.npy      # Emotion optimization vectors
│       └── theta_memnet_*.npy      # Memorability optimization vectors
├── assessor_scores/                 # EmoNet/MemNet scores
├── statistics/                      # Statistical test results
├── metrics/                         # SSIM, pixel correlation metrics
├── graphs/                          # Visualization outputs
└── nifti/                          # Brain mapping volumes
```

---

## Key Implementation Details

### fMRI Preprocessing
```python
train_fmri = train_fmri / 300  # Scale factor
norm_mean = np.mean(train_fmri, axis=0)
norm_scale = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean) / norm_scale
```

### Ridge Regression Configuration
```python
reg = sklearn.linear_model.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
```

### VDVAE Configuration
- Model: ImageNet64 pretrained
- Latent layers: 31 hierarchical levels
- Z-dimension: 16 per layer

---

## References

- **Brain-Diffuser**: Ozcelik, F., & VanRullen, R. (2023). [Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion](https://arxiv.org/abs/2303.05334). *arXiv preprint*.

- **VDVAE**: Child, R. (2020). [Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images](https://arxiv.org/abs/2011.10650). OpenAI.

- **Versatile Diffusion**: Xu, X., et al. (2023). [Versatile Diffusion: Text, Images and Variations All in One Diffusion Model](https://arxiv.org/abs/2211.08332). SHI Labs.

- **Natural Scenes Dataset**: Allen, E. J., et al. (2022). [A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence](https://www.nature.com/articles/s41593-021-00962-x). *Nature Neuroscience*.

- **EmoNet**: Kragel, P. A., et al. (2019). [Emotion schemas are embedded in the human visual system](https://www.science.org/doi/10.1126/sciadv.aaw4358). *Science Advances*.

- **MemNet**: Khosla, A., et al. (2015). [Understanding and Predicting Image Memorability at a Large Scale](https://people.csail.mit.edu/khosla/papers/iccv2015_khosla.pdf). *ICCV*.

### Code Attribution

- VDVAE implementation: [openai/vdvae](https://github.com/openai/vdvae)
- Versatile Diffusion: [SHI-Labs/Versatile-Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion)
- Original brain-diffuser: [ozcelikfu/brain-diffuser](https://github.com/ozcelikfu/brain-diffuser)

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

<img width="1009" height="552" alt="Screenshot from 2025-07-18 15-10-47" src="https://github.com/user-attachments/assets/c98749a8-a2b0-4109-bc33-38349c5df071" />

