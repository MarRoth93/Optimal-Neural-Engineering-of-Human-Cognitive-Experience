# Analysis Overview: Model Assessor Scores & Human Behavioral Data

This analysis script loads model assessor scores and human behavioral data, processes them, and produces several visualizations to compare models (VDVAE, Versatile) against human ratings on two networks (EmoNet and MemNet). Below is a summary of what data was used, how it was treated, and how to interpret each type of plot.

---

## Data Loading

### Model Data
- **Source**: Pickle files (`*.pkl`) per subject, network, and model (VDVAE, Versatile).
- **Content**: Alpha-parameter modulated scores for images.
- **Structure**: Nested dictionary `model_data[network][model][subject]` with scores at 5 alpha levels: `[-4, -2, 0, 2, 4]`.

### Human Data
- **Source**: CSV file `human_df_detrended.csv`.
- **Content**: Behavioral ratings for valence (EmoNet) and memorability (MemNet).
- **Preprocessing**:
  - Conditions (e.g., `valence-4`, `mem+2`) mapped to numeric alpha values.
  - Grouped by alpha to compute mean ratings.
  - Normalized per subject using min-max scaling.

---

## Data Treatment

- **Normalization**: Model and human scores were min-max normalized within subjects to enable direct comparison across alpha levels.
- **Slopes**: Response slopes (change in score per alpha unit) computed using linear regression for each image.
- **Rate of Change (ROC)**: Ratios of modulated scores to baseline (`alpha_0`) for models and humans.

---

## Plot Interpretations

### Normalized Mean Scores
#### Per-Subject Comparison
- **What it shows**: For each subject, compares normalized mean scores across alpha levels between:
  - Human ratings (Valence or Memorability)
  - VDVAE and Versatile model scores
- **How to read**:
  - X-axis: Alpha levels (-4 to +4)
  - Y-axis: Normalized mean scores (0 to 1 scale)
  - Each line represents a model or human ratings
  - Trends indicate sensitivity of models/humans to alpha modulation

#### Averaged Across Subjects
- **What it shows**: Similar to above but averaged across all subjects.
- **How to read**:
  - Smoother lines represent overall trends
  - Useful for identifying systematic differences between models and humans

---

### Slope Distributions
#### Per-Subject Histograms
- **What it shows**: Distribution of response slopes for each subject and model.
- **How to read**:
  - X-axis: Slope (Δ score per alpha unit)
  - Y-axis: Number of images
  - Wider distributions indicate more variability in sensitivity to alpha modulation.

#### Pooled Histogram
- **What it shows**: All subjects' slopes combined into one distribution per model.
- **How to read**:
  - Direct comparison of model behavior across subjects
  - Skewness and spread reveal whether models respond more steeply or uniformly

---

### Rate of Change (ROC) Plots
#### Per-Subject ROC Histograms
- **What it shows**: For each subject, histograms of ROC values (modulated/baseline scores) for EmoNet and MemNet.
- **How to read**:
  - X-axis: ROC values
  - Y-axis: Count of images
  - Two overlaid histograms: VDVAE vs. Versatile
  - Shifts in distribution center indicate stronger/weaker modulation sensitivity

#### Overall ROC Histograms
- **What it shows**: Pooled ROC histograms across all subjects for each network.
- **How to read**:
  - Highlights global differences between models’ modulation sensitivity

#### Model vs. Bootstrap-Resampled Human ROC
- **What it shows**: Comparison of overall model ROC distributions to bootstrap-resampled human ROC.
- **How to read**:
  - Allows direct visual comparison of model and human modulation behavior
  - Overlap suggests similar sensitivity; separation indicates discrepancies

---

## Key Takeaways
- Are models over- or under-sensitive relative to humans?
- Do models reproduce human variability in slopes and ROC?
- Which network (EmoNet/MemNet) shows stronger alignment?


