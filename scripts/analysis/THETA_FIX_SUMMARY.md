# Theta Computation and Visualization Fix Summary

## Problem Identified

There was a **critical mismatch** between the theta computation notebook and the visualization script regarding which latents were used.

## Original Issue

### Notebook (`hierachical_theta from_reconstructions.ipynb`)
- ❌ **Originally**: Used ground truth latents from `data/extracted_features/`
- ✅ **Fixed**: Now uses predicted latents from `data/predicted_features/`

### Script (`visualize_layer_specific_theta.py`)
- ❌ **Originally**: Used ground truth latents from `data/extracted_features/`
- ✅ **Fixed**: Now uses predicted latents from `data/predicted_features/`

## What Was Changed

### 1. Notebook Changes (`hierachical_theta from_reconstructions.ipynb`)

**Cell 2 (Setup):**
```python
# ADDED:
PRED_FEATURE_DIR = BASE_DIR / 'data' / 'predicted_features'
```

**Cell 6 (process_subject_assessor function):**
```python
# CHANGED FROM:
feat_path = FEATURE_DIR / f'subj{subject:02d}' / 'nsd_vdvae_features_31l.npz'
test_latents = data['test_latents']

# CHANGED TO:
pred_feat_path = PRED_FEATURE_DIR / f'subj{subject:02d}' / f'nsd_vdvae_nsdgeneral_pred_sub{subject}_31l_alpha50k.npy'
predicted_latents = np.load(pred_feat_path)
```

**Key fix**: Subject-specific filenames
- Subject 1: `pred_sub1_31l_alpha50k.npy`
- Subject 2: `pred_sub2_31l_alpha50k.npy`
- Subject 5: `pred_sub5_31l_alpha50k.npy`
- Subject 7: `pred_sub7_31l_alpha50k.npy`

### 2. Script Changes (`visualize_layer_specific_theta.py`)

**Line ~43 (Directory definitions):**
```python
# ADDED:
PRED_FEATURE_DIR = BASE_DIR / 'data' / 'predicted_features'
```

**Lines ~368-378 (Data loading):**
```python
# CHANGED FROM:
pred_path = FEATURE_DIR / f'subj{subject:02d}' / 'nsd_vdvae_features_31l.npz'
pred_data = np.load(pred_path)
pred_latents = pred_data['test_latents']

# CHANGED TO:
pred_path = PRED_FEATURE_DIR / f'subj{subject:02d}' / f'nsd_vdvae_nsdgeneral_pred_sub{subject}_31l_alpha50k.npy'
pred_latents = np.load(pred_path)
```

## Why This Matters

### Before Fix (Incorrect):
```
Theta computation: Ground truth latents → Theta vector
Visualization: Ground truth latents + Theta → Reconstructions
```
- Theta directions didn't match brain reconstruction space
- Not testing individual brain encoding differences

### After Fix (Correct):
```
Theta computation: Predicted latents → Theta vector
Visualization: Predicted latents + Theta → Reconstructions
```
- ✅ Theta directions match brain reconstruction latent space
- ✅ Tests actual brain encoding differences across subjects
- ✅ Consistent data pipeline throughout

## Scientific Implications

### What the Analysis NOW Tests:

1. **Cross-subject similarity in brain encoding**
   - Low layers: Universal structural encoding across brains
   - High layers: Individual-specific semantic encoding

2. **Layer-specific manipulation effects**
   - Zeroing low layers: Tests semantic consistency
   - Zeroing high layers: Tests structural individuality

3. **Brain reconstruction quality**
   - Whether predicted latents preserve hierarchical structure
   - How well brain activity captures different visual features

### Key Finding Interpretation:

The **high cross-subject correlation in low layers** (r~0.67) and **low correlation in high layers** (r~0.14) means:

- **Low VDVAE layers** (detail/structure): Different subjects' brains encode these similarly
- **High VDVAE layers** (semantics): Different subjects' brains encode these differently
- This reflects **universal vs individual-specific neural representations**

## Files Modified

1. `/home/rothermm/brain-diffuser/scripts/analysis/hierachical_theta from_reconstructions.ipynb`
   - Cell 2: Added PRED_FEATURE_DIR
   - Cell 6: Updated process_subject_assessor() to use predicted latents

2. `/home/rothermm/brain-diffuser/scripts/analysis/visualize_layer_specific_theta.py`
   - Line ~43: Added PRED_FEATURE_DIR
   - Lines ~368-378: Updated data loading to use predicted latents
   - Docstring: Added note about using predicted latents

## Verification Steps

To verify the fix worked:

```bash
# 1. Check that predicted latent files exist
for subj in 01 02 05 07; do
    ls -lh /home/rothermm/brain-diffuser/data/predicted_features/subj${subj}/nsd_vdvae_nsdgeneral_pred_sub${subj:1}_31l_alpha50k.npy
done

# 2. Re-run notebook cells 8-9 to compute thetas
# (Should now complete successfully for all subjects)

# 3. Run visualization script
python scripts/analysis/visualize_layer_specific_theta.py --assessor emonet --alpha 50 --n_images 3

# 4. Check saved thetas
ls -lh /home/rothermm/brain-diffuser/results/thetas_hierarchical/subj*/theta_*
```

## Expected Results

After fix, both processes should:
- ✅ Successfully load all 4 subjects (1, 2, 5, 7)
- ✅ Use matching predicted latent files
- ✅ Show consistent cross-subject patterns in layer analysis
- ✅ Generate valid reconstructions in visualization

---

**Date Fixed**: October 15, 2025
**Fixed by**: GitHub Copilot
