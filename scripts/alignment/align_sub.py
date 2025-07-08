#!/usr/bin/env python3
import argparse
import os
import numpy as np
from sklearn.linear_model import RidgeCV
import tqdm

# --- Project root -------------------------------------------------------------
BASE_DIR = "/home/rothermm/brain-diffuser"
os.chdir(BASE_DIR)

def main():
    # --- Parse arguments -------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Align a source subject to template subject 1'
    )
    parser.add_argument(
        "-sub", "--sub",
        type=int,
        choices=[1, 2, 5, 7],
        help="Source subject number",
        default=1
    )
    args = parser.parse_args()
    source_sub = args.sub
    target_sub = 1
    if source_sub == target_sub:
        print(f"Source and target are both {target_sub}: alignment will be identity.")

    # --- File paths ------------------------------------------------------------
    base_in  = "data/processed_data"
    base_out = "data/aligned_subjects"

    source_test_pth  = os.path.join(
        base_in,
        f"subj{source_sub:02d}",
        f"nsd_test_fmriavg_nsdgeneral_sub{source_sub}.npy"
    )
    source_train_pth = os.path.join(
        base_in,
        f"subj{source_sub:02d}",
        f"nsd_train_fmriavg_nsdgeneral_sub{source_sub}.npy"
    )
    target_test_pth  = os.path.join(
        base_in,
        f"subj{target_sub:02d}",
        f"nsd_test_fmriavg_nsdgeneral_sub{target_sub}.npy"
    )
    target_train_pth = os.path.join(
        base_in,
        f"subj{target_sub:02d}",
        f"nsd_train_fmriavg_nsdgeneral_sub{target_sub}.npy"
    )

    out_dir = os.path.join(
        base_out,
        f"subj{source_sub:02d}",
        f"aligned_to_subj{target_sub:02d}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # --- Load data -------------------------------------------------------------
    # Corrected the ellipsis character here
    print("Loading data...") 
    source_test  = np.load(source_test_pth)
    source_train = np.load(source_train_pth)
    target_test  = np.load(target_test_pth)
    target_train = np.load(target_train_pth)

    # --- Prepare index splits ---------------------------------------------------
    # This dictionary currently only processes the full dataset ("100%").
    # If you intend to process different fractions, you'll need to add more entries.
    indices_100 = np.arange(len(target_test))
    percent_indices = {"100": indices_100}

    # --- Alignment loop --------------------------------------------------------
    for frac, idxs in tqdm.tqdm(percent_indices.items(), desc="Fractions"):
        # Initialize RidgeCV aligner with specified alpha values for regularization
        aligner = RidgeCV(alphas=[1e2, 1e3, 1e4, 5e4], fit_intercept=True)
        
        # Fit the aligner using the source and target test data for the current fraction
        # Note: It's common to fit on training data. Here, it seems a portion of test data is used for fitting.
        aligner.fit(source_test[idxs], target_test[idxs])

        # Predict aligned versions of the full source test and train datasets
        src_test_aligned  = aligner.predict(source_test)
        src_train_aligned = aligner.predict(source_train)

        # Calculate mean and standard deviation of the target test data for rescaling
        tgt_mean = target_test.mean(0)
        tgt_std  = target_test.std(0)

        # Define a rescaling function to match the target data's distribution
        def rescale(mat):
            # Calculate standard deviation of the input matrix, adding a small epsilon to avoid division by zero
            mat_std = mat.std(0) + 1e-8 
            # Z-score the input matrix (subtract mean, divide by standard deviation)
            mat_z    = (mat - mat.mean(0)) / mat_std
            # Rescale the z-scored matrix to match the target's standard deviation and mean
            return mat_z * tgt_std + tgt_mean

        # Rescale the aligned source test and train data
        src_test_adj  = rescale(src_test_aligned)
        src_train_adj = rescale(src_train_aligned)

        # --- Save outputs -----------------------------------------------------
        # Construct the output file path for the aligned training data
        train_output_filename = f"nsd_train_fmriavg_nsdgeneral_sub{source_sub}_ridge_fraction-{frac}.npy"
        np.save(
            os.path.join(out_dir, train_output_filename),
            src_train_adj
        )
        
        # Construct the output file path for the aligned test data
        test_output_filename = f"nsd_test_fmriavg_nsdgeneral_sub{source_sub}_ridge_fraction-{frac}.npy"
        np.save(
            os.path.join(out_dir, test_output_filename),
            src_test_adj
        )

    print(f"Done! Aligned data saved under {out_dir}")

if __name__ == "__main__":
    main()
