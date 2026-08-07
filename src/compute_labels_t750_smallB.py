"""
Compute KDE-based log-likelihood labels for LAN training.

For each unique (lam, B, coherence) combination, fit a KDE over the
simulated RT distributions separately for each response (-1, +1),
then evaluate the KDE at each observed (rt, response) to get
approximate log p(rt, response | lam, B, coherence).

This follows the approach described in Fengler et al. (2021).
"""
import numpy as np
import pathlib
import pickle
from scipy.stats import gaussian_kde

DATA_PATH  = pathlib.Path(__file__).resolve().parents[1] / "train_data" / "brunton_training_data_t750_smallB.npy"
OUT_PATH   = pathlib.Path(__file__).resolve().parents[1] / "train_data" / "brunton_labeled_t750_smallB.pickle"


def compute_labels(data, bandwidth=0.1):
    """
    data: array of shape (N, 5) with columns [lam, B, coherence, rt, response]
    Returns labels array of shape (N,) with log-likelihood values.
    """
    labels = np.full(len(data), -10.0, dtype=np.float32)

    lam  = data[:, 0]
    B    = data[:, 1]
    coh  = data[:, 2]
    rts  = data[:, 3]
    resp = data[:, 4]

    # Round params for grouping
    lam_r = np.round(lam, 2)
    B_r   = np.round(B, 2)
    coh_r = np.round(coh, 3)

    keys = np.stack([lam_r, B_r, coh_r], axis=1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    print(f"Computing KDE labels for {len(unique_keys)} unique parameter groups...")

    for i, key in enumerate(unique_keys):
        if i % 500 == 0:
            print(f"  Group {i}/{len(unique_keys)}")

        group_mask = inverse == i
        group_rts  = rts[group_mask]
        group_resp = resp[group_mask]

        for r in [-1.0, 1.0]:
            r_mask = group_resp == r
            full_mask = group_mask.copy()
            full_mask[group_mask] = r_mask

            rt_subset = group_rts[r_mask]
            if len(rt_subset) < 5:
                continue

            try:
                kde = gaussian_kde(rt_subset, bw_method=bandwidth)
                prop = r_mask.mean()  # proportion of this response
                log_dens = np.log(kde(rt_subset) * prop + 1e-10)
                labels[full_mask] = log_dens.astype(np.float32)
            except Exception:
                pass

    return labels


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}")
    data = np.load(DATA_PATH)
    print(f"  Shape: {data.shape}")

    labels = compute_labels(data)
    print(f"  Labels range: {labels.min():.2f} to {labels.max():.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Saved labeled data to {OUT_PATH}")
