"""
Parameter recovery check for the Brunton LAN.

Simulates behavioral data from known (lam, B) parameter combinations,
fits the LAN to each, and checks if recovered posteriors contain the true values.

True parameter sets tested:
    (lam=-0.3, B=1.5) — leaky integrator
    (lam= 0.0, B=1.0) — perfect integrator
    (lam= 0.3, B=2.0) — unstable integrator

Usage:
    python src/param_recovery.py
    python src/param_recovery.py --n_trials 500  # small/fast version
"""
import argparse
import pathlib
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pytensor.config.floatX = "float32"
from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

from hssm.distribution_utils import make_distribution, make_likelihood_callable

# Import simulator
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate_data import simulate_one_trial, COH_LEVELS

ONNX_PATH = pathlib.Path(__file__).resolve().parents[1] / "train_data" / "brunton.onnx"
OUT_PATH  = pathlib.Path(__file__).resolve().parents[1] / "train_data" / "param_recovery"

TRUE_PARAMS = [
    {"lam": -0.3, "B": 1.5, "label": "leaky"},
    {"lam":  0.0, "B": 1.0, "label": "perfect"},
    {"lam":  0.3, "B": 2.0, "label": "unstable"},
]


def simulate_dataset(lam: float, B: float, n_trials: int, seed: int = 0) -> pd.DataFrame:
    """Simulate n_trials from Brunton model at given (lam, B)."""
    np.random.seed(seed)
    rows = []
    for _ in range(n_trials):
        coh = float(np.random.choice(COH_LEVELS))
        rt, resp = simulate_one_trial(lam, B, coh)
        rows.append({"rt": rt, "response": resp, "coherence": abs(coh)})
    return pd.DataFrame(rows, dtype=np.float32)


def build_brunton_distribution():
    loglik_op = make_likelihood_callable(
        loglik=str(ONNX_PATH),
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[False, False, True],
    )
    return make_distribution(
        rv="brunton",
        loglik=loglik_op,
        list_params=["lam", "B", "coherence"],
        bounds={"lam": (-0.5, 0.5), "B": (0.5, 3.0)},
    )


def fit_one(df: pd.DataFrame, BruntonDist, draws: int = 500, tune: int = 500):
    obs = df[["rt", "response", "coherence"]].values.astype(np.float32)
    with pm.Model():
        lam = pm.Uniform("lam", lower=-0.5, upper=0.5)
        B   = pm.Uniform("B",   lower=0.5,  upper=3.0)
        BruntonDist("obs", lam=lam, B=B, coherence=obs[:, 2], observed=obs[:, :2])
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=2,
                          target_accept=0.9, random_seed=42, progressbar=False)
    return idata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    args = parser.parse_args()

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    BruntonDist = build_brunton_distribution()

    results = []
    for tp in TRUE_PARAMS:
        print(f"\nRecovering: lam={tp['lam']}, B={tp['B']} ({tp['label']})")

        df = simulate_dataset(tp["lam"], tp["B"], args.n_trials)
        print(f"  Simulated {len(df)} trials, accuracy={( df['response']==1).mean():.3f}")

        idata = fit_one(df, BruntonDist, args.draws, args.tune)

        lam_post = idata.posterior["lam"].values.flatten()
        B_post   = idata.posterior["B"].values.flatten()

        results.append({
            "label":        tp["label"],
            "true_lam":     tp["lam"],
            "true_B":       tp["B"],
            "mean_lam":     lam_post.mean(),
            "hdi3_lam":     np.percentile(lam_post, 3),
            "hdi97_lam":    np.percentile(lam_post, 97),
            "mean_B":       B_post.mean(),
            "hdi3_B":       np.percentile(B_post, 3),
            "hdi97_B":      np.percentile(B_post, 97),
            "recovered_lam": np.percentile(lam_post, 3) <= tp["lam"] <= np.percentile(lam_post, 97),
            "recovered_B":   np.percentile(B_post, 3)   <= tp["B"]   <= np.percentile(B_post, 97),
        })
        print(f"  lam: true={tp['lam']:.2f}, estimated={lam_post.mean():.3f} "
              f"[{np.percentile(lam_post,3):.3f}, {np.percentile(lam_post,97):.3f}]")
        print(f"  B:   true={tp['B']:.2f}, estimated={B_post.mean():.3f} "
              f"[{np.percentile(B_post,3):.3f}, {np.percentile(B_post,97):.3f}]")

        idata.to_netcdf(str(OUT_PATH / f"recovery_{tp['label']}.nc"))

    # Summary table
    results_df = pd.DataFrame(results)
    print("\n=== PARAMETER RECOVERY SUMMARY ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(OUT_PATH / "recovery_summary.csv", index=False)

    # Recovery plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, param, true_col, mean_col, lo_col, hi_col in [
        (axes[0], "lam", "true_lam", "mean_lam", "hdi3_lam", "hdi97_lam"),
        (axes[1], "B",   "true_B",   "mean_B",   "hdi3_B",   "hdi97_B"),
    ]:
        for _, row in results_df.iterrows():
            ax.errorbar(row[true_col], row[mean_col],
                       yerr=[[row[mean_col]-row[lo_col]], [row[hi_col]-row[mean_col]]],
                       fmt='o', capsize=5, label=row["label"])
        lims = [results_df[true_col].min()-0.2, results_df[true_col].max()+0.2]
        ax.plot(lims, lims, 'k--', alpha=0.4, label='identity')
        ax.set_xlabel(f"True {param}")
        ax.set_ylabel(f"Estimated {param}")
        ax.set_title(f"Parameter Recovery: {param}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH / "recovery_plot.png", dpi=150)
    print(f"Recovery plot saved to {OUT_PATH / 'recovery_plot.png'}")


if __name__ == "__main__":
    main()
