"""
Fit Brunton LAN to RNN behavioral data using HSSM.

Uses the trained ONNX network to estimate posterior distributions
over Brunton parameters (lam, B) from RNN-generated (rt, response, coherence).

Usage:
    python src/fit_lan.py --gain 1.0 --seed 42
"""
import argparse
import pathlib
import pandas as pd
import numpy as np
import pymc as pm
import pytensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pytensor.config.floatX = "float32"
from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

from hssm.distribution_utils import make_distribution, make_likelihood_callable

ONNX_PATH = pathlib.Path(__file__).resolve().parents[1] / "train_data" / "brunton.onnx"
DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "train_data" / "rnn_data"
OUT_PATH  = pathlib.Path(__file__).resolve().parents[1] / "train_data" / "fit_results"


def load_rnn_data(gain: float, seed: int) -> pd.DataFrame:
    """Load processed RNN behavioral data for a given gain and seed."""
    fname = DATA_PATH / f"hssm_ready_nxx1_s{seed}_g{gain}.parquet"
    if not fname.exists():
        raise FileNotFoundError(f"Data file not found: {fname}\nRun process_data.py first.")
    df = pd.read_parquet(fname)
    print(f"  Loaded {len(df):,} trials (gain={gain}, seed={seed})")
    print(f"  RT range: {df['rt'].min():.3f} - {df['rt'].max():.3f} s")
    print(f"  Accuracy: {(df['response']==1.0).mean():.3f}")
    return df


def build_brunton_distribution():
    """Build HSSM-compatible distribution from trained ONNX LAN."""
    print(f"Loading ONNX from {ONNX_PATH}")
    loglik_op = make_likelihood_callable(
        loglik=str(ONNX_PATH),
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[False, False, True],  # lam=fixed, B=fixed, coherence=regression
    )
    BruntonDist = make_distribution(
        rv="brunton",
        loglik=loglik_op,
        list_params=["lam", "B", "coherence"],
        bounds={"lam": (-0.5, 0.5), "B": (0.5, 3.0)},
    )
    return BruntonDist


def fit(df: pd.DataFrame, BruntonDist, gain: float, seed: int, draws: int = 1000, tune: int = 1000):
    """Fit Brunton LAN to data using PyMC NUTS."""
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Observed data: array of shape (n_trials, 3) = [rt, response, coherence]
    obs = df[["rt", "response", "coherence"]].values.astype(np.float32)

    with pm.Model() as model:
        lam = pm.Uniform("lam", lower=-0.5, upper=0.5)
        B   = pm.Uniform("B",   lower=0.5,  upper=3.0)

        _ = BruntonDist("obs", lam=lam, B=B, coherence=obs[:, 2], observed=obs[:, :2])

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=4,
            target_accept=0.9,
            random_seed=42,
        )

    out_file = OUT_PATH / f"brunton_fit_nxx1_s{seed}_g{gain}"
    idata.to_netcdf(str(out_file))
    print(f"Saved to {out_file}")

    # Print summary
    import arviz as az
    summary = az.summary(idata, var_names=["lam", "B"])
    print(summary)

    return idata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    args = parser.parse_args()

    print(f"Fitting Brunton LAN to RNN data (gain={args.gain}, seed={args.seed})")
    BruntonDist = build_brunton_distribution()
    df = load_rnn_data(args.gain, args.seed)
    fit(df, BruntonDist, args.gain, args.seed, args.draws, args.tune)


if __name__ == "__main__":
    main()
