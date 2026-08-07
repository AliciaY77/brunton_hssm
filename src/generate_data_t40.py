"""
Brunton accumulator simulator for LAN training.

T=40 version for matching RNN timescale.

Matches the exact implementation in RNN_Threshold_Alicia/scripts/brunton_model.py:
    a <- a + lambda*a*dt + sigma_s*x_t + acc_noise*randn()

Free parameters for LAN training:
    lam     : memory parameter, range (-2.0, 2.0)
    B       : decision bound |a| >= B, range (1.0, 15.0)

Fixed parameters (matching original):
    sigma_s  = 1.0
    acc_noise = 0.1  (per-timestep)
    T        = 40
    dt       = 1.0
    noise    = 0.5  (stimulus noise std, per-timestep)

Coherence is included as a trial-level covariate (not a free parameter).
Each simulated trial uses a randomly drawn coherence from [-0.15, 0.15].

Output format for HSSM:
    rt       : reaction time in seconds (timesteps * 0.025)
    response : 1 (correct/right) or -1 (incorrect/left)
    coherence: absolute coherence value used in that trial
"""
import numpy as np
import pathlib

# Fixed parameters matching brunton_model.py
SIGMA_S   = 1.0
ACC_NOISE = 0.1  # per-timestep
T         = 40
DT        = 1.0
NOISE     = 0.5  # per-timestep
COH_LEVELS = np.linspace(-0.15, 0.15, 11)


def simulate_one_trial(lam, B, coherence):
    """Simulate a single trial. Returns (rt_seconds, response).
    response: 1=correct, -1=incorrect (accuracy coding for HSSM)
    """
    a = 0.0
    rt = T
    for t in range(T):
        x_t = coherence + NOISE * np.random.randn()
        a = a + lam * a * DT + SIGMA_S * x_t + ACC_NOISE * np.random.randn()
        if abs(a) >= B:
            rt = t + 1
            break
    # Accuracy coding: correct if accumulator sign matches coherence sign
    if coherence == 0:
        response = 1.0  # treat zero coherence as correct by convention
    elif coherence > 0:
        response = 1.0 if a > 0 else -1.0
    else:
        response = 1.0 if a < 0 else -1.0
    rt_seconds = rt * 0.025
    return rt_seconds, response


def generate_training_data(n_parameter_sets=2000, n_samples_per_set=500, seed=42):
    """
    Generate LAN training data by uniformly sampling (lam, B) parameter sets
    and simulating trials with random coherence levels.

    Returns numpy array of shape (n_parameter_sets * n_samples_per_set, 5):
        columns: [lam, B, coherence, rt, response]

    Note: coherence is included as a trial-level covariate so the LAN
    can learn how coherence modulates the RT/choice distribution.
    """
    np.random.seed(seed)

    lam_range = (-2.0, 2.0)
    B_range   = (1.0, 15.0)

    all_rows = []
    total = n_parameter_sets * n_samples_per_set

    for i in range(n_parameter_sets):
        if i % 200 == 0:
            print(f"  Parameter set {i}/{n_parameter_sets} "
                  f"({len(all_rows):,}/{total:,} trials)")

        lam = np.random.uniform(*lam_range)
        B   = np.random.uniform(*B_range)

        for _ in range(n_samples_per_set):
            coh = np.random.choice(COH_LEVELS)
            rt, resp = simulate_one_trial(lam, B, coh)
            all_rows.append([lam, B, float(coh), rt, float(resp)])

    data = np.array(all_rows, dtype=np.float32)
    print(f"Done. Generated {len(data):,} training examples.")
    return data


if __name__ == "__main__":
    out_dir = pathlib.Path(__file__).resolve().parents[1] / "train_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "brunton_training_data_t40.npy"

    print("Generating Brunton LAN training data...")
    print(f"  Fixed: sigma_s={SIGMA_S}, acc_noise={ACC_NOISE}, T={T}, dt={DT}, noise={NOISE}")
    print(f"  Free:  lam in (-2.0, 2.0), B in (1.0, 15.0)")
    print(f"  Covariate: coherence in {COH_LEVELS.tolist()}")

    data = generate_training_data(n_parameter_sets=2000, n_samples_per_set=500)
    np.save(out_path, data)
    print(f"Saved to {out_path}  shape={data.shape}")
    print("Columns: [lam, B, coherence, rt, response]")
