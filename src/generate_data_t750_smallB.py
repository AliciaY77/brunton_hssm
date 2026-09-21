"""
Brunton accumulator simulator for LAN training.

T=750, small B version matching RNN timescale.

Matches the exact implementation in RNN_Threshold_Alicia/scripts/brunton_model.py:
    a <- a + lambda*a*dt + sigma_s*x_t + acc_noise*randn()

Free parameters for LAN training:
    B       : decision bound |a| >= B, range (8.0, 35.0)

Fixed parameters (updated per Ivan's advice):
    lam      = 0.0  (fixed, not free — not identifiable from RT+choice alone)
    sigma_s  = 1.0
    acc_noise = 0.6  (per-timestep)
    T        = 750
    dt       = 0.001  (so 750 steps = 750ms, not 15s)
    noise    = 1.0  (stimulus noise std, per-timestep; e100 family uses NOISE_IN=1.0)

Coherence is included as a trial-level covariate (not a free parameter).
Each simulated trial uses a randomly drawn coherence from [-0.15, 0.15].

Output format for HSSM:
    rt       : reaction time in seconds (timesteps * dt)
    response : 1 (correct/right) or -1 (incorrect/left)
    coherence: absolute coherence value used in that trial
"""
import numpy as np
import pathlib

# Fixed parameters — updated per Ivan's advice
SIGMA_S   = 1.0
ACC_NOISE = 0.6      # was 0.1 — needed to hit target accuracy
T         = 750
DT        = 0.001    # was 0.02 — so 750 steps = 750ms, not 15s
NOISE     = 1.0      # was 0.5 — e100 family uses NOISE_IN=1.0
COH_LEVELS = np.linspace(-0.15, 0.15, 11)

# Free parameters
# lam is FIXED at 0.0 (not free) — not identifiable from RT+choice alone
LAM = 0.0
B_RANGE = (8.0, 35.0)   # was (0.01, 0.5) — must cover target B≈22-23


def simulate_one_trial(B, coherence):
    """Simulate a single trial with lam fixed at 0.
    Returns (rt_seconds, response).
    response: 1=correct, -1=incorrect (accuracy coding for HSSM)
    """
    a = 0.0
    rt = T
    for t in range(T):
        x_t = coherence + NOISE * np.random.randn()
        a = a + LAM * a * DT + SIGMA_S * x_t + ACC_NOISE * np.random.randn()
        if abs(a) >= B:
            rt = t + 1
            break
    # Zero-coherence bug fix: random 50/50 response
    if coherence == 0:
        response = 1.0 if np.random.rand() < 0.5 else -1.0
    elif coherence > 0:
        response = 1.0 if a > 0 else -1.0
    else:
        response = 1.0 if a < 0 else -1.0
    rt_seconds = rt * DT
    return rt_seconds, response


def generate_training_data(n_parameter_sets=500, n_samples_per_set=4000, seed=42):
    """
    Generate LAN training data by uniformly sampling B parameter sets
    and simulating trials with random coherence levels.

    Returns numpy array of shape (n_parameter_sets * n_samples_per_set, 4):
        columns: [B, coherence, rt, response]

    Note: coherence (absolute value) is included as a trial-level covariate
    so the LAN can learn how coherence modulates the RT/choice distribution.
    """
    np.random.seed(seed)

    all_rows = []
    total = n_parameter_sets * n_samples_per_set

    for i in range(n_parameter_sets):
        if i % 50 == 0:
            print(f"  Parameter set {i}/{n_parameter_sets} "
                  f"({len(all_rows):,}/{total:,} trials)")

        B = np.random.uniform(*B_RANGE)

        for _ in range(n_samples_per_set):
            coh = float(np.random.choice(COH_LEVELS))
            rt, resp = simulate_one_trial(B, coh)
            all_rows.append([B, abs(coh), rt, float(resp)])

    data = np.array(all_rows, dtype=np.float32)
    print(f"Done. Generated {len(data):,} training examples.")
    return data


if __name__ == "__main__":
    out_dir = pathlib.Path(__file__).resolve().parents[1] / "train_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "brunton_training_data_t750_smallB.npy"

    print("Generating Brunton LAN training data...")
    print(f"  Fixed: lam={LAM}, sigma_s={SIGMA_S}, acc_noise={ACC_NOISE}, T={T}, dt={DT}, noise={NOISE}")
    print(f"  Free:  B in {B_RANGE}")
    print(f"  Covariate: coherence in {COH_LEVELS.tolist()}")

    data = generate_training_data(n_parameter_sets=500, n_samples_per_set=4000)
    np.save(out_path, data)
    print(f"Saved to {out_path}  shape={data.shape}")
    print("Columns: [B, coherence, rt, response]")
