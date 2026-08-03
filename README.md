# brunton_hssm

LAN-based fitting of the Brunton accumulator model to RNN behavioral data.

## what this does

fits the brunton drift-diffusion accumulator to rnn-generated behavior using likelihood approximation networks (LANs). the pipeline has 5 steps:

1. **generate training data** — simulate (rt, choice) from brunton model across many (lam, B) parameter combinations
2. **compute kde labels** — approximate log-likelihoods using kernel density estimation
3. **train lan** — train a small MLP to predict log-likelihoods, export to ONNX
4. **fit lan** — use trained LAN in HSSM to get posterior estimates of (lam, B) from RNN data
5. **param recovery** — simulate from known parameters, fit, check recovery

## structure

src/ scripts for each pipeline step
train_data/ training data, trained model, fit results (not tracked by git)
bash/ SLURM job scripts for oscar
cluster/log/ job logs (not tracked by git)


## how to run

run steps in order on oscar:
```bash
sbatch bash/run_generate.sh
sbatch bash/run_labels.sh
sbatch bash/run_train.sh
sbatch bash/run_fit.sh      # fits nxx1, seed=42, gain=1.0
sbatch bash/run_recovery.sh # small test run
parameters

brunton model free parameters:

lam ∈ (-0.5, 0.5): memory parameter (negative=leaky, zero=perfect, positive=unstable)
B ∈ (0.5, 3.0): decision bound height

fixed: sigma_s=1.0, acc_noise=0.1, T=40 timesteps, noise=0.5
