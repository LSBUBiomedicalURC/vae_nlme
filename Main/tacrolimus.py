"""
tacrolimus.py
-------------

Parameter estimation + Covariate selection of a nonlinear mixed effects
model using a variational autoencoder (VAE) for the real Tacrolimus PK
dataset from conditioning_limits_PK (Case Study 3).

Mirrors Main/theophylline.py's structure (dimensions, prior h/h_inverse,
Encoder/Decoder init, initalize_C, burn-in, training loop with
covariate_selection, log-likelihood, printOutput) but is wired to:
  - data_loading.load_two_file_wide (generalized loader, n_cov read at runtime
    from the covariates CSV header -- NOT hardcoded),
  - VAE/decoder_tacrolimus.Decoder_tacrolimus: single fixed dose D=300,
    fixed ka=0.502, fixed absorption lag t0=0.346 (ground truth confirmed by
    the user -- NOT a repeated q12h-dosing model; only ke and V are
    subject-specific, so z_dim=2),
  - functions_tacrolimus.initalize_C / EmpiricalBayesEstimate (generalized
    log-linear covariate design that auto-detects categorical covariates,
    instead of functions_theo.initalize_C's hardcoded Sex special-case).

Ground truth (per conditioning_limits_PK/config.py, verified against the
real CSVs in this script's own smoke test):
    N_TRUE = 4 informative covariates = first 4 covariate columns
             (Age, Hemoglobin, Albumin, SNP)
    Next 7 columns = clinical noise (Sex, Weight, Race1..Race5 one-hot)
    Remaining 250 columns = RNAseq pure-noise covariates (RNASEQ_NOISE_DIM=250)
    Total raw covariate columns = 261 (N_COND_TOTAL in config.py)

Per-subject dose: every subject receives the SAME single dose D=300 (ground
truth confirmed by the user). Tacrolimus_parameters.csv (mg_twice_daily_dose,
CL, V, ke, ...) is NOT the generator behind Tacrolimus_data.csv -- it is a
different/legacy simulation and is not used here.
"""
#########################################################
# Import
#########################################################
import argparse
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import torch

from functions import *
from functions_tacrolimus import initalize_C, EmpiricalBayesEstimate
from VAE.decoder_tacrolimus import Decoder_tacrolimus
from VAE.encoder import *
from ParaUpdate.pop_parameter import *
from data_loading import load_two_file_wide

#########################################################
# CLI
#########################################################
pk_data_dir_default = os.path.join(parent, 'Data') + os.sep
#pk_data_dir_default = r'..\Data'
# NOTE: vae_nlme-main\Data also has Tacrolimus_data.csv / Tacrolimus_covariates.csv, but that copy's
# covariates file only has 11 columns (no RNAseq noise block) -- the conditioning_limits_PK copy
# (261 covariates) is required for the uninformative-covariate stress test, so it's the default.

parser = argparse.ArgumentParser(description="VAE-nlme Tacrolimus case study")
parser.add_argument('--data_dir', default=pk_data_dir_default,
                    help="Directory containing Tacrolimus_data*.csv / Tacrolimus_covariates*.csv / "
                         "Tacrolimus_parameters.csv")
parser.add_argument('--size', default='', choices=['', '_1000', '_10000'],
                    help="Dataset size suffix: '' (10000, default Tacrolimus_data.csv), "
                         "'_1000' (Tacrolimus_data_1000.csv), '_10000' (Tacrolimus_data_10000.csv)")
parser.add_argument('--iters', type=int, default=300)
parser.add_argument('--iters_burn_in', type=int, default=100)
parser.add_argument('--smoke_test', action='store_true',
                    help="Only build Encoder/Decoder/pop_parameter and run initalize_C "
                         "(model-construction smoke test), do not train.")
parser.add_argument('--solver', default='GUROBI',
                    help="cvxpy solver for pop_parameter's covariate-selection MIQP step "
                         "(default: GUROBI). See solver_utils.py for why most free solvers "
                         "cannot solve this problem class (it's mixed-integer, not just QP); "
                         "SCIP is the one untested-but-plausible free alternative.")
parser.add_argument('--allow_incompatible_solver', action='store_true',
                    help="Skip solver_utils's known-incompatible-solver guard.")
parser.add_argument('--n_batch', type=int, default=None,
                    help="Use only the first n_batch subjects (default: all). Useful to keep "
                         "pop_parameter's covariate-selection problem tractable for a given "
                         "Gurobi license / solver (see solver_utils.py).")
parser.add_argument('--n_cov', type=int, default=None,
                    help="Use only the first n_cov covariate columns (default: all 261 -- the "
                         "first 4 are the true covariates Age/Hemoglobin/Albumin/SNP, the rest "
                         "are clinical-noise then RNAseq-noise columns, in that order -- see "
                         "conditioning_limits_PK/config.py). Lets you sweep the number of "
                         "uninformative covariates directly from real data.")
parser.add_argument('--skip_ll', action='store_true',
                    help="Skip log-likelihood / EBE computation (for faster stress-test sweeps "
                         "where only covariate-selection metrics are needed).")
parser.add_argument('--results_json', default=None,
                    help="If given, write a small JSON summary (selected covariates, "
                         "population estimates, OFV) to this path -- for automated stress-test "
                         "driving without parsing stdout (see Main/run_stress_test.py).")
parser.add_argument('--plot_dir', default=None,
                    help="Directory to save convergence figures (PDF). Files are named "
                         "tacrolimus_ncov{n_cov}_popParam.pdf / _beta.pdf. "
                         "Default: None (no figures saved in batch/subprocess mode).")
parser.add_argument('--standardise_C', action='store_true',
                    help="Divide each covariate column of C_regression by its across-subject "
                         "standard deviation before passing it to pop_parameter. Isolates "
                         "whether selection instability with growing n_cov is a scale effect "
                         "vs a penalty-dimensionality effect. Default: off.")
parser.add_argument('--miqp_every', type=int, default=1,
                    help="Run the covariate-selection MIQP only every N-th training iteration "
                         "(default: 1 = every iteration). Increasing this (e.g. 5 or 10) gives "
                         "a proportional speedup at the cost of coarser selection updates, which "
                         "is useful for slow solvers (SCIP) or high z_dim (Paclitaxel z_dim=6).")
parser.add_argument('--seed', type=int, default=1,
                    help="Random seed for torch and numpy (default: 1, matching the original "
                         "hardcoded value). Pass different values to replicate across seeds.")
args, _ = parser.parse_known_args()

from solver_utils import set_pop_parameter_solver
set_pop_parameter_solver(args.solver, allow_incompatible=args.allow_incompatible_solver)

# Pick a manual seed for randomization
torch.manual_seed(args.seed)
np.random.seed(args.seed)

#########################################################
# Load Data
#########################################################
conc_path = os.path.join(args.data_dir, f'Tacrolimus_data{args.size}.csv')
cov_path = os.path.join(args.data_dir, f'Tacrolimus_covariates{args.size}.csv')

# Number of timepoints (t=0..48h, 49 points) is read at runtime from the CSV
# itself (data_loading.load_two_file_wide infers it from column count), but
# the *sample times* convention (t = 0, 1, ..., DATA_DIM-1) matches
# conditioning_limits_PK/config.py's SAMPLE_TIMES = [float(t) for t in range(DATA_DIM)].
with open(conc_path) as _f:
    _n_time_cols = len(_f.readline().strip().split(',')) - 1
sample_times = [float(t) for t in range(_n_time_cols)]

# Ground truth: every subject gets the SAME single dose D=300 (confirmed by
# the user) -- not a per-subject dose, so we pass a scalar.
(data, data_in, lengths, dose, weight_pop, covariates, covariates_in,
 covariate_names, n_cov) = load_two_file_wide(
    conc_path, cov_path, id_col='ID',
    dose=300.0,
    sample_times=sample_times,
)

#########################################################
# Subset subjects / covariates (--n_batch / --n_cov)
#########################################################
if args.n_batch is not None:
    nb = min(args.n_batch, data.shape[0])
    data, data_in, lengths, dose = data[:nb], data_in[:nb], lengths[:nb], dose[:nb]
    covariates, covariates_in = covariates[:nb], covariates_in[:nb]
if args.n_cov is not None:
    nc = min(args.n_cov, n_cov)
    data = data[:, :, :3 + nc]
    covariates, covariates_in = covariates[:, :nc], covariates_in[:, :nc]
    covariate_names = covariate_names[:nc]
    n_cov = nc
    weight_pop = covariates[:, 0].mean() if nc > 0 else weight_pop

#########################################################
# Dimensions
#########################################################
nbatch = data.shape[0]         # Number of individuals
x_dim = 2                     # Dimensionality of the observations
z_dim = 2                     # Number of individual PK parameters (ke, V) --
                               # ka=0.502 and lag t0=0.346 are FIXED, not estimated.
# n_cov is read at runtime from the covariates CSV header (261 for the full
# Tacrolimus_covariates.csv, per conditioning_limits_PK/config.py's
# N_COND_TOTAL) -- never hardcoded.
N_TRUE = 4  # ground-truth informative covariates: Age, Hemoglobin, Albumin, SNP
            # (first 4 columns -- see conditioning_limits_PK/config.py)

#########################################################
# Prior distribution
#########################################################
h = lambda x: x.exp()
h_inverse = lambda x: x.log()

#########################################################
# Initialization LSTM Encoder
#########################################################
h_dim = 25
# NOTE: LSTM_Encoder applies h_inverse internally to mu0 -- mu0 must be passed
# in RAW (linear) scale here, matching Main/theophylline.py's
# mu0 = torch.tensor([1, 0.5, 15]) convention.
mu0    = torch.tensor([0.015, 3700.0])  # population priors for [ke (h⁻¹), V (L)];
                                        # true averages: ke≈0.015, V≈3700 (D=300 mg, conc in mg/L)
sigma0 = torch.tensor([1e-1, 1e-1]).log()
Encoder = LSTM_Encoder(x_dim, h_dim, z_dim, nbatch, n_cov, mu0, sigma0, h_inverse)

#########################################################
# Initialization Decoder
#########################################################
Decoder = lambda z_normal, time, h: Decoder_tacrolimus(z_normal, time, h)

#########################################################
# Full Covariate Model
#########################################################
C, C_regression = initalize_C(nbatch, z_dim, n_cov, covariates, weight_pop)
if args.standardise_C and n_cov > 0:
    col_std = C_regression[:, :, 1:].std(dim=1, keepdim=True).clamp(min=1e-8)
    C_regression = C_regression.clone()
    C_regression[:, :, 1:] = C_regression[:, :, 1:] / col_std
    for k in range(z_dim):
        start = z_dim + k * n_cov
        C[:, k, start:start + n_cov] = C[:, k, start:start + n_cov] / col_std[k, 0, :]
names_co = [f'beta_{p}_{cov}' for p in ['ke', 'V'] for cov in covariate_names]
penalized_indices = np.arange(1, n_cov + 1)
M = C.shape[2] - z_dim

#########################################################
# Iterations Setup
#########################################################
iters_burn_in = args.iters_burn_in
iters         = args.iters
L_iter        = 5
burn_in_iter  = L_iter * iters_burn_in
# kl_iter and gamma_iter must be <= iters; scale proportionally so short runs
# (e.g. --iters 30) stay numerically stable.  The original paper used iters=300
# with kl_iter=50 (~17%) and gamma_iter=250 (~83%).
# Use original paper values (kl_iter=50, gamma_iter=250) when iters >= 300.
# Scale proportionally for shorter runs so kl/gamma phases stay < iters.
if iters >= 300:
    kl_iter, gamma_iter = 50, 250
else:
    kl_iter    = max(1, min(50,  int(round(0.17 * iters))))
    gamma_iter = max(1, min(iters - 1, int(round(0.83 * iters))))
alpha_KL   = torch.linspace(0.01, 1, kl_iter)
smoothing  = False

#########################################################
# Initialize population parameters updates
#########################################################
pop = pop_parameter(z_dim, nbatch, gamma_iter, data, C, C_regression,
                    C[:, :z_dim, :z_dim], penalized_indices, n_cov, kl_iter, lengths, 2)

#########################################################
# Smoke test stop: do not train, just confirm shapes wire correctly
#########################################################
if args.smoke_test:
    print("Smoke test: Encoder, Decoder, pop_parameter, initalize_C constructed successfully.")
    print(f"  nbatch={nbatch}  n_cov={n_cov}  z_dim={z_dim}  M={M}")
    print(f"  data.shape={tuple(data.shape)}  C.shape={tuple(C.shape)}  "
          f"C_regression.shape={tuple(C_regression.shape)}")
    # Run one Encoder forward pass and one Decoder forward pass to verify wiring.
    z_normal, mu, L, log_sigma, eps = Encoder(data_in, covariates_in, lengths)
    pred_x = Decoder(z_normal, data[:, :, 0], h)
    print(f"  Encoder output z_normal.shape={tuple(z_normal.shape)}  "
          f"Decoder output pred_x.shape={tuple(pred_x.shape)}")
    print(f"  pred_x has NaN: {torch.isnan(pred_x).any().item()}")
    z_pop, omega_pop, a, mu_smooth = pop.update_pop(mu.detach(), L.detach(), pred_x.detach(), 1,
                                                    covariate_selection=True, update_pop=True)
    print(f"  pop.update_pop ran successfully. z_pop.shape={tuple(z_pop.shape)}  "
          f"omega_pop.shape={tuple(omega_pop.shape)}")
    sys.exit(0)

#########################################################
# Burn in
#########################################################
(Encoder, optimizer, pred_x, mu, L, a, b, z_pop_iter_bi, omega_pop_iter_bi,
 a_iter_bi, elbo_iter_bi) = initalizeEncoder(
    iters_burn_in, L_iter, Encoder, Decoder, data, data_in, z_dim, covariates_in, lengths, h, pop)
pred_x_mean = pred_x.detach()

#########################################################
# Setup Optimizer
#########################################################
optimizer.param_groups[0]['lr'] = 5e-3

#########################################################
# Initialize tensors to store results
#########################################################
z_pop_iter = torch.zeros(iters, z_dim + M)
omega_pop_iter = torch.zeros(iters, z_dim)
a_iter = torch.zeros(iters)
Elbo_iter = torch.zeros(iters)

#########################################################
# Training
#########################################################
print('')
print('#############################################')
print('Training VAE (Tacrolimus)')
z_pop_cached = None  # last full-length z_pop from a MIQP call; reused on non-MIQP iterations
for iter in range(1, iters + 1):
    if iter > 1:
        pred_x_mean = data_matrix[L_iter - 2:L_iter].mean(dim=0)
    do_miqp = (iter == 1) or (iter % args.miqp_every == 0)
    if iter > gamma_iter:
        z_pop_ret, omega_pop, a, _ = pop.update_pop(mu.detach(), L.detach(), pred_x_mean, iter,
                                                    covariate_selection=do_miqp, smoothing=True, update_pop=True)
    else:
        z_pop_ret, omega_pop, a, mu_smooth = pop.update_pop(mu.detach(), L.detach(), pred_x_mean, iter,
                                                            covariate_selection=do_miqp, update_pop=True)
    if do_miqp:
        z_pop_cached = z_pop_ret
    z_pop = z_pop_cached

    data_matrix = torch.zeros(L_iter, nbatch, data.shape[1], 1)
    ELBO = torch.zeros(L_iter)
    for l in range(L_iter):
        z_normal, mu, L, log_sigma, eps = Encoder(data_in, covariates_in, lengths)
        pred_x = Decoder(z_normal, data[:, :, 0], h)
        data_matrix[l] = pred_x.clone().detach()

        z_pop_batch = torch.zeros(nbatch, z_dim)
        for i in range(nbatch):
            z_pop_batch[i] = torch.matmul(C[i], z_pop)
        p_x_z = p_x_z_compute(data[:, :, 1].view(data.shape[0], data.shape[1], 1), pred_x, [a, b], lengths)
        p_z = p_z_compute(z_normal, z_pop_batch, omega_pop)
        q_z = q_z_x_compute(eps, torch.diagonal(L, dim1=1, dim2=2))
        DKL = p_z - q_z
        if iter < kl_iter:
            elbo = p_x_z + alpha_KL[iter] * DKL
        else:
            elbo = p_x_z + DKL
        ELBO[l] = p_x_z + DKL

        if torch.isfinite(elbo):
            elbo.backward()
            torch.nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    if iter < gamma_iter:
        print(f"Iteration {iter}/{gamma_iter}", end="\r")
    else:
        print(f"Iteration {iter - gamma_iter}/{iters - gamma_iter} (smoothing)", end="\r")

    z_pop_iter[iter - 1] = torch.hstack((h(z_pop[:z_dim]), z_pop[z_dim:]))
    omega_pop_iter[iter - 1] = omega_pop.sqrt()
    a_iter[iter - 1] = a
    Elbo_iter[iter - 1] = ELBO.mean()

#########################################################
# Compute log likelihood
#########################################################
LL_lin_mu = LL_lin = LL_is = float('nan')
if not args.skip_ll:
    print('')
    print('#############################################')
    print('Compute log likelihood and empirical Bayes estimate')
    LL_lin_mu = LogLikelihood_linearization(z_pop, omega_pop, [a, b], data, mu_smooth, C, h,
                                            data[:, :, 0], lengths, Decoder)
    phi_opt = torch.zeros(nbatch, z_dim)
    for i in range(nbatch):
        phi_opt[i, :] = EmpiricalBayesEstimate(data[i, :, :], z_pop, omega_pop, mu[i], [a, b], C[i], h)
    LL_lin = LogLikelihood_linearization(z_pop, omega_pop, [a, b], data, phi_opt, C, h,
                                         data[:, :, 0], lengths, Decoder)
    LL_is, _ = LogLikelihood_sample(1000, z_pop, omega_pop, [a, b], data, mu, L, C, h,
                                    data[:, :, 0], lengths, Decoder)
    print('#############################################')
else:
    print('(Skipping log-likelihood / EBE computation -- pass without --skip_ll for full output)')

#########################################################
# Output results
#########################################################
z_pop_iter = torch.vstack((torch.hstack((z_pop_iter_bi, torch.zeros(iters_burn_in, M))), z_pop_iter))
omega_pop_iter = torch.vstack((omega_pop_iter_bi, omega_pop_iter))
a_iter = torch.hstack((a_iter_bi, a_iter))

print('')
print('#############################################')
print('ESTIMATION OF THE POPULATION PARAMETERS (Tacrolimus)')
print('#############################################')
z_pop_h = h(z_pop)
print(f'{"ke_pop:":<15} {z_pop_h[0]:>12.6f}')
print(f'{"V_pop:":<15} {z_pop_h[1]:>12.4f}')
count = 0
for k in range(z_dim, len(z_pop)):
    if z_pop[k] != 0:
        print(f'{names_co[k - z_dim] + ":":<25} {z_pop[k]:>12.4f}')
        count += 1
print('')
print(f'Number of selected (nonzero) covariate effects: {count} / {M}')
print(f'-2 Log likelihood (linearization, OFV): {2 * LL_lin:.2f}')
print(f'-2 Log likelihood (importance sampling, OFV): {2 * LL_is:.2f}')

if args.results_json:
    import json
    beta = z_pop[z_dim:].detach().numpy().reshape(z_dim, n_cov) if n_cov > 0 else np.zeros((z_dim, 0))
    selected = (np.abs(beta) > 1e-9).any(axis=0).tolist() if n_cov > 0 else []
    with open(args.results_json, 'w') as f:
        json.dump(dict(
            dataset='tacrolimus', n_batch=nbatch, n_cov=n_cov, z_dim=z_dim, seed=args.seed,
            n_true_cov=4, n_observations=int(lengths.sum().item()),
            covariate_names=covariate_names, selected=selected, n_selected=count, M=M,
            z_pop_structural=z_pop_h[:z_dim].tolist(),
            beta=beta.tolist(),
            beta_max=float(np.abs(beta).max()) if beta.size else 0.0,
            big_m=100, standardise_C=args.standardise_C, miqp_every=args.miqp_every,
            ofv_lin=(None if LL_lin != LL_lin else float(2 * LL_lin)), ofv_is=(None if LL_is != LL_is else float(2 * LL_is)),
        ), f, indent=2)

if args.plot_dir:
    import os as _os
    from visualization import plotConvergence_pop, plotConvergence_covariate, plotConvergence_beta
    _os.makedirs(args.plot_dir, exist_ok=True)
    _tag = f'tacrolimus_ncov{n_cov}'
    plotConvergence_pop(
        Elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        param_names=['ke', 'V'],
        save_path=_os.path.join(args.plot_dir, f'{_tag}_popParam.pdf'))
    plotConvergence_covariate(
        z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        z_dim=z_dim, n_cov=n_cov, param_names=['ke', 'V'],
        cov_names=covariate_names, max_cov=8,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_covariate.pdf'))
    plotConvergence_beta(
        z_pop, z_dim, n_cov, ['ke', 'V'], covariate_names,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_beta.pdf'))
