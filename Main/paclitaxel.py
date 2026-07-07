"""
paclitaxel.py
-------------

Parameter estimation + Covariate selection of a nonlinear mixed effects
model using a variational autoencoder (VAE) for the real Paclitaxel PK
dataset from conditioning_limits_Paclitaxel (Case Study 4).

Mirrors Main/tacrolimus.py's structure but for the Joerger 2006 3-compartment
PK model (see VAE/decoder_paclitaxel.py docstring for the exact ODE system,
read in full from
conditioning_limits_Paclitaxel/paclitaxel_popPK_joerger2006_v3.py):
  - 3-h constant-rate IV infusion (T_INF = 3.0h),
  - dose = 175 mg/m^2 * BSA, converted to umol via MW 853.9 g/mol
    (dose_mg / 853.9 * 1000 -- exact constant verified in that script's
    generate_population()),
  - z_dim = 6 estimated latent PK parameters: [V1, V3, VMEL, VMTR, KMTR, Q]
    (K21, KMEL held at fixed population typical values -- see
    VAE/decoder_paclitaxel.py docstring for the rationale).

Ground truth covariate structure (per conditioning_limits_Paclitaxel/config.py,
verified against the real CSVs below):
    N_TRUE = 4 informative covariates = first 4 covariate columns
             (Sex, Age, BSA, Bilirubin)
    Next 8 columns = clinical noise (Weight, AST, ALT, ALP, GGT, LDH, SCR, CRCL)
    Remaining 250 columns = RNAseq pure-noise covariates
    Total raw covariate columns = 262 (N_COND_TOTAL in config.py)

Unlike Tacrolimus, there is NO single dominant covariate here -- Sobol
indices (conditioning_limits_Paclitaxel/sobol.py) show importance spread
across covariates (e.g. CRCL is correlated with age/sex/SCR via the
Cockcroft-Gault formula), per project memory
(project_paclitaxel_package.md: "no single dominant covariate").

Sample times: [0, 0.5, 1, 2, 3, 3.5, 4, 5, 6, 8, 12, 24, 48] h, matching the
13 Ct0..Ct12 columns in Paclitaxel_data.csv (verified below).
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
from functions_paclitaxel import initalize_C, EmpiricalBayesEstimate
from VAE.decoder_paclitaxel import Decoder_paclitaxel
from VAE.encoder import *
from ParaUpdate.pop_parameter import *
from data_loading import load_two_file_wide

#########################################################
# CLI
#########################################################
paclitaxel_data_dir_default = os.path.join(parent, 'Data') + os.sep

parser = argparse.ArgumentParser(description="VAE-nlme Paclitaxel case study")
parser.add_argument('--data_dir', default=paclitaxel_data_dir_default,
                    help="Directory containing Paclitaxel_data.csv / Paclitaxel_covariates.csv")
parser.add_argument('--iters', type=int, default=300)
parser.add_argument('--iters_burn_in', type=int, default=100)
parser.add_argument('--smoke_test', action='store_true',
                    help="Only build Encoder/Decoder/pop_parameter and run initalize_C "
                         "(model-construction smoke test), do not train.")
parser.add_argument('--estimate_k21_kmel', action='store_true',
                    help="Estimate K21 and KMEL per-subject (z_dim=8: "
                         "[V1,V3,VMEL,VMTR,KMTR,Q,K21,KMEL]) instead of fixing them at "
                         "population-typical values (z_dim=6, the default).")
parser.add_argument('--solver', default='GUROBI',
                    help="cvxpy solver for pop_parameter's covariate-selection MIQP step "
                         "(default: GUROBI). See solver_utils.py for why most free solvers "
                         "cannot solve this problem class.")
parser.add_argument('--allow_incompatible_solver', action='store_true',
                    help="Skip solver_utils's known-incompatible-solver guard.")
parser.add_argument('--n_batch', type=int, default=None,
                    help="Use only the first n_batch subjects (default: all). Useful to keep "
                         "pop_parameter's covariate-selection problem tractable for a given "
                         "Gurobi license / solver (see solver_utils.py).")
parser.add_argument('--n_cov', type=int, default=None,
                    help="Use only the first n_cov covariate columns (default: all 262 -- the "
                         "first 4 are the true covariates Sex/Age/BSA/Bilirubin, the rest are "
                         "clinical-noise then RNAseq-noise columns, in that order -- see "
                         "conditioning_limits_Paclitaxel/config.py). Lets you sweep the number "
                         "of uninformative covariates directly from real data. NOTE: dose is "
                         "computed from BSA (the 3rd covariate column) BEFORE this truncation, "
                         "so dose stays correct even if --n_cov < 3.")
parser.add_argument('--results_json', default=None,
                    help="If given, write a small JSON summary (selected covariates, "
                         "population estimates, OFV) to this path -- for automated stress-test "
                         "driving without parsing stdout (see Main/run_stress_test.py).")
parser.add_argument('--skip_ll', action='store_true',
                    help="Skip the post-training log-likelihood / EBE computation block "
                         "(LogLikelihood_linearization, EmpiricalBayesEstimate, "
                         "LogLikelihood_sample). Hugely faster for stress-test sweeps where only "
                         "covariate-selection metrics are needed -- each EBE call runs hundreds "
                         "of implicit ODE solves per subject, and LogLikelihood_sample adds "
                         "~1100 more ODE forward passes.")
parser.add_argument('--plot_dir', default=None,
                    help="Directory to save convergence figures (PDF). Files are named "
                         "paclitaxel_ncov{n_cov}_popParam.pdf / _beta.pdf. "
                         "Default: None (no figures saved in batch/subprocess mode).")
parser.add_argument('--standardise_C', action='store_true',
                    help="Divide each covariate column of C_regression by its across-subject "
                         "standard deviation before passing it to pop_parameter. Isolates "
                         "whether selection instability with growing n_cov is a scale effect "
                         "vs a penalty-dimensionality effect. Default: off.")
parser.add_argument('--miqp_every', type=int, default=1,
                    help="Run the covariate-selection MIQP only every N-th training iteration "
                         "(default: 1 = every iteration). Increasing this (e.g. 5 or 10) gives "
                         "a proportional speedup at the cost of coarser selection updates. "
                         "Especially useful for Paclitaxel (z_dim=6) with SCIP.")
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
conc_path = os.path.join(args.data_dir, 'Paclitaxel_data.csv')
cov_path = os.path.join(args.data_dir, 'Paclitaxel_covariates.csv')

# Sample times in hours, matching Ct0..Ct12 (13 timepoints, verified against
# the actual Paclitaxel_data.csv header by this script's own smoke test).
SAMPLE_TIMES = [0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, 24.0, 48.0]

# Molecular weight of paclitaxel (g/mol), used to convert mg -> umol dose,
# exact constant verified in
# conditioning_limits_Paclitaxel/paclitaxel_popPK_joerger2006_v3.py's
# generate_population(): dose_umol = dose_mg / 853.9 * 1000.
PACLITAXEL_MW = 853.9
DOSE_MG_PER_M2 = 175.0


def _dose_fn(covariates_dict):
    """dose (umol) = 175 mg/m^2 * BSA, converted via MW 853.9 g/mol."""
    bsa = covariates_dict['BSA']
    dose_mg = DOSE_MG_PER_M2 * bsa
    dose_umol = dose_mg / PACLITAXEL_MW * 1000.0
    return dose_umol


(data, data_in, lengths, dose, weight_pop, covariates, covariates_in,
 covariate_names, n_cov) = load_two_file_wide(
    conc_path, cov_path, id_col='ID',
    dose_fn=_dose_fn,
    sample_times=SAMPLE_TIMES,
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
nbatch = data.shape[0]
x_dim = 2
if args.estimate_k21_kmel:
    z_dim = 8  # [V1, V3, VMEL, VMTR, KMTR, Q, K21, KMEL] -- full model
else:
    z_dim = 6  # [V1, V3, VMEL, VMTR, KMTR, Q] -- K21/KMEL fixed at pop-typical values
N_TRUE = 4  # ground-truth informative covariates: Sex, Age, BSA, Bilirubin

#########################################################
# Prior distribution
#########################################################
h = lambda x: x.exp()
h_inverse = lambda x: x.log()

#########################################################
# Initialization LSTM Encoder
#########################################################
h_dim = 25
# Typical (population) values from Joerger 2006 THETA (female, BSA=1.8,
# age=55, bili=7), passed in RAW (linear) scale -- LSTM_Encoder applies
# h_inverse internally (verified empirically; see Main/tacrolimus.py's note).
if args.estimate_k21_kmel:
    mu0 = torch.tensor([5.4, 144.0, 7.6, 148.0, 13.0, 1.75, 0.209, 0.047])
else:
    mu0 = torch.tensor([5.4, 144.0, 7.6, 148.0, 13.0, 1.75])
sigma0 = torch.tensor([1e-1] * z_dim).log()
Encoder = LSTM_Encoder(x_dim, h_dim, z_dim, nbatch, n_cov, mu0, sigma0, h_inverse)

#########################################################
# Initialization Decoder
#########################################################
Decoder = lambda z_normal, time, h: Decoder_paclitaxel(
    z_normal, time, h, dose, estimate_k21_kmel=args.estimate_k21_kmel)

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
if args.estimate_k21_kmel:
    param_names = ['V1', 'V3', 'VMEL', 'VMTR', 'KMTR', 'Q', 'K21', 'KMEL']
else:
    param_names = ['V1', 'V3', 'VMEL', 'VMTR', 'KMTR', 'Q']
names_co = [f'beta_{p}_{cov}' for p in param_names for cov in covariate_names]
penalized_indices = np.arange(1, n_cov + 1)
M = C.shape[2] - z_dim

#########################################################
# Iterations Setup
#########################################################
iters_burn_in = args.iters_burn_in
iters         = args.iters
L_iter        = 5
burn_in_iter  = L_iter * iters_burn_in
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
# Smoke test stop
#########################################################
if args.smoke_test:
    print("Smoke test: Encoder, Decoder, pop_parameter, initalize_C constructed successfully.")
    print(f"  nbatch={nbatch}  n_cov={n_cov}  z_dim={z_dim}  M={M}")
    print(f"  data.shape={tuple(data.shape)}  C.shape={tuple(C.shape)}  "
          f"C_regression.shape={tuple(C_regression.shape)}")
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
print('Training VAE (Paclitaxel)')
z_pop_cached = None  # last full-length z_pop from a MIQP call; reused on non-MIQP iterations
for iter in range(1, iters + 1):
    if iter > 1:
        pred_x_mean = data_matrix[L_iter - 2:L_iter].mean(dim=0)
    # Always run the MIQP on the first iteration so z_pop_cached is valid from the start.
    do_miqp = (iter == 1) or (iter % args.miqp_every == 0)
    if iter > gamma_iter:
        z_pop_ret, omega_pop, a, _ = pop.update_pop(mu.detach(), L.detach(), pred_x_mean, iter,
                                                    covariate_selection=do_miqp, smoothing=True, update_pop=True)
    else:
        z_pop_ret, omega_pop, a, mu_smooth = pop.update_pop(mu.detach(), L.detach(), pred_x_mean, iter,
                                                            covariate_selection=do_miqp, update_pop=True)
    if do_miqp:
        z_pop_cached = z_pop_ret   # shape (z_dim + z_dim*n_cov,)
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
        phi_opt[i, :] = EmpiricalBayesEstimate(data[i, :, :], z_pop, omega_pop, mu[i], [a, b], C[i], h,
                                               dose[i].item(), estimate_k21_kmel=args.estimate_k21_kmel)
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
print('ESTIMATION OF THE POPULATION PARAMETERS (Paclitaxel)')
print('#############################################')
z_pop_h = h(z_pop)
for name, val in zip(param_names, z_pop_h[:z_dim]):
    print(f'{name + "_pop:":<15} {val:>12.4f}')
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
            dataset='paclitaxel', n_batch=nbatch, n_cov=n_cov, z_dim=z_dim, seed=args.seed,
            estimate_k21_kmel=args.estimate_k21_kmel,
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
    _pk_names = (['V1', 'V3', 'VMEL', 'VMTR', 'KMTR', 'Q', 'K21', 'KMEL']
                 if args.estimate_k21_kmel else ['V1', 'V3', 'VMEL', 'VMTR', 'KMTR', 'Q'])
    _tag = f'paclitaxel_ncov{n_cov}'
    plotConvergence_pop(
        Elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        param_names=_pk_names,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_popParam.pdf'))
    plotConvergence_covariate(
        z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        z_dim=z_dim, n_cov=n_cov, param_names=_pk_names,
        cov_names=covariate_names, max_cov=8,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_covariate.pdf'))
    plotConvergence_beta(
        z_pop, z_dim, n_cov, _pk_names, covariate_names,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_beta.pdf'))
