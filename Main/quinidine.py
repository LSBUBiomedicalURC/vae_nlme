"""
quinidine.py
------------

Parameter estimation + Covariate selection of a nonlinear mixed effects
model using a variational autoencoder (VAE) for the real Quinidine dense PK
dataset (Data/Quinidine_data.csv + Quinidine_covariates.csv, committed here
as a snapshot copy of the source shared with the CVAE-LASSO project at
cvae_lasso/Data_real/ -- re-sync manually if that source is updated).

Mirrors Main/tacrolimus.py's structure but is wired to:
  - data_loading.load_two_file_wide, with sample times parsed from the
    'CtHH' column headers (e.g. 'Ct05' -> 0.5h, 'Ct240' -> 24.0h),
  - VAE/decoder_quinidine.Decoder_quinidine: 2-compartment first-order
    absorption model with lag, z_dim=6, parameters
    [Tlag, ka, Cl, V1, Q, V2] -- matches the parameter set reported by
    Ayral et al. 2021 (COSSAC paper, Table 1) for "Quinidine PK",
  - functions_quinidine.initalize_C / EmpiricalBayesEstimate.

Ground truth / structural model notes
--------------------------------------
Real covariates (n_true_cov = 10, per Ayral 2021 Table 1's "7 - SEX, RACE,
logAGE, logHT, logWT, logDIABP, logSYSBP", with RACE one-hot-expanded to
RACE_1..4): SEX, AGE, HEIGHT, WEIGHT, SYSBP, DIABP, RACE_1, RACE_2, RACE_3,
RACE_4. Remaining covariate columns are TCGA RNAseq pure-noise covariates.

DOSE: confirmed by the user as a single fixed dose of 400 (mg) for every
subject (--dose, default 400.0) -- unlike Tacrolimus's fixed D=300 or
Paclitaxel's BSA-derived per-subject dose, this is a constant across all
subjects, not per-subject-varying.
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
from functions_quinidine import initalize_C, EmpiricalBayesEstimate
from VAE.decoder_quinidine import Decoder_quinidine
from VAE.encoder import *
from ParaUpdate.pop_parameter import *
from data_loading import load_two_file_wide

#########################################################
# CLI
#########################################################
pk_data_dir_default = os.path.join(parent, 'Data') + os.sep

parser = argparse.ArgumentParser(description="VAE-nlme Quinidine dense PK case study")
parser.add_argument('--data_dir', default=pk_data_dir_default,
                    help="Directory containing Quinidine_data.csv / Quinidine_covariates.csv")
parser.add_argument('--dose', type=float, default=400.0,
                    help="Constant dose for all subjects (default 400.0mg, confirmed by "
                         "the user -- same dose for every subject, not per-subject-varying).")
parser.add_argument('--iters', type=int, default=300)
parser.add_argument('--iters_burn_in', type=int, default=100)
parser.add_argument('--smoke_test', action='store_true',
                    help="Only build Encoder/Decoder/pop_parameter and run initalize_C "
                         "(model-construction smoke test), do not train.")
parser.add_argument('--solver', default='GUROBI',
                    help="cvxpy solver for pop_parameter's covariate-selection MIQP step.")
parser.add_argument('--allow_incompatible_solver', action='store_true')
parser.add_argument('--n_batch', type=int, default=None,
                    help="Use only the first n_batch subjects (default: all).")
parser.add_argument('--n_cov', type=int, default=None,
                    help="Use only the first n_cov covariate columns (default: all -- the "
                         "first 10 are the true covariates SEX/AGE/HEIGHT/WEIGHT/SYSBP/DIABP/"
                         "RACE_1..4, the rest are RNAseq-noise columns).")
parser.add_argument('--skip_ll', action='store_true')
parser.add_argument('--results_json', default=None)
parser.add_argument('--plot_dir', default=None)
parser.add_argument('--standardise_C', action='store_true')
parser.add_argument('--miqp_every', type=int, default=1)
parser.add_argument('--seed', type=int, default=1,
                    help="Random seed for torch and numpy (default: 1).")
args, _ = parser.parse_known_args()

from solver_utils import set_pop_parameter_solver
set_pop_parameter_solver(args.solver, allow_incompatible=args.allow_incompatible_solver)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#########################################################
# Load Data
#########################################################
conc_path = os.path.join(args.data_dir, 'Quinidine_data.csv')
cov_path  = os.path.join(args.data_dir, 'Quinidine_covariates.csv')

with open(conc_path) as _f:
    _header = _f.readline().strip().split(',')
# column names 'CtHH' encode sample time in tenths of an hour, e.g.
# 'Ct05' -> 0.5h, 'Ct240' -> 24.0h
sample_times = [float(c[2:]) / 10.0 for c in _header[1:]]

(data, data_in, lengths, dose, weight_pop, covariates, covariates_in,
 covariate_names, n_cov) = load_two_file_wide(
    conc_path, cov_path, id_col='ID',
    dose=args.dose,
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
nbatch = data.shape[0]
x_dim = 2
z_dim = 6  # [Tlag, ka, Cl, V1, Q, V2]
N_TRUE = 10  # SEX, AGE, HEIGHT, WEIGHT, SYSBP, DIABP, RACE_1..4 (Ayral 2021 Table 1)

#########################################################
# Prior distribution
#########################################################
h = lambda x: x.exp()
h_inverse = lambda x: x.log()

#########################################################
# Initialization LSTM Encoder
#########################################################
h_dim = 25
# Population priors, order [Tlag, ka, Cl, V1, Q, V2] -- reasonable
# literature-informed magnitudes for quinidine (Tlag~0.3h, ka~2/h,
# Cl~17 L/h, V1~40L, Q~15 L/h, V2~100L), NOT user-confirmed ground truth
# (this is real, not simulated, data).
mu0    = torch.tensor([0.3, 2.0, 17.0, 40.0, 15.0, 100.0])
sigma0 = torch.tensor([1e-1] * z_dim).log()
Encoder = LSTM_Encoder(x_dim, h_dim, z_dim, nbatch, n_cov, mu0, sigma0, h_inverse)

#########################################################
# Initialization Decoder
#########################################################
Decoder = lambda z_normal, time, h: Decoder_quinidine(z_normal, time, h, dose)

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
PARAM_NAMES = ['Tlag', 'ka', 'Cl', 'V1', 'Q', 'V2']
names_co = [f'beta_{p}_{cov}' for p in PARAM_NAMES for cov in covariate_names]
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
print('Training VAE (Quinidine)')
z_pop_cached = torch.zeros(z_dim + z_dim * n_cov)
for iter in range(1, iters + 1):
    if iter > 1:
        pred_x_mean = data_matrix[L_iter - 2:L_iter].mean(dim=0)
    do_miqp = (iter % args.miqp_every == 0)
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
        phi_opt[i, :] = EmpiricalBayesEstimate(data[i, :, :], z_pop, omega_pop, mu[i], [a, b],
                                               C[i], h, dose[i])
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
print('ESTIMATION OF THE POPULATION PARAMETERS (Quinidine)')
print('#############################################')
z_pop_h = h(z_pop)
for p_i, p_name in enumerate(PARAM_NAMES):
    print(f'{p_name + "_pop:":<15} {z_pop_h[p_i]:>12.6f}')
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
            dataset='quinidine', n_batch=nbatch, n_cov=n_cov, z_dim=z_dim, seed=args.seed,
            n_true_cov=N_TRUE, n_observations=int(lengths.sum().item()),
            covariate_names=covariate_names, selected=selected, n_selected=count, M=M,
            z_pop_structural=z_pop_h[:z_dim].tolist(),
            beta=beta.tolist(),
            beta_max=float(np.abs(beta).max()) if beta.size else 0.0,
            big_m=100, standardise_C=args.standardise_C, miqp_every=args.miqp_every,
            dose=args.dose,
            ofv_lin=(None if LL_lin != LL_lin else float(2 * LL_lin)),
            ofv_is=(None if LL_is != LL_is else float(2 * LL_is)),
        ), f, indent=2)

if args.plot_dir:
    import os as _os
    from visualization import plotConvergence_pop, plotConvergence_covariate, plotConvergence_beta
    _os.makedirs(args.plot_dir, exist_ok=True)
    _tag = f'quinidine_ncov{n_cov}'
    plotConvergence_pop(
        Elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        param_names=PARAM_NAMES,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_popParam.pdf'))
    plotConvergence_covariate(
        z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        z_dim=z_dim, n_cov=n_cov, param_names=PARAM_NAMES,
        cov_names=covariate_names, max_cov=8,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_covariate.pdf'))
    plotConvergence_beta(
        z_pop, z_dim, n_cov, PARAM_NAMES, covariate_names,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_beta.pdf'))
