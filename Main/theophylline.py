"""
Created on 16.07.2025

@author: Jan Rohleff

Parameter estimation + Covariate selection of a nonlinear mixed effects model using a variational autoencoder (VAE) for theophylline data (Case Study ).
"""
#########################################################
# Import
#########################################################
import argparse
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from functions import *
from functions_theo import *
from VAE.decoder import *
from VAE.encoder import *
from ParaUpdate.pop_parameter import *
from visualization import *
import torch

#########################################################
# CLI -- additive stress-test knobs, default behaviour (n_noise_cov=0)
# is IDENTICAL to the original script.
#########################################################
parser = argparse.ArgumentParser(description="VAE-nlme theophylline case study "
                                              "(+ optional uninformative-covariate stress test)")
parser.add_argument('--n_noise_cov', type=int, default=0,
                    help="Number of extra synthetic uninformative covariates to append to the "
                         "real (weight, sex) covariate set, for stress-testing covariate-selection "
                         "robustness as n_cov grows. 0 (default) reproduces the original script "
                         "exactly.")
parser.add_argument('--noise_source', choices=['iid', 'correlated', 'tgca'], default='iid',
                    help="'iid': pure noise, lognormal(0, noise_sigma) so it stays positive "
                         "(required by functions_theo.initalize_C's log-linear design). "
                         "'correlated': noisy copies of the real `weight` covariate at a given "
                         "SNR (--noise_snr), i.e. partially-informative covariates, to test "
                         "graceful-degradation rather than all-or-nothing selection behaviour. "
                         "'tgca': real RNAseq columns from TGCA_genes.csv (loaded from the Data "
                         "directory). Columns are z-scored then exponentiated so they are positive "
                         "and compatible with initalize_C's log-linear design.")
parser.add_argument('--noise_sigma', type=float, default=0.5,
                    help="Lognormal shape parameter for --noise_source iid.")
parser.add_argument('--noise_snr', type=float, default=1.0,
                    help="Signal-to-noise ratio for --noise_source correlated: weight is "
                         "rescaled to unit variance and corrupted with N(0, 1/snr) noise, then "
                         "exponentiated back to a positive, log-linear-design-compatible scale.")
parser.add_argument('--noise_seed', type=int, default=0)
parser.add_argument('--standardise_C', action='store_true',
                    help="Divide each covariate column of C_regression by its across-subject "
                         "standard deviation before passing it to pop_parameter. This controls "
                         "for scale differences between true and noise covariates and isolates "
                         "whether instability is a penalty-dimensionality effect (vs a scale "
                         "effect). Default: off (reproduces original behaviour).")
parser.add_argument('--miqp_every', type=int, default=1,
                    help="Run the covariate-selection MIQP only every N-th training iteration "
                         "(default: 1 = every iteration). Useful for large n_cov or slow solvers.")
parser.add_argument('--seed', type=int, default=1,
                    help="Random seed for torch and numpy (default: 1, matching the original "
                         "hardcoded value). Pass different values to replicate across seeds.")
parser.add_argument('--solver', default='GUROBI',
                    help="cvxpy solver for pop_parameter's covariate-selection MIQP step. "
                         "See solver_utils.py: most free solvers cannot solve this problem "
                         "class (it's mixed-integer, not just QP).")
parser.add_argument('--allow_incompatible_solver', action='store_true',
                    help="Skip solver_utils's known-incompatible-solver guard.")
parser.add_argument('--n_batch', type=int, default=None,
                    help="Use only the first n_batch subjects (default: all 12). Note theophylline "
                         "only has 12 subjects total, so this mostly matters for fast smoke tests.")
parser.add_argument('--n_cov', type=int, default=None,
                    help="Use only the first n_cov covariate columns AFTER --n_noise_cov injection "
                         "(default: all -- 2 real [weight, sex] + n_noise_cov injected). Mostly "
                         "useful to cap n_cov when sweeping --n_noise_cov to a fixed total.")
parser.add_argument('--iters', type=int, default=None, help="Default: 300.")
parser.add_argument('--iters_burn_in', type=int, default=None, help="Default: 100.")
parser.add_argument('--gamma_iter', type=int, default=None, help="Default: 250.")
parser.add_argument('--skip_ll', action='store_true',
                    help="Skip log-likelihood / EBE computation (for faster stress-test sweeps "
                         "where only covariate-selection metrics are needed).")
parser.add_argument('--results_json', default=None,
                    help="If given, write a small JSON summary (selected covariates, "
                         "population estimates, OFV) to this path -- for automated stress-test "
                         "driving without parsing stdout (see Main/run_stress_test.py).")
parser.add_argument('--plot_dir', default=None,
                    help="Directory to save convergence figures (PDF). Files are named "
                         "theophylline_nnoise{n_noise_cov}_popParam.pdf / _beta.pdf. "
                         "Default: None (no figures saved in batch/subprocess mode).")
args, _ = parser.parse_known_args()

from solver_utils import set_pop_parameter_solver
set_pop_parameter_solver(args.solver, allow_incompatible=args.allow_incompatible_solver)

# Pick a manual seed for randomization
torch.manual_seed(args.seed)
import numpy as np; np.random.seed(args.seed)

#########################################################
# Load Data
#########################################################
path = os.path.join(parent, 'Data') + os.sep  # was 'VAE_nlme/Data/', which doesn't match this
                                              # checkout's folder name (vae_nlme-main) and was
                                              # never runnable from here; use the script's own
                                              # location instead (same fix as Main/tacrolimus.py
                                              # and Main/paclitaxel.py already use).
data, data_in, lengths, dose, weight_pop, covariates, covariates_in = load_data(path + 'theophylline_data.pt')

#########################################################
# Stress test: inject extra uninformative / partially-informative covariates
#########################################################
noise_names = []
if args.n_noise_cov > 0:
    nbatch_tmp = covariates.shape[0]
    g = torch.Generator().manual_seed(args.noise_seed)
    if args.noise_source == 'iid':
        # Positive (lognormal) so functions_theo.initalize_C's
        # log(cov / weight_pop) design term stays well-defined, like the
        # real (positive) `weight` covariate it appends after.
        noise_raw = torch.exp(torch.randn(nbatch_tmp, args.n_noise_cov, generator=g) * args.noise_sigma)
        noise_names = [f'noise_iid_{i}' for i in range(args.n_noise_cov)]
    elif args.noise_source == 'tgca':
        tgca_path = path + "TGCA_genes.csv"
        if not os.path.exists(tgca_path):
            raise FileNotFoundError(
                f"--noise_source tgca requires {tgca_path}. "
                "Place TGCA_genes.csv (subjects × genes, one header row) in the Data directory.")
        data_rnaseq = np.loadtxt(tgca_path, delimiter=",", skiprows=1)
        if data_rnaseq.ndim == 1:
            data_rnaseq = data_rnaseq[:, None]
        n_rows, n_cols = data_rnaseq.shape
        if n_rows < nbatch_tmp:
            raise ValueError(
                f"TGCA_genes.csv has only {n_rows} rows but {nbatch_tmp} subjects are needed. "
                "Use --n_batch to reduce the subject count.")
        if n_cols < args.n_noise_cov:
            raise ValueError(
                f"TGCA_genes.csv has only {n_cols} gene columns but --n_noise_cov={args.n_noise_cov} "
                "was requested.")
        raw = data_rnaseq[:nbatch_tmp, :args.n_noise_cov].astype(np.float32)
        # z-score each gene column, then exponentiate so values are positive and
        # compatible with initalize_C's log(cov/weight_pop) log-linear design.
        # exp(z-score) is centred at 1.0 (exp(0)=1) and always > 0, exactly as
        # the real `weight` covariate is after log-normalisation.
        raw = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)
        noise_raw = torch.from_numpy(np.exp(raw))
        noise_names = [f'tgca_gene_{i}' for i in range(args.n_noise_cov)]

    else:  # 'correlated': noisy, attenuated copies of the real weight covariate
        weight_raw = covariates[:, 0]
        weight_z = (weight_raw - weight_raw.mean()) / weight_raw.std()
        eps = torch.randn(nbatch_tmp, args.n_noise_cov, generator=g) / max(args.noise_snr, 1e-6) ** 0.5
        noise_raw = torch.exp(weight_z.unsqueeze(1) + eps)
        noise_names = [f'noise_corr_snr{args.noise_snr}_{i}' for i in range(args.n_noise_cov)]

    covariates = torch.cat([covariates, noise_raw], dim=1)
    noise_in = noise_raw.clone()
    noise_in = (noise_in - noise_in.mean(dim=0)) / noise_in.std(dim=0)
    covariates_in = torch.cat([covariates_in, noise_in], dim=1)

#########################################################
# Subset subjects / covariates (--n_batch / --n_cov)
#########################################################
all_cov_names_pre_subset = ['weight', 'sex'] + noise_names
if args.n_batch is not None:
    nb = min(args.n_batch, data.shape[0])
    data, data_in, lengths, dose = data[:nb], data_in[:nb], lengths[:nb], dose[:nb]
    covariates, covariates_in = covariates[:nb], covariates_in[:nb]
if args.n_cov is not None:
    nc = min(args.n_cov, covariates.shape[1])
    covariates, covariates_in = covariates[:, :nc], covariates_in[:, :nc]
    all_cov_names_pre_subset = all_cov_names_pre_subset[:nc]
    weight_pop = covariates[:, 0].mean() if nc > 0 else weight_pop

#########################################################
# Dimensions
#########################################################
nbatch = data.shape[0]       # Number of individuals
x_dim  = 2                   # Dimensionality of the observations
z_dim  = 3                   # Number of individual parameters (ka, ke, V)
n_cov  = covariates.shape[1] # Number of covariates (real + injected noise, if any)
#########################################################
# Prior distribution
#########################################################
h         = lambda x: x.exp()
h_inverse = lambda x: x.log()
#########################################################
# Initialization LSTM Encoder
#########################################################
h_dim   = 25                                     # Hidden dimension of the LSTM
sigma0  = torch.tensor([1e-2, 5e-3, 1e-1]).log() # Initial standard deviation of the posterior distribution      
mu0     = torch.tensor([1, 0.5, 15])             # Initial mean of the posterior distribution
Encoder = LSTM_Encoder(x_dim, h_dim, z_dim, nbatch, n_cov, mu0, sigma0, h_inverse)
#Encoder = torch.compile(Encoder)
#########################################################
# Initialization Decoder
#########################################################
Decoder = lambda z_normal, time, h: Decoder_theophylline(z_normal, time, h, dose)
#########################################################
# Full Covariate Model
#########################################################
C, C_regression   = initalize_C(nbatch, z_dim, n_cov, covariates, weight_pop)
if args.standardise_C and n_cov > 0:
    # Divide covariate columns (indices 1..) of C_regression by their across-subject SD.
    # The intercept column (index 0) is left unchanged.
    col_std = C_regression[:, :, 1:].std(dim=1, keepdim=True).clamp(min=1e-8)
    C_regression = C_regression.clone()
    C_regression[:, :, 1:] = C_regression[:, :, 1:] / col_std
    # Mirror the same scaling in C so the encoder's covariate design is consistent
    for k in range(z_dim):
        start = z_dim + k * n_cov
        C[:, k, start:start + n_cov] = C[:, k, start:start + n_cov] / col_std[k, 0, :]
names_co          = [f'beta_{p}_{cov}' for p in ['ka', 'ke', 'V'] for cov in all_cov_names_pre_subset]
penalized_indices = np.arange(1,n_cov+ 1)
M                 = C.shape[2] - z_dim   
#########################################################
# Iterations Setup
#########################################################
iters_burn_in = args.iters_burn_in if args.iters_burn_in is not None else 100
iters         = args.iters         if args.iters         is not None else 300
L_iter        = 5
# Scale kl_iter and gamma_iter to iters so short runs stay numerically stable.
# Original paper: iters=300, kl_iter=50 (~17%), gamma_iter=250 (~83%).
# If the user supplied --gamma_iter explicitly, honour it (clamped to iters).
if iters >= 300:
    kl_iter    = 50
    gamma_iter = args.gamma_iter if args.gamma_iter is not None else 250
else:
    kl_iter    = max(1, min(50,  int(round(0.17 * iters))))
    gamma_iter = max(1, min(args.gamma_iter if args.gamma_iter is not None else iters - 1,
                            int(round(0.83 * iters))))
burn_in_iter  = L_iter * iters_burn_in           # Overall burn-in gradient steps
alpha_KL      = torch.linspace(0.01, 1, kl_iter) # KL annealing factor
smoothing     = False
#########################################################
# Initialize population parameters updates
#########################################################
pop =  pop_parameter(z_dim, nbatch, gamma_iter, data, C, C_regression, C[:,:z_dim,:z_dim], penalized_indices, n_cov, kl_iter, lengths, 2)
#########################################################
# Burn in
#########################################################
Encoder, optimizer, pred_x, mu, L, a, b, z_pop_iter_bi, omega_pop_iter_bi, a_iter_bi, elbo_iter_bi = initalizeEncoder(iters_burn_in, L_iter, Encoder, Decoder, data, data_in, z_dim, covariates_in, lengths, h, pop)
pred_x_mean = pred_x.detach()
#########################################################
# Setup Optimizer
#########################################################
optimizer.param_groups[0]['lr'] = 5e-3 # Learning rate
#########################################################
# Initialize tensors to store results
#########################################################
z_pop_iter     = torch.zeros(iters, z_dim + M)
omega_pop_iter = torch.zeros(iters, z_dim)
a_iter         = torch.zeros(iters)
Elbo_iter      = torch.zeros(iters)
#########################################################
# Training
#########################################################
print('')
print('#############################################')
print('Training VAE')
z_pop_cached = None  # last full-length z_pop from a MIQP call; reused on non-MIQP iterations
for iter in range(1,iters + 1):
    #########################################################
    # Update population parameters
    #########################################################
    if iter > 1:
        pred_x_mean = data_matrix[L_iter-2:L_iter].mean(dim = 0)
    do_miqp = (iter == 1) or (iter % args.miqp_every == 0)
    if iter > gamma_iter:
        z_pop_ret, omega_pop, a, _ = pop.update_pop(mu.detach(), L.detach(), pred_x_mean, iter, covariate_selection=do_miqp, smoothing=True, update_pop=True)
    else:
        z_pop_ret, omega_pop, a, mu_smooth = pop.update_pop(mu.detach(), L.detach(), pred_x_mean, iter, covariate_selection=do_miqp, update_pop=True)
    if do_miqp:
        z_pop_cached = z_pop_ret
    z_pop = z_pop_cached
    
    data_matrix = torch.zeros(L_iter, nbatch, data.shape[1], 1) # Initialize tensor to store prediction and ELBO over L_iter gradient steps
    ELBO        = torch.zeros(L_iter)
    for l in range(L_iter):
        #########################################################
        # Encoder
        #########################################################
        z_normal, mu, L, log_sigma, eps = Encoder(data_in, covariates_in, lengths)
        #########################################################
        # Decoder
        #########################################################
        pred_x = Decoder(z_normal, data[:,:,0], h)
        data_matrix[l] = pred_x.clone().detach()
        #########################################################
        # Loss computation
        #########################################################
        z_pop_batch = torch.zeros(nbatch, z_dim)
        for i in range(nbatch):
            z_pop_batch[i] = torch.matmul(C[i], z_pop)
        p_x_z = p_x_z_compute(data[:,:,1].view(data.shape[0],data.shape[1],1), pred_x, [a,b], lengths) # Compute likelihood p(x|z)
        p_z   = p_z_compute(z_normal, z_pop_batch, omega_pop)                                          # Compute prior p(z)
        q_z   = q_z_x_compute(eps, torch.diagonal(L, dim1=1, dim2=2))                                  # Compute posterior q(z|x)
        DKL   = p_z - q_z                                                                              # Compute KL divergence DKL = p(z) - q(z|x)
        if iter < kl_iter:
            elbo = p_x_z + alpha_KL[iter] *DKL 
        else:
            elbo = p_x_z + DKL 
        # Store ELBO
        ELBO[l] = p_x_z + DKL 
        

        if torch.isfinite(elbo):
            elbo.backward()
            optimizer.step()
        optimizer.zero_grad()
 
    #########################################################
    # Print
    #########################################################
    if iter < gamma_iter:
        print(f"Iteration {iter}/{gamma_iter}", end="\r")
    else:
        print(f"Iteration {iter - gamma_iter}/{iters- gamma_iter} (smoothing)", end="\r")

    #########################################################
    # Save Iteration results
    #########################################################
    z_pop_iter[iter-1]     = torch.hstack((h(z_pop[:z_dim]), z_pop[z_dim:]))
    omega_pop_iter[iter-1] = omega_pop.sqrt()
    a_iter[iter-1]         = a
    Elbo_iter[iter-1]      = ELBO.mean()
    

#########################################################
# Compute log likelihood
#########################################################
LL_lin_mu = LL_lin = LL_is = float('nan')
if not args.skip_ll:
    print('')
    print('#############################################')
    print('Compute log likelihood and empirical Bayes estimate')
    LL_lin_mu = LogLikelihood_linearization(z_pop, omega_pop, [a,b], data, mu_smooth, C, h, data[:,:,0], lengths, Decoder)
    phi_opt   = torch.zeros(nbatch, z_dim)
    for i in range(nbatch):
        phi_opt[i,:] = EmpiricalBayesEstimate(data[i,:,:], z_pop, omega_pop, mu[i], [a,b], C[i], h)
    LL_lin   = LogLikelihood_linearization(z_pop, omega_pop, [a,b], data, phi_opt, C, h, data[:,:,0], lengths, Decoder)
    LL_is, _ = LogLikelihood_sample(3000, z_pop, omega_pop, [a,b], data, mu, L, C, h, data[:,:,0], lengths, Decoder)
    print('#############################################')
else:
    print('(Skipping log-likelihood / EBE computation -- pass without --skip_ll for full output)')


#########################################################
# Output results
#########################################################
z_pop_iter     = torch.vstack((torch.hstack((z_pop_iter_bi ,torch.zeros(iters_burn_in,M))), z_pop_iter))
omega_pop_iter = torch.vstack((omega_pop_iter_bi, omega_pop_iter))
a_iter         = torch.hstack((a_iter_bi, a_iter))


printOutput_theo(z_pop, omega_pop, a, b, z_dim, nbatch, lengths.sum(), h, names_co, LL_lin_mu, LL_is)

if args.plot_dir:
    import os as _os
    from visualization import plotConvergence_pop, plotConvergence_covariate, plotConvergence_beta
    _os.makedirs(args.plot_dir, exist_ok=True)
    _tag = f'theophylline_nnoise{args.n_noise_cov}_ncov{n_cov}'
    plotConvergence_pop(
        Elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        param_names=['ka', 'ke', 'V'],
        save_path=_os.path.join(args.plot_dir, f'{_tag}_popParam.pdf'))
    plotConvergence_covariate(
        z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        z_dim=z_dim, n_cov=n_cov, param_names=['ka', 'ke', 'V'],
        cov_names=all_cov_names_pre_subset, max_cov=8,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_covariate.pdf'))
    plotConvergence_beta(
        z_pop, z_dim, n_cov, ['ka', 'ke', 'V'], all_cov_names_pre_subset,
        save_path=_os.path.join(args.plot_dir, f'{_tag}_beta.pdf'))

if args.results_json:
    import json
    import numpy as np
    beta = z_pop[z_dim:].detach().numpy().reshape(z_dim, n_cov) if n_cov > 0 else np.zeros((z_dim, 0))
    selected = (np.abs(beta) > 1e-9).any(axis=0).tolist() if n_cov > 0 else []
    n_selected = int(sum(selected))
    z_pop_h = h(z_pop)
    with open(args.results_json, 'w') as f:
        json.dump(dict(
            dataset='theophylline', n_batch=nbatch, n_cov=n_cov, z_dim=z_dim, seed=args.seed,
            n_true_cov=2, n_noise_cov=args.n_noise_cov, noise_source=args.noise_source,
            n_observations=int(lengths.sum().item()),
            covariate_names=all_cov_names_pre_subset, selected=selected, n_selected=n_selected, M=M,
            z_pop_structural=z_pop_h[:z_dim].tolist(),
            beta=beta.tolist(),
            beta_max=float(np.abs(beta).max()) if beta.size else 0.0,
            big_m=100, standardise_C=args.standardise_C, miqp_every=args.miqp_every,
            ofv_lin=(None if LL_lin != LL_lin else float(2 * LL_lin)), ofv_is=(None if LL_is != LL_is else float(2 * LL_is)),
        ), f, indent=2)
