"""
functions_warfarin.py
-------------------------

NEW, additive ancillary helper module for Case Study 6 (Warfarin PK),
mirroring functions_tacrolimus.py's role but for the 1-compartment
first-order-absorption-with-lag structural model in VAE/decoder_warfarin.py.

Data sources
------------
- Warfarin_data.csv: wide format, 14 fixed nominal slots
  (0.5,1,1.5,2,3,6,9,12,24,36,48,72,96,120h), blank = not observed for that
  subject. VAE-NLME's ODE-based decoder needs no fixed grid, so each
  subject's blanks are simply dropped and the remaining observations
  compacted into a dense variable-length (time, conc) sequence padded via
  `lengths` -- this also means VAE-NLME sees the exact same
  duplicate-averaged values as CVAE-LASSO for subject 8's replicate
  measurements at t=3/6/9/12h, instead of the two methods disagreeing on
  those four points as they did when VAE-NLME read the separate
  Warfarin_long.csv.
- Warfarin_covariates.csv: AGE, SEX, WT + 250 TCGA RNAseq noise genes.
- Dose is NOT read from a stored value -- it's computed on the fly as
  1.5 * WT per subject (matches the original recorded per-subject doses in
  the raw NONMEM file almost exactly: ratio 1.4991-1.5007 across all 32
  subjects).

Both CSVs originate from cvae_lasso/Data_real/ (shared with the CVAE-LASSO
project) and are committed here as a snapshot copy under this repo's Data/
directory so vae_nlme-main is runnable standalone from a fresh clone. If
the cvae_lasso source is corrected/updated, this copy must be re-synced
manually -- there is no automatic link between the two.
"""

import csv

import numpy as np
import torch
from scipy.optimize import minimize

WARFARIN_SLOTS = [0.5, 1.0, 1.5, 2.0, 3.0, 6.0, 9.0, 12.0, 24.0, 36.0, 48.0, 72.0, 96.0, 120.0]


def load_warfarin(data_path, cov_path):
    """
    Load Warfarin_data.csv (wide, 14-slot, blanks=missing) +
    Warfarin_covariates.csv (real + noise), returning the same 9-tuple
    contract as data_loading.load_two_file_wide / load_nonmem_like:
    (data, data_in, lengths, dose, weight_pop, covariates, covariates_in,
    covariate_names, n_cov).
    """
    raw = np.genfromtxt(data_path, delimiter=',', skip_header=1, filling_values=np.nan)
    if raw.ndim == 1:
        raw = raw[None, :]
    ids = raw[:, 0]
    X = raw[:, 1:]  # [nbatch, 14]
    nbatch = X.shape[0]

    with open(cov_path) as f:
        reader = csv.reader(f)
        cov_header = next(reader)
        cov_rows = list(reader)
    cov_ids = np.array([float(r[0]) for r in cov_rows])
    covariate_names = cov_header[1:]
    n_cov = len(covariate_names)
    cov_matrix = np.array([[float(v) for v in r[1:]] for r in cov_rows])

    if not np.array_equal(ids, cov_ids):
        raise ValueError("Warfarin data/covariate subject IDs do not match")

    dose_arr = 1.5 * cov_matrix[:, covariate_names.index('WT')]

    # Compact each subject's observed (non-blank) slots into a dense prefix
    # sequence -- columns are already in ascending time order, so no
    # explicit sort is needed.
    lengths_list = []
    per_subj_times, per_subj_conc = [], []
    for i in range(nbatch):
        obs = ~np.isnan(X[i])
        per_subj_times.append(np.asarray(WARFARIN_SLOTS)[obs])
        per_subj_conc.append(X[i][obs])
        lengths_list.append(int(obs.sum()))

    T = max(lengths_list)
    lengths = torch.tensor(lengths_list, dtype=torch.int32)

    data = torch.zeros(nbatch, T, 3 + n_cov, dtype=torch.float32)
    cov_t = torch.from_numpy(cov_matrix).float()
    for i in range(nbatch):
        n_i = lengths_list[i]
        data[i, :, 2] = float(dose_arr[i])
        data[i, :, 3:] = cov_t[i]
        data[i, :n_i, 0] = torch.from_numpy(per_subj_times[i]).float()
        data[i, :n_i, 1] = torch.from_numpy(per_subj_conc[i]).float()

    dose = data[:, 0, 2].clone()
    weight_pop = data[:, 0, 3].mean() if n_cov > 0 else torch.tensor(float('nan'))
    covariates = data[:, 0, 3:].clone()

    #########################################################
    # Standardize input data (mirrors data_loading.load_nonmem_like)
    #########################################################
    data_in = data[:, :, :2].clone()
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    data_mean = data_in[:, :, 1][mask].mean()
    data_std = data_in[:, :, 1][mask].std()
    time_max = data_in[:, :, 0][mask].max()
    data_in[:, :, 0] = data_in[:, :, 0] / time_max
    data_in[:, :, 1] = (data_in[:, :, 1] - data_mean) / data_std
    pad_mask = ~mask
    data_in[:, :, 0][pad_mask] = 0.0
    data_in[:, :, 1][pad_mask] = 0.0

    covariates_in = covariates.clone()
    for j in range(n_cov):
        col = covariates_in[:, j]
        std = col.std()
        if std > 0:
            covariates_in[:, j] = (col - col.mean()) / std

    return (data, data_in, lengths, dose, weight_pop, covariates, covariates_in,
            covariate_names, n_cov)


def initalize_C(nbatch, z_dim, n_cov, covariates, weight_pop):
    """Generalizes functions_theo.initalize_C to z_dim=4 (Warfarin's
    [Tlag, ka, Cl, V]) and an arbitrary n_cov, with the same
    categorical-vs-continuous auto-detection as functions_tacrolimus.initalize_C
    (Warfarin's only categorical true covariate is Sex)."""
    is_categorical = torch.tensor(
        [len(torch.unique(covariates[:, j])) <= 5 for j in range(n_cov)], dtype=torch.bool
    )
    log_cov = torch.log(covariates.clamp(min=1e-6) / weight_pop.clamp(min=1e-6))
    cov_term = torch.where(is_categorical.unsqueeze(0), covariates, log_cov)

    C = torch.zeros(nbatch, z_dim, z_dim + z_dim * n_cov)
    C[:, :, :z_dim] = torch.eye(z_dim).unsqueeze(0).expand(nbatch, z_dim, z_dim)
    for k in range(z_dim):
        C[:, k, z_dim + k * n_cov: z_dim + (k + 1) * n_cov] = cov_term

    C_regression = torch.zeros(z_dim, nbatch, 1 + n_cov)
    C_regression[:, :, 0] = 1
    C_regression[:, :, 1:] = cov_term.unsqueeze(0).expand(z_dim, nbatch, n_cov)

    return C, C_regression


def EmpiricalBayesEstimate(data, z_pop, omega_pop, mu, res, C, h, dose):
    """
    Per-subject MAP (empirical Bayes) estimate of phi = [Tlag, ka, Cl, V],
    mirroring functions_tacrolimus.EmpiricalBayesEstimate but with per-subject
    dose and BOTH ka and Tlag estimated (not fixed population constants).

    data : [T, 3+n_cov]  one subject's full data row block
           (data[:,0]=time, data[:,1]=observed concentration).
    """
    a = res[0].numpy()
    b = res[1].numpy()
    Cz = torch.matmul(C, z_pop).detach().numpy()
    data_np = data.detach().numpy()
    omega_pop_np = omega_pop.detach().numpy()
    mu_np = mu.detach().numpy()
    dose_val = float(dose)

    def f(t, Tlag, ka, Cl, V):
        ke = Cl / V
        if abs(ka - ke) < 1e-9:
            ke = ke + 1e-6
        if t < Tlag:
            return 0.0
        dt = t - Tlag
        return dose_val * ka / (V * (ka - ke)) * (np.exp(-ke * dt) - np.exp(-ka * dt))

    def EBE(phi):
        phi_h = h(torch.tensor(phi)).numpy()
        Tlag, ka, Cl, V = phi_h
        pred_x = np.array([f(t, Tlag, ka, Cl, V) for t in data_np[:, 0]])
        sigma = a + b * pred_x
        sigma = np.clip(sigma, 1e-8, None)
        epsilon = (data_np[:, 1] - pred_x) / sigma
        tmp = 0.5 * np.sum((phi - Cz) ** 2 / omega_pop_np)
        loss = np.sum(0.5 * (epsilon) ** 2 + np.log(sigma)) + tmp
        return loss

    phi = minimize(EBE, mu_np, method='Nelder-Mead',
                   options={'xatol': 1e-6, 'maxiter': 300}).x
    return torch.tensor(phi)
