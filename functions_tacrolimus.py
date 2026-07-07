"""
functions_tacrolimus.py
-------------------------

NEW, additive ancillary helper module for Case Study 3 (Tacrolimus), mirroring
the role of functions_theo.py (load_data / initalize_C / EmpiricalBayesEstimate)
but wired to data_loading.load_two_file_wide and the Tacrolimus structural
decoder.

Does NOT modify functions_theo.py / functions_neonates.py.
"""

import numpy as np
import torch
from scipy.optimize import minimize

from data_loading import load_two_file_wide


def load_tacrolimus(conc_path, cov_path):
    """
    Thin wrapper around data_loading.load_two_file_wide for the Tacrolimus
    dataset. Ground truth (confirmed by the user) uses a single fixed dose
    of 300 for every subject, a fixed ka=0.502 and a fixed absorption lag
    t0=0.346 -- there is no per-subject dose to read from a covariate or
    parameters file, so the scalar default dose=300.0 in
    VAE/decoder_tacrolimus.py is exact, not an approximation.
    """
    from data_loading import load_two_file_wide as _load
    return _load(conc_path, cov_path, id_col='ID', dose=300.0)


def initalize_C(nbatch, z_dim, n_cov, covariates, weight_pop):
    """
    Builds the per-subject linear-in-log-covariates design tensor C and the
    regression design tensor C_regression, mirroring functions_theo.initalize_C
    but generalized to an arbitrary n_cov (functions_theo.initalize_C hardcodes
    a special-case `if j == 1` for the categorical Sex covariate of the
    theophylline dataset; here all n_cov covariates are continuous (Age,
    Hemoglobin, Albumin, SNP, ... after caller-side standardization in
    data_loading), so every covariate enters via log(cov_ij / weight_pop) --
    i.e. log(standardized_covariate / mean(standardized_covariate)). Since
    covariates_in fed to the Encoder is already z-scored (mean ~0), callers
    should pass the RAW (unstandardized) `covariates` tensor here (as
    functions_theo.py does), not covariates_in, for the log-linear regression
    design to be meaningful; raw values are guaranteed > 0 only for genuinely
    continuous positive covariates (Age, Hemoglobin, Albumin) -- the SNP
    covariate (categorical, integer-coded, can be 0) is handled by an
    `is_categorical` mask so it enters linearly (like functions_theo's
    Sex special-case at j==1), not log-linearly.
    """
    # Vectorized (the naive triple Python for-loop over nbatch*z_dim*n_cov is
    # prohibitively slow for n_cov ~ 261 and nbatch ~ 10000 -- ~7.8M Python-level
    # tensor-index ops). Equivalent values, computed via broadcasting instead.
    is_categorical = torch.tensor(
        [len(torch.unique(covariates[:, j])) <= 5 for j in range(n_cov)], dtype=torch.bool
    )

    log_cov = torch.log(covariates.clamp(min=1e-6) / weight_pop.clamp(min=1e-6))  # [nbatch, n_cov]
    cov_term = torch.where(is_categorical.unsqueeze(0), covariates, log_cov)       # [nbatch, n_cov]

    C = torch.zeros(nbatch, z_dim, z_dim + z_dim * n_cov)
    C[:, :, :z_dim] = torch.eye(z_dim).unsqueeze(0).expand(nbatch, z_dim, z_dim)
    # For each k in [0, z_dim), the n_cov covariate-effect columns occupy
    # [z_dim + k*n_cov : z_dim + (k+1)*n_cov), and are identical (cov_term)
    # across k (matches functions_theo.initalize_C's structure where each
    # individual parameter's covariate block uses the same per-subject
    # covariate values, only the regression coefficient differs by k).
    for k in range(z_dim):
        C[:, k, z_dim + k * n_cov: z_dim + (k + 1) * n_cov] = cov_term

    C_regression = torch.zeros(z_dim, nbatch, 1 + n_cov)
    C_regression[:, :, 0] = 1
    C_regression[:, :, 1:] = cov_term.unsqueeze(0).expand(z_dim, nbatch, n_cov)

    return C, C_regression


def EmpiricalBayesEstimate(data, z_pop, omega_pop, mu, res, C, h,
                           dose=300.0, ka=0.502, t0=0.346):
    """
    Per-subject MAP (empirical Bayes) estimate of phi = [ke, V], mirroring
    functions_theo.EmpiricalBayesEstimate but using the Tacrolimus structural
    model: a single dose, fixed ka and fixed absorption lag t0 (see
    VAE/decoder_tacrolimus.py) -- only ke and V are subject-specific.

    data : [T, 3+n_cov]  one subject's full data row block
           (data[:,0]=time, data[:,1]=observed concentration).
    """
    a = res[0].numpy()
    b = res[1].numpy()
    Cz = torch.matmul(C, z_pop).detach().numpy()
    data_np = data.detach().numpy()
    omega_pop_np = omega_pop.detach().numpy()
    mu_np = mu.detach().numpy()

    def f(t, ke, V):
        if abs(ka - ke) < 1e-9:
            ke = ke + 1e-6
        if t < t0:
            return 0.0
        dt = t - t0
        return dose * ka / (V * (ka - ke)) * (np.exp(-ke * dt) - np.exp(-ka * dt))

    def EBE(phi):
        phi_h = h(torch.tensor(phi)).numpy()
        ke, V = phi_h[0], phi_h[1]
        pred_x = np.array([f(t, ke, V) for t in data_np[:, 0]])
        sigma = a + b * pred_x
        sigma = np.clip(sigma, 1e-8, None)
        epsilon = (data_np[:, 1] - pred_x) / sigma
        tmp = 0.5 * np.sum((phi - Cz) ** 2 / omega_pop_np)
        loss = np.sum(0.5 * (epsilon) ** 2 + np.log(sigma)) + tmp
        return loss

    phi = minimize(EBE, mu_np, method='Nelder-Mead',
                   options={'xatol': 1e-6, 'maxiter': 300}).x
    return torch.tensor(phi)
