"""
functions_quinidine.py
-------------------------

NEW, additive ancillary helper module for Case Study 5 (Quinidine dense PK),
mirroring functions_tacrolimus.py / functions_paclitaxel.py's role but for
the 2-compartment first-order-absorption-with-lag structural model in
VAE/decoder_quinidine.py.

Does NOT modify functions_theo.py / functions_neonates.py.
"""

import numpy as np
import torch
from scipy.optimize import minimize


def initalize_C(nbatch, z_dim, n_cov, covariates, weight_pop):
    """Generalizes functions_theo.initalize_C to z_dim=6 (Quinidine's
    [Tlag, ka, Cl, V1, Q, V2]) and an arbitrary n_cov, with the same
    categorical-vs-continuous auto-detection as functions_tacrolimus.initalize_C
    (Quinidine's categorical true/clinical covariates are Sex and the RACE_1..4
    one-hot columns; everything else, including the 250 RNAseq noise columns,
    is continuous)."""
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


def _simulate_central_conc(t_obs, Tlag, ka, Cl, V1, Q, V2, dose_val):
    """
    Numpy/scipy reference ODE integration of the 2-compartment
    first-order-absorption-with-lag model for a single subject (used only
    inside the per-subject scalar Nelder-Mead optimization of
    EmpiricalBayesEstimate, mirroring functions_paclitaxel._simulate_central_conc).
    """
    from scipy.integrate import solve_ivp

    def rhs(tp, y):
        Ad, A1, A2 = y
        dAd = -ka * Ad
        dA1 = ka * Ad - (Cl / V1 + Q / V1) * A1 + (Q / V2) * A2
        dA2 = (Q / V1) * A1 - (Q / V2) * A2
        return [dAd, dA1, dA2]

    t_shifted = np.clip(t_obs - Tlag, 0.0, None)
    t_eval = np.unique(t_shifted)
    t_max = float(t_eval.max()) if len(t_eval) else 0.0
    sol = solve_ivp(rhs, (0, t_max), [dose_val, 0.0, 0.0],
                     t_eval=t_eval, method="Radau", rtol=1e-6, atol=1e-9)
    conc_at_unique = sol.y[1] / V1
    conc = np.interp(t_shifted, t_eval, conc_at_unique)
    return conc


def EmpiricalBayesEstimate(data, z_pop, omega_pop, mu, res, C, h, dose):
    """
    Per-subject MAP (empirical Bayes) estimate of
    phi = [Tlag, ka, Cl, V1, Q, V2] (z_dim=6), mirroring functions_theo's
    version but using the Quinidine 2-compartment structural model.

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
    t_obs = data_np[:, 0]

    def EBE(phi):
        phi_h = h(torch.tensor(phi)).numpy()
        Tlag, ka, Cl, V1, Q, V2 = phi_h
        try:
            pred_x = _simulate_central_conc(t_obs, Tlag, ka, Cl, V1, Q, V2, dose_val)
        except Exception:
            return 1e12
        sigma = a + b * pred_x
        sigma = np.clip(sigma, 1e-6, None)
        epsilon = (data_np[:, 1] - pred_x) / sigma
        tmp = 0.5 * np.sum((phi - Cz) ** 2 / omega_pop_np)
        loss = np.sum(0.5 * (epsilon) ** 2 + np.log(sigma)) + tmp
        return loss

    phi = minimize(EBE, mu_np, method='Nelder-Mead',
                   options={'xatol': 1e-4, 'maxiter': 300}).x
    return torch.tensor(phi)
