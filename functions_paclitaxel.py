"""
functions_paclitaxel.py
-------------------------

NEW, additive ancillary helper module for Case Study 4 (Paclitaxel), mirroring
functions_tacrolimus.py's role but for the Joerger 2006 3-compartment
structural model.

initalize_C / EmpiricalBayesEstimate generalize functions_theo.py's versions
to z_dim = 6 latent PK parameters and the auto-detected
categorical-vs-continuous covariate handling that data_loading.load_two_file_wide
+ this dataset's ~290 raw covariate columns require (Sex is the only
categorical column among Paclitaxel's true/clinical covariates; everything
else, including the 250 RNAseq noise columns, is continuous).

Does NOT modify functions_theo.py / functions_neonates.py.
"""

import numpy as np
import torch
from scipy.optimize import minimize

from VAE.decoder_paclitaxel import _rate_in


def initalize_C(nbatch, z_dim, n_cov, covariates, weight_pop):
    """
    Generalizes functions_theo.initalize_C to z_dim = 6 (Paclitaxel's
    [V1, V3, VMEL, VMTR, KMTR, Q]) and an arbitrary n_cov, with the same
    categorical-vs-continuous auto-detection as functions_tacrolimus.initalize_C
    (Paclitaxel's only categorical true/clinical covariate is Sex; see
    conditioning_limits_Paclitaxel/config.py).
    """
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


def _simulate_central_conc(t_obs, V1, V3, VMEL, VMTR, KMTR, Q, K21, KMEL, dose_umol, t_inf):
    """
    Numpy/scipy reference ODE integration of the Joerger 2006 3-compartment
    model for a single subject (used only inside the per-subject scalar
    Nelder-Mead optimization of EmpiricalBayesEstimate, where a numpy/scipy
    solve is simpler and fast enough at this per-call granularity than going
    through torchode for a single subject's gradient-free EBE).
    """
    from scipy.integrate import solve_ivp

    def rhs(t, y, rate_in):
        A1, A2, A3 = y
        C1 = max(A1, 0.0) / V1
        C3 = max(A3, 0.0) / V3
        EL = VMEL * C1 / (KMEL + C1)
        TR_12 = VMTR * C1 / (KMTR + C1)
        TR_21 = K21 * max(A2, 0.0)
        TR_13 = Q * C1
        TR_31 = Q * C3
        dA1 = rate_in - EL - TR_12 + TR_21 - TR_13 + TR_31
        dA2 = TR_12 - TR_21
        dA3 = TR_13 - TR_31
        return [dA1, dA2, dA3]

    rate_in = dose_umol / t_inf
    t_max = float(max(t_obs.max(), t_inf))
    t_eval = np.unique(np.concatenate([t_obs, [t_inf]]))
    t_eval = t_eval[t_eval <= t_max]

    sol1 = solve_ivp(rhs, (0, t_inf), [0.0, 0.0, 0.0],
                     t_eval=t_eval[t_eval <= t_inf], args=(rate_in,),
                     method="Radau", rtol=1e-6, atol=1e-9)
    y_end = sol1.y[:, -1] if sol1.y.shape[1] > 0 else [0.0, 0.0, 0.0]

    t_eval2 = t_eval[t_eval >= t_inf]
    if len(t_eval2) == 0:
        t_eval2 = np.array([t_inf])
    sol2 = solve_ivp(rhs, (t_inf, t_max), y_end,
                     t_eval=t_eval2, args=(0.0,),
                     method="Radau", rtol=1e-6, atol=1e-9)

    all_t = np.concatenate([sol1.t, sol2.t])
    all_A1 = np.concatenate([sol1.y[0], sol2.y[0]])
    conc = np.interp(t_obs, all_t, all_A1) / V1
    return conc


def EmpiricalBayesEstimate(data, z_pop, omega_pop, mu, res, C, h, dose, K21=0.209, KMEL=0.047,
                           t_inf=3.0, estimate_k21_kmel=False):
    """
    Per-subject MAP (empirical Bayes) estimate of
    phi = [V1, V3, VMEL, VMTR, KMTR, Q] (z_dim=6, K21/KMEL fixed at the given
    population-typical values), or phi = [V1, V3, VMEL, VMTR, KMTR, Q, K21,
    KMEL] (z_dim=8) when `estimate_k21_kmel=True`, mirroring functions_theo's
    version but using the Paclitaxel 3-compartment structural model.
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
        if estimate_k21_kmel:
            V1, V3, VMEL, VMTR, KMTR, Q, K21_i, KMEL_i = phi_h
        else:
            V1, V3, VMEL, VMTR, KMTR, Q = phi_h
            K21_i, KMEL_i = K21, KMEL
        try:
            pred_x = _simulate_central_conc(t_obs, V1, V3, VMEL, VMTR, KMTR, Q, K21_i, KMEL_i,
                                            dose_val, t_inf)
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
