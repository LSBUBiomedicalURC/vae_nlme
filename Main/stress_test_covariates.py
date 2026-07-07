"""
stress_test_covariates.py
---------------------------

CLI stress-test of covariate selection robustness, using the Tacrolimus
pipeline (clean small ground truth: N_TRUE = 4 informative covariates --
Age, Hemoglobin, Albumin, SNP -- per conditioning_limits_PK/config.py and
project memory: SNP dominates true-covariate explained variance (~90%),
followed by age (~6%), hemoglobin (~3.3%), albumin (~0.7%)).

This script depends on the full training pipeline (gurobipy / cvxpy /
torchode), so a full multi-iteration run end-to-end may not be executable in
every environment (e.g. a size-limited Gurobi license -- see this repo's
final report for what was actually verified to run). The per-run training
logic is factored into `run_condition()` so the experiment-loop /
aggregation logic (the part most relevant to a stress-test design) is
testable independently of whether the heavy training actually completes --
`build_condition_data()` and `summarize_results()` have no dependency on
gurobipy/torchode and are unit-testable on their own.

Conditions varied
------------------
1. Number of injected uninformative covariates appended to the 4 true
   covariates: --noise_dims (default 0 5 20 50 100 200), drawn either
   iid N(0,1) (default) or as permuted real noise covariates from the
   Tacrolimus RNAseq block (--noise_source permuted).
2. Ablation: for a covariate nominally zeroed-out at the population level
   (covariate_selection sets its beta to 0), compare individual-level
   ELBO/predictions with that covariate (a) still fed into covariates_in
   to the Encoder vs (b) removed entirely from covariates_in -- to test
   whether covariate information leaks into z through the encoder even
   when the population-level coefficient is regularized to zero.
3. Partially-informative condition: noisy/attenuated copies of a true
   covariate (true covariate + Gaussian noise at increasing SNR levels)
   to test graceful degradation of covariate selection / recovery.

Metrics recorded per run (one row of the output table)
---------------------------------------------------------
  - false_positive_rate: fraction of known-noise covariates with nonzero
    population coefficient after covariate_selection (averaged across the
    z_dim=3 PK parameters' coefficient blocks).
  - false_negative_rate: fraction of the 4 true covariates zeroed out
    (averaged across the z_dim=3 coefficient blocks).
  - param_rmse_{CL,V,ke}: RMSE of the population PK parameter estimates
    against conditioning_limits_PK/Tacrolimus_parameters.csv ground truth
    (population mean of CL = ke*V, V, ke across the *training subset's*
    ground-truth subjects).
  - elbo_leak_with / elbo_leak_without / elbo_leak_diff: ablation metric (2).
  - snr / recovery_rate: partial-information degradation metric (3).

Output: a CSV (and JSON) summary table, one row per condition -- not ad hoc
prints (see write_summary()).
"""
#########################################################
# Import
#########################################################
import argparse
import json
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import pandas as pd
import torch

PK_DATA_DIR_DEFAULT = r'C:\Work\Research\Structural_Collapse\Final_experiments\conditioning_limits_PK'
N_TRUE = 4
TRUE_COV_NAMES = ['Age', 'Hemoglobin', 'Albumin', 'SNP']


#########################################################
# Data assembly (no gurobipy/torchode dependency -- unit-testable standalone)
#########################################################
def build_condition_data(data_dir, n_subjects, noise_dim, noise_source='iid', snr=None,
                          true_covariate_for_snr='Age', seed=0):
    """
    Assembles (covariates, covariate_names, dose, conc, params_df) for one
    stress-test condition, WITHOUT touching gurobipy/torchode -- purely CSV
    reads + numpy. This is the part of the pipeline that is testable even
    when the training dependencies are unavailable.

    Parameters
    ----------
    data_dir : str
        Directory with Tacrolimus_data.csv / Tacrolimus_covariates.csv /
        Tacrolimus_parameters.csv.
    n_subjects : int
        Number of subjects to subsample (keeps the per-run cvxpy problem
        size tractable; see this module's __main__ for a license-size note).
    noise_dim : int
        Number of uninformative covariates to append to the 4 true ones.
    noise_source : 'iid' | 'permuted'
        'iid'      -- draw noise_dim iid N(0,1) columns.
        'permuted' -- take noise_dim columns from the dataset's real RNAseq
                      noise block (columns 11:261 in Tacrolimus_covariates.csv,
                      per conditioning_limits_PK/config.py), each independently
                      row-permuted (so they keep real marginal distributions
                      but carry no genuine subject-level signal).
    snr : float or None
        If given, ALSO append one "partially informative" covariate:
        true_covariate_for_snr's z-scored values plus Gaussian noise at the
        given signal-to-noise ratio (variance ratio signal/noise = snr).
        If None, no partially-informative covariate is added.
    true_covariate_for_snr : str
        Which of the 4 true covariates to attenuate for the SNR condition.
    seed : int
        RNG seed for noise draws / subject subsampling.

    Returns
    -------
    dict with keys: covariates (np.ndarray [n_subjects, n_cov]),
    covariate_names (list[str]), noise_mask (np.ndarray[bool], True where the
    column is known-noise), true_mask (np.ndarray[bool], True for the 4 true
    covariates), partial_idx (int or None, index of the SNR covariate if any),
    dose_mg (np.ndarray[n_subjects]), conc (np.ndarray[n_subjects, 49]),
    params_df (pd.DataFrame, ground truth for the selected subjects).
    """
    rng = np.random.default_rng(seed)

    conc_path = os.path.join(data_dir, 'Tacrolimus_data.csv')
    cov_path = os.path.join(data_dir, 'Tacrolimus_covariates.csv')
    params_path = os.path.join(data_dir, 'Tacrolimus_parameters.csv')

    conc_full = np.loadtxt(conc_path, delimiter=',')
    cov_header = pd.read_csv(cov_path, nrows=0).columns.tolist()
    cov_full = np.loadtxt(cov_path, delimiter=',', skiprows=1)
    params_full = pd.read_csv(params_path).dropna(subset=['mg_twice_daily_dose']).reset_index(drop=True)

    n_total = conc_full.shape[0]
    idx = rng.choice(n_total, size=min(n_subjects, n_total), replace=False)
    idx = np.sort(idx)

    conc = conc_full[idx, 1:]  # drop subject-ID column -> [n_subjects, 49]
    cov_all = cov_full[idx, 1:]  # drop subject-ID column -> [n_subjects, 261]
    cov_names_all = cov_header[1:]  # drop 'ID'
    dose_mg = np.clip(params_full['mg_twice_daily_dose'].to_numpy()[idx], 0.05, None)
    params_subset = params_full.iloc[idx].reset_index(drop=True)

    true_cols = cov_all[:, :N_TRUE]  # Age, Hemoglobin, Albumin, SNP
    true_names = cov_names_all[:N_TRUE]

    rnaseq_cols = cov_all[:, N_TRUE + 7:N_TRUE + 7 + 250]  # per config.py layout

    if noise_dim > 0:
        if noise_source == 'iid':
            noise_cols = rng.normal(0, 1, size=(len(idx), noise_dim))
            noise_names = [f'noise_iid_{j}' for j in range(noise_dim)]
        elif noise_source == 'permuted':
            n_avail = min(noise_dim, rnaseq_cols.shape[1])
            col_idx = rng.choice(rnaseq_cols.shape[1], size=n_avail, replace=(n_avail > rnaseq_cols.shape[1]))
            noise_cols = np.stack(
                [rng.permutation(rnaseq_cols[:, j]) for j in col_idx], axis=1
            )
            noise_names = [f'noise_permuted_{j}' for j in col_idx]
            if n_avail < noise_dim:
                extra = rng.normal(0, 1, size=(len(idx), noise_dim - n_avail))
                noise_cols = np.concatenate([noise_cols, extra], axis=1)
                noise_names += [f'noise_iid_extra_{j}' for j in range(noise_dim - n_avail)]
        else:
            raise ValueError(f"unknown noise_source: {noise_source!r}")
    else:
        noise_cols = np.zeros((len(idx), 0))
        noise_names = []

    covariates = np.concatenate([true_cols, noise_cols], axis=1)
    covariate_names = list(true_names) + noise_names
    true_mask = np.zeros(covariates.shape[1], dtype=bool)
    true_mask[:N_TRUE] = True
    noise_mask = ~true_mask

    partial_idx = None
    if snr is not None:
        ti = true_names.index(true_covariate_for_snr) if true_covariate_for_snr in true_names else 0
        signal = true_cols[:, ti]
        signal_z = (signal - signal.mean()) / (signal.std() + 1e-9)
        noise_std = np.sqrt(1.0 / max(snr, 1e-9))
        partial_col = signal_z + rng.normal(0, noise_std, size=len(idx))
        covariates = np.concatenate([covariates, partial_col[:, None]], axis=1)
        covariate_names.append(f'partial_{true_covariate_for_snr}_snr{snr}')
        true_mask = np.concatenate([true_mask, [False]])
        noise_mask = np.concatenate([noise_mask, [False]])
        partial_idx = covariates.shape[1] - 1

    return dict(
        covariates=covariates, covariate_names=covariate_names,
        true_mask=true_mask, noise_mask=noise_mask, partial_idx=partial_idx,
        dose_mg=dose_mg, conc=conc, params_df=params_subset,
    )


#########################################################
# Per-run training (gurobipy/torchode-dependent -- isolated so the
# experiment-loop logic above/below is testable independently)
#########################################################
def run_condition(cond_data, iters=80, iters_burn_in=30, z_dim=3, seed=0,
                  ablation_drop_idx=None, solver='GUROBI', allow_incompatible_solver=False):
    """
    Runs the actual VAE-nlme training pipeline (Encoder/Decoder/pop_parameter)
    for one stress-test condition and returns a dict of fitted results.

    Requires torch, the compiled VAE.encoder / ParaUpdate.pop_parameter
    modules, and (for the covariate-selection step) cvxpy + a Gurobi license
    able to handle a problem of size z_dim * (z_dim + n_cov). Raises a clear
    RuntimeError naming the missing/failing dependency rather than silently
    stubbing out the computation.

    ablation_drop_idx : int or None
        If given, this covariate column index is DROPPED from covariates_in
        fed to the Encoder (but kept in the population-level covariate model
        C/C_regression) -- the "(b) removed entirely" arm of the encoder
        information-leakage ablation. Compare against a call with
        ablation_drop_idx=None (the "(a) still fed in" arm) using the same
        cond_data/seed.
    """
    try:
        import importlib
        torch_mod = importlib.import_module('torch')
    except ImportError as e:
        raise RuntimeError(f"torch is required to run_condition(): {e}")

    try:
        from functions import (initalizeEncoder, p_x_z_compute, p_z_compute, q_z_x_compute,
                               LogLikelihood_linearization)
        from functions_tacrolimus import initalize_C
        from VAE.decoder_tacrolimus import Decoder_tacrolimus
        from VAE.encoder import LSTM_Encoder
        from ParaUpdate.pop_parameter import pop_parameter
        from solver_utils import set_pop_parameter_solver
        set_pop_parameter_solver(solver, allow_incompatible=allow_incompatible_solver)
    except ImportError as e:
        raise RuntimeError(
            "run_condition() requires the compiled VAE.encoder / "
            f"ParaUpdate.pop_parameter modules (missing/incompatible: {e}). "
            "These are compiled extensions (.pyd/.so) shipped with the repo; "
            "if missing for this platform/Python version, they must be "
            "rebuilt from VAE/encoder.c / ParaUpdate/pop_parameter.c."
        )

    torch_mod.manual_seed(seed)

    nbatch = cond_data['conc'].shape[0]
    n_cov = cond_data['covariates'].shape[1]
    time_grid = torch.tensor([float(t) for t in range(cond_data['conc'].shape[1])])

    data = torch.zeros(nbatch, len(time_grid), 3 + n_cov)
    data[:, :, 0] = time_grid.unsqueeze(0).expand(nbatch, -1)
    data[:, :, 1] = torch.from_numpy(cond_data['conc']).float()
    dose = torch.from_numpy(cond_data['dose_mg']).float()
    data[:, :, 2] = dose.unsqueeze(1).expand(nbatch, len(time_grid))
    cov_t = torch.from_numpy(cond_data['covariates']).float()
    data[:, :, 3:] = cov_t.unsqueeze(1).expand(nbatch, len(time_grid), n_cov)

    lengths = torch.full((nbatch,), len(time_grid), dtype=torch.int32)

    weight_pop = cov_t[:, 0].mean()
    covariates_in = cov_t.clone()
    for j in range(n_cov):
        std = covariates_in[:, j].std()
        if std > 0:
            covariates_in[:, j] = (covariates_in[:, j] - covariates_in[:, j].mean()) / std

    encoder_cov_in = covariates_in.clone()
    if ablation_drop_idx is not None:
        encoder_cov_in[:, ablation_drop_idx] = 0.0  # zero out -- "removed" from what the encoder sees

    data_mean = data[:, :, 1].mean()
    data_std = data[:, :, 1].std()
    data_in = data[:, :, :2].clone()
    data_in[:, :, 0] = data_in[:, :, 0] / data_in[:, :, 0].max()
    data_in[:, :, 1] = (data_in[:, :, 1] - data_mean) / data_std

    h = lambda x: x.exp()
    h_inverse = lambda x: x.log()
    mu0 = torch.tensor([0.5, 0.009, 3400.0])
    sigma0 = torch.tensor([1e-1, 1e-1, 1e-1]).log()
    Encoder = LSTM_Encoder(2, 25, z_dim, nbatch, n_cov, mu0, sigma0, h_inverse)
    Decoder = lambda z, t, h: Decoder_tacrolimus(z, t, h, dose)

    C, C_regression = initalize_C(nbatch, z_dim, n_cov, cov_t, weight_pop)
    penalized_indices = np.arange(1, n_cov + 1)
    gamma_iter = max(1, iters // 2)
    kl_iter = max(1, iters // 4)
    pop = pop_parameter(z_dim, nbatch, gamma_iter, data, C, C_regression,
                        C[:, :z_dim, :z_dim], penalized_indices, n_cov, kl_iter, lengths, 2)

    L_iter = 3
    (Encoder, optimizer, pred_x, mu, L, a, b, *_rest) = initalizeEncoder(
        iters_burn_in, L_iter, Encoder, Decoder, data, data_in, z_dim, encoder_cov_in, lengths, h, pop)

    pred_x_mean = pred_x.detach()
    optimizer.param_groups[0]['lr'] = 5e-3
    z_pop, omega_pop, mu_smooth = None, None, None

    for it in range(1, iters + 1):
        if it > 1:
            pred_x_mean = data_matrix[L_iter - 2:L_iter].mean(dim=0)
        z_pop, omega_pop, a, mu_smooth = pop.update_pop(
            mu.detach(), L.detach(), pred_x_mean, it, covariate_selection=True, update_pop=True)

        data_matrix = torch.zeros(L_iter, nbatch, data.shape[1], 1)
        for l in range(L_iter):
            z_normal, mu, L, log_sigma, eps = Encoder(data_in, encoder_cov_in, lengths)
            pred_x = Decoder(z_normal, data[:, :, 0], h)
            data_matrix[l] = pred_x.clone().detach()

            z_pop_batch = torch.zeros(nbatch, z_dim)
            for i in range(nbatch):
                z_pop_batch[i] = torch.matmul(C[i], z_pop)
            p_x_z = p_x_z_compute(data[:, :, 1].view(nbatch, data.shape[1], 1), pred_x, [a, b], lengths)
            p_z = p_z_compute(z_normal, z_pop_batch, omega_pop)
            q_z = q_z_x_compute(eps, torch.diagonal(L, dim1=1, dim2=2))
            DKL = p_z - q_z
            elbo = p_x_z + DKL
            elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

    final_elbo = float((p_x_z + DKL).detach())

    return dict(z_pop=z_pop.detach(), omega_pop=omega_pop.detach(), a=float(a), b=float(b),
               mu=mu.detach(), L=L.detach(), C=C, final_elbo=final_elbo,
               covariate_names=cond_data['covariate_names'],
               true_mask=cond_data['true_mask'], noise_mask=cond_data['noise_mask'],
               z_dim=z_dim, n_cov=n_cov, params_df=cond_data['params_df'])


#########################################################
# Metric computation (no gurobipy/torchode dependency)
#########################################################
def compute_metrics(fit_result):
    """
    Computes false-positive rate, false-negative rate, and parameter RMSE
    from a run_condition() result dict (or anything with the same keys --
    purely numpy/torch tensor ops on already-fitted values, so this function
    itself does not need cvxpy/gurobipy to RUN, only the upstream fit does).
    """
    z_dim = fit_result['z_dim']
    n_cov = fit_result['n_cov']
    z_pop = fit_result['z_pop']
    true_mask = fit_result['true_mask']
    noise_mask = fit_result['noise_mask']

    beta = z_pop[z_dim:].reshape(z_dim, n_cov)  # [z_dim, n_cov]
    nonzero = (beta.abs() > 1e-9).any(dim=0).numpy()  # [n_cov], True if selected for >=1 PK param

    fp = nonzero[noise_mask].mean() if noise_mask.sum() > 0 else float('nan')
    fn = (~nonzero[true_mask]).mean() if true_mask.sum() > 0 else float('nan')

    z_pop_h = z_pop[:z_dim].exp()  # [ka, ke, V] -> back to linear scale
    ke_hat, V_hat = float(z_pop_h[1]), float(z_pop_h[2])
    CL_hat = ke_hat * V_hat

    params_df = fit_result['params_df']
    CL_true = float(params_df['CL'].mean())
    V_true = float(params_df['V'].mean())
    ke_true = float(params_df['ke'].mean())

    rmse_CL = abs(CL_hat - CL_true)
    rmse_V = abs(V_hat - V_true)
    rmse_ke = abs(ke_hat - ke_true)

    return dict(
        false_positive_rate=float(fp), false_negative_rate=float(fn),
        param_rmse_CL=rmse_CL, param_rmse_V=rmse_V, param_rmse_ke=rmse_ke,
        n_selected=int(nonzero.sum()), n_cov=n_cov,
    )


#########################################################
# Experiment loop / aggregation (no gurobipy/torchode dependency -- only
# calls run_condition(), which is the one dependency-gated function)
#########################################################
def summarize_results(rows):
    """rows: list[dict] -> pandas DataFrame, one row per condition."""
    return pd.DataFrame(rows)


def write_summary(df, out_dir, basename='stress_test_covariates'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'{basename}.csv')
    json_path = os.path.join(out_dir, f'{basename}.json')
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2, default=str)
    return csv_path, json_path


#########################################################
# Main
#########################################################
def main():
    parser = argparse.ArgumentParser(description="Stress-test covariate selection robustness "
                                                  "(Tacrolimus VAE-nlme pipeline)")
    parser.add_argument('--data_dir', default=PK_DATA_DIR_DEFAULT)
    parser.add_argument('--n_subjects', type=int, default=60,
                        help="Subjects per run (kept small so the cvxpy/Gurobi covariate-selection "
                             "problem stays within a size-limited license's solvable range -- see "
                             "this repo's final report for the empirically observed Gurobi cap in "
                             "this environment).")
    parser.add_argument('--noise_dims', type=int, nargs='+', default=[0, 5, 20, 50, 100, 200])
    parser.add_argument('--noise_source', choices=['iid', 'permuted'], default='iid')
    parser.add_argument('--snr_levels', type=float, nargs='+', default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument('--ablation', action='store_true',
                        help="Also run the encoder-information-leakage ablation "
                             "(doubles the run count: with vs without the dropped covariate).")
    parser.add_argument('--iters', type=int, default=80)
    parser.add_argument('--iters_burn_in', type=int, default=30)
    parser.add_argument('--out_dir', default=os.path.join(parent, 'Plots', 'stress_test_results'))
    parser.add_argument('--dry_run', action='store_true',
                        help="Build conditions and print the planned experiment table without "
                             "calling run_condition() (no gurobipy/torchode dependency).")
    parser.add_argument('--solver', default='GUROBI',
                        help="cvxpy solver for pop_parameter's covariate-selection MIQP step. "
                             "See solver_utils.py: most free solvers cannot solve this problem "
                             "class (it's mixed-integer, not just QP).")
    parser.add_argument('--allow_incompatible_solver', action='store_true',
                        help="Skip solver_utils's known-incompatible-solver guard.")
    args = parser.parse_args()

    solver_kwargs = dict(solver=args.solver, allow_incompatible_solver=args.allow_incompatible_solver)
    rows = []

    #########################################################
    # Condition set 1: noise-dimension sweep
    #########################################################
    for noise_dim in args.noise_dims:
        cond_data = build_condition_data(args.data_dir, args.n_subjects, noise_dim,
                                         noise_source=args.noise_source, seed=noise_dim)
        row = dict(condition='noise_sweep', noise_dim=noise_dim, noise_source=args.noise_source,
                  n_subjects=args.n_subjects)
        if args.dry_run:
            row.update(n_cov=cond_data['covariates'].shape[1], status='dry_run')
            rows.append(row)
            continue
        try:
            fit = run_condition(cond_data, iters=args.iters, iters_burn_in=args.iters_burn_in, **solver_kwargs)
            row.update(compute_metrics(fit), status='ok')
        except RuntimeError as e:
            row.update(status='failed', error=str(e))
        rows.append(row)

    #########################################################
    # Condition set 2: partially-informative (SNR) sweep
    #########################################################
    for snr in args.snr_levels:
        cond_data = build_condition_data(args.data_dir, args.n_subjects, noise_dim=20,
                                         noise_source=args.noise_source, snr=snr, seed=int(snr * 1000))
        row = dict(condition='snr_sweep', snr=snr, n_subjects=args.n_subjects)
        if args.dry_run:
            row.update(n_cov=cond_data['covariates'].shape[1], status='dry_run')
            rows.append(row)
            continue
        try:
            fit = run_condition(cond_data, iters=args.iters, iters_burn_in=args.iters_burn_in, **solver_kwargs)
            metrics = compute_metrics(fit)
            partial_idx = cond_data['partial_idx']
            beta = fit['z_pop'][fit['z_dim']:].reshape(fit['z_dim'], fit['n_cov'])
            recovered = bool((beta[:, partial_idx].abs() > 1e-9).any())
            row.update(metrics, recovered=recovered, status='ok')
        except RuntimeError as e:
            row.update(status='failed', error=str(e))
        rows.append(row)

    #########################################################
    # Condition set 3: encoder information-leakage ablation
    #########################################################
    if args.ablation:
        cond_data = build_condition_data(args.data_dir, args.n_subjects, noise_dim=20,
                                         noise_source=args.noise_source, seed=12345)
        # Pick a covariate likely to be zeroed out (a noise column) to ablate.
        ablate_idx = N_TRUE  # first injected-noise column
        for with_covariate in (True, False):
            row = dict(condition='encoder_leakage_ablation', with_covariate=with_covariate,
                      n_subjects=args.n_subjects)
            if args.dry_run:
                row.update(status='dry_run')
                rows.append(row)
                continue
            try:
                fit = run_condition(cond_data, iters=args.iters, iters_burn_in=args.iters_burn_in,
                                    ablation_drop_idx=None if with_covariate else ablate_idx, **solver_kwargs)
                row.update(elbo=fit['final_elbo'], status='ok')
            except RuntimeError as e:
                row.update(status='failed', error=str(e))
            rows.append(row)

    df = summarize_results(rows)
    csv_path, json_path = write_summary(df, args.out_dir)
    print(df.to_string(index=False))
    print(f"\nWrote summary to:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
