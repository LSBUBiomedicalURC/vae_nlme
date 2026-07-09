"""
run_stress_test.py
---------------------

Drives the uninformative-covariate stress test ACROSS all three case studies
(Tacrolimus, Paclitaxel, Theophylline), by invoking each dataset's own
Main/<dataset>.py as a subprocess with --n_batch / --n_cov (or
--n_noise_cov for theophylline, which has only 2 real covariates) and
reading back a small --results_json summary (selected covariates, OFV,
structural population parameter estimates) -- rather than re-deriving a
separate in-process training loop per dataset (as Main/stress_test_covariates.py
does for Tacrolimus alone). This reuses the already-verified Main scripts
directly, at the cost of one subprocess + one cvxpy solve per condition.

Why covariate growth is implemented differently per dataset
-------------------------------------------------------------
- Tacrolimus / Paclitaxel: the real covariate files already contain a large
  RNAseq noise block (250 columns) after the true + clinical covariates, in a
  fixed column order (true first). So sweeping --n_cov directly grows the
  number of uninformative covariates using REAL data, no synthetic injection
  needed.
- Theophylline: only has 2 real covariates (weight, sex) total, so growing
  n_cov requires Main/theophylline.py's --n_noise_cov synthetic-injection
  knob instead.

Practical run-time note (see solver_utils.py's empirically measured MIQP
scaling): pop_parameter's covariate-selection step is solved ONCE PER
TRAINING ITERATION, and its solve time blows up steeply once
z_dim * n_cov exceeds roughly 150-200 (Tacrolimus z_dim=2: ~breaks down
around n_cov=130-160; Paclitaxel z_dim=6 needs ~5x fewer covariates for the
same variable count). This script's default covariate sweeps are kept inside
the empirically fast range (verified in this repo's session) so a full sweep
finishes in reasonable wall-clock time with the free SCIP solver -- they do
NOT reach the full 250+-noise-covariate regime. Use --tacro_n_cov /
--pacli_n_cov / --theo_n_noise_cov to push further at your own risk (and a
longer --timeout).
"""
#########################################################
# Import
#########################################################
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)

SCRIPTS = {
    'tacrolimus': 'tacrolimus.py',
    'paclitaxel': 'paclitaxel.py',
    'theophylline': 'theophylline.py',
    'quinidine': 'quinidine.py',
    'warfarin': 'warfarin.py',
}

DEFAULT_SWEEPS = {
    'tacrolimus': [4, 8, 15, 30, 50, 80],
    'paclitaxel': [4, 8, 12, 20, 30],
    'theophylline': [0, 5, 10, 20, 30],  # n_noise_cov (added ON TOP of the 2 real covariates)
    'quinidine': [10, 15, 20, 30, 50, 80],  # total n_cov (10 real + noise)
    'warfarin': [3, 8, 13, 23, 33],  # total n_cov (3 real + 0/5/10/20/30 noise)
}


def save_summary_table(rows, save_path):
    """Build and save the experiment summary table from a list of result dicts.

    Columns written:
      drug  n_noise  n_true  n_true_selected  n_noise_selected  AIC  BIC  BICc

    Information criteria are computed from ofv_lin (= -2*LL, linearisation):
      k  = 2*z_dim + 1 + n_selected   (structural + IIV + residual + selected betas)
      AIC  = OFV + 2*k
      BIC  = OFV + k*ln(N)            N = n_batch (number of subjects)
      BICc = OFV + ln(N)*(z_dim + n_selected) + ln(n_obs)*(z_dim + 1)
                                       n_obs = total observations

    Conditions where ofv_lin is None (--skip_ll) have NaN information criteria.
    Conditions that timed out or failed are included with NaN numeric columns.
    """
    summary_rows = []
    for r in rows:
        status  = r.get('status', 'unknown')
        dataset = r.get('dataset', r.get('drug', '?'))
        ok      = status == 'ok'

        n_cov         = r.get('n_cov',     r.get('sweep_value', None)) if ok else r.get('sweep_value')
        n_true_cov    = r.get('n_true_cov', None)
        n_noise_cov   = r.get('n_noise_cov', None)
        selected      = r.get('selected', [])
        z_dim         = r.get('z_dim',    None)
        n_batch       = r.get('n_batch',  r.get('n_batch', None))
        n_obs         = r.get('n_observations', None)
        ofv           = r.get('ofv_lin',  None)
        n_selected    = r.get('n_selected', None)
        time_elapsed   = r.get('time_elapsed', None)

        # ---- split selected into true vs noise ----
        n_true   = n_true_cov if n_true_cov is not None else None
        n_noise  = (n_cov - n_true_cov) if (n_cov is not None and n_true_cov is not None) else n_noise_cov

        if selected and n_true_cov is not None:
            n_true_sel  = int(sum(selected[:n_true_cov]))
            n_noise_sel = int(sum(selected[n_true_cov:]))
        else:
            n_true_sel = n_noise_sel = None

        # ---- information criteria ----
        aic = bic = bicc = float('nan')
        if ofv is not None and z_dim is not None and n_selected is not None:
            k   = 2 * z_dim + 1 + n_selected
            aic = ofv + 2 * k
            if n_batch:
                bic  = ofv + k * math.log(n_batch)
                if n_obs:
                    bicc = ofv + math.log(n_batch) * (z_dim + n_selected) + math.log(n_obs) * (z_dim + 1)

        summary_rows.append(dict(
            drug=dataset,
            status=status,
            n_true=n_true,
            n_noise=n_noise,
            n_true_selected=n_true_sel,
            n_noise_selected=n_noise_sel,
            AIC=round(aic, 2) if not math.isnan(aic) else None,
            BIC=round(bic, 2) if not math.isnan(bic) else None,
            BICc=round(bicc, 2) if not math.isnan(bicc) else None,
            time_elapsed=round(time_elapsed, 2) if time_elapsed is not None else None
        ))

    df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def run_one(dataset, n_batch, sweep_value, iters, iters_burn_in, solver,
           allow_incompatible_solver, timeout, seed=1, plot_dir=None,
           standardise_C=False, miqp_every=1, skip_ll=False, theo_noise_source=None):
    """
    Runs Main/<dataset>.py as a subprocess for one (n_batch, sweep_value, seed)
    condition and returns a result dict (parsed from --results_json, or a
    'failed' row with the captured stderr tail on any error/timeout).
    """
    script_path = os.path.join(HERE, SCRIPTS[dataset])
    results_fd, results_path = tempfile.mkstemp(suffix='.json')
    os.close(results_fd)
    os.remove(results_path)  # subprocess creates it; just reserve the path

    args = [sys.executable, script_path,
           '--iters', str(iters), '--iters_burn_in', str(iters_burn_in),
           '--solver', solver, '--results_json', results_path,
           '--n_batch', str(n_batch), '--seed', str(seed)]#, '--skip_ll']
    if allow_incompatible_solver:
        args.append('--allow_incompatible_solver')
    if plot_dir:
        args += ['--plot_dir', plot_dir]
    if standardise_C:
        args.append('--standardise_C')
    if miqp_every != 1:
        args += ['--miqp_every', str(miqp_every)]
    if skip_ll:
        args.append('--skip_ll')
    if dataset == 'theophylline':
        args += ['--n_noise_cov', str(sweep_value)]
        if theo_noise_source is not None:
            args += ['--noise_source', theo_noise_source]
    else:
        args += ['--n_cov', str(sweep_value)]

    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'        # no GUI windows / plt.show() blocking in batch runs
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # SCIP/torch OpenMP runtime conflict workaround

    row = dict(dataset=dataset, n_batch=n_batch, sweep_value=sweep_value, solver=solver, seed=seed)
    try:
        start_exp = time.perf_counter()
        proc = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True,
                              timeout=timeout)
        end_exp = time.perf_counter()

    except subprocess.TimeoutExpired:
        row.update(status='timeout', error=f'exceeded --timeout={timeout}s')
        return row

    if proc.returncode != 0:
        row.update(status='failed', error=proc.stderr[-3000:])
        return row

    if not os.path.exists(results_path):
        row.update(status='failed', error='process exited 0 but produced no --results_json '
                                          '(unexpected -- check stdout/stderr manually)',
                  stdout_tail=proc.stdout[-1000:])
        return row

    with open(results_path) as f:
        result = json.load(f)
    os.remove(results_path)
    row.update(result, status='ok')
    row.update(result, time_elapsed=end_exp - start_exp)
    return row


def save_aggregated_table(rows, save_path):
    """Aggregate per-seed rows into one row per (drug, n_noise) condition.

    For each condition, reports:
      - n_seeds_ok            : number of seeds that completed without error
      - n_true_sel_mean/std   : mean ± SD of true-covariate selection count
      - n_noise_sel_mean/std  : mean ± SD of noise-covariate selection count
      - BICc_mean/std         : mean ± SD of BICc across seeds
      - sel_freq_<cov>        : per-covariate selection frequency (fraction of ok seeds)

    The per-covariate frequency columns make it easy to distinguish robustly
    selected covariates (freq=1.0) from marginally selected ones (freq<1.0) and
    noise covariates that occasionally leak in, without collapsing that
    information into a single count.
    """
    import collections, numpy as _np

    # group ok rows by (dataset, sweep_value)
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get('status') == 'ok':
            groups[(r['dataset'], r['sweep_value'])].append(r)

    agg_rows = []
    for (dataset, sweep_value), grp in sorted(groups.items()):
        n_ok = len(grp)
        ref  = grp[0]

        n_true_cov  = ref.get('n_true_cov')
        n_cov_total = ref.get('n_cov', sweep_value)
        cov_names   = ref.get('covariate_names', [f'cov{i}' for i in range(n_cov_total or 0)])
        
        
        # Per-seed selection vectors (list of bool per covariate)
        time_elapsed = [r.get('time_elapsed', _np.nan) for r in grp]
        sel_matrix = [r.get('selected', []) for r in grp]

        # Pad/truncate so all rows have the same length
        max_len = max((len(s) for s in sel_matrix), default=0)
        sel_matrix = [s + [False] * (max_len - len(s)) for s in sel_matrix]
        sel_arr = _np.array(sel_matrix, dtype=float)  # shape (n_ok, n_cov)

        sel_freq = sel_arr.mean(axis=0) if sel_arr.size else _np.array([])

        n_true_sel  = sel_arr[:, :n_true_cov].sum(axis=1) if (n_true_cov and sel_arr.size) else _np.array([_np.nan])
        n_noise_sel = sel_arr[:, n_true_cov:].sum(axis=1) if (n_true_cov and sel_arr.size) else _np.array([_np.nan])

        biccs = _np.array([r.get('bicc', _np.nan) for r in grp])
        # BICc may come from summary_table computation, not the raw JSON; recompute if needed
        # (use the same formula as save_summary_table)
        biccs_computed = []
        for r in grp:
            ofv    = r.get('ofv_lin')
            z_dim  = r.get('z_dim')
            n_sel  = r.get('n_selected')
            n_bat  = r.get('n_batch')
            n_obs  = r.get('n_observations')
            if ofv is not None and z_dim and n_sel is not None and n_bat and n_obs:
                biccs_computed.append(ofv + math.log(n_bat) * (z_dim + n_sel) + math.log(n_obs) * (z_dim + 1))
            else:
                biccs_computed.append(_np.nan)
        biccs = _np.array(biccs_computed)

        row = dict(
            drug=dataset,
            n_noise=(n_cov_total - n_true_cov) if (n_cov_total and n_true_cov) else sweep_value,
            n_true=n_true_cov,
            n_seeds_ok=n_ok,
            n_true_sel_mean=round(float(_np.nanmean(n_true_sel)), 3),
            n_true_sel_std=round(float(_np.nanstd(n_true_sel)), 3),
            n_noise_sel_mean=round(float(_np.nanmean(n_noise_sel)), 3),
            n_noise_sel_std=round(float(_np.nanstd(n_noise_sel)), 3),
            BICc_mean=round(float(_np.nanmean(biccs)), 2) if not _np.all(_np.isnan(biccs)) else None,
            BICc_std=round(float(_np.nanstd(biccs)), 2) if not _np.all(_np.isnan(biccs)) else None,
            time_mean=round(float(_np.nanmean(time_elapsed)), 2) if not _np.all(_np.isnan(time_elapsed)) else None,
            time_std=round(float(_np.nanstd(time_elapsed)), 2) if not _np.all(_np.isnan(time_elapsed)) else None,
           )
        # Per-covariate selection frequency columns
        for i, freq in enumerate(sel_freq):
            name = cov_names[i] if i < len(cov_names) else f'cov{i}'
            row[f'sel_freq_{name}'] = round(float(freq), 3)

        agg_rows.append(row)

    df = pd.DataFrame(agg_rows)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run the uninformative-covariate stress test across Tacrolimus, "
                    "Paclitaxel, and Theophylline.")
    parser.add_argument('--datasets', nargs='+', choices=list(SCRIPTS), default=list(SCRIPTS),
                        help="Which case studies to run (default: all three).")
    parser.add_argument('--n_batch', type=int, default=30,
                        help="Subjects per run for Tacrolimus/Paclitaxel (default: 30; "
                            "Theophylline always uses its full 12 subjects unless overridden "
                            "via --theo_n_batch).")
    parser.add_argument('--theo_n_batch', type=int, default=None,
                        help="Override --n_batch for theophylline specifically (default: None, "
                            "meaning use all 12 subjects).")
    parser.add_argument('--tacro_n_cov', type=int, nargs='+', default=None,
                        help=f"Override the Tacrolimus n_cov sweep (default: "
                            f"{DEFAULT_SWEEPS['tacrolimus']}).")
    parser.add_argument('--pacli_n_cov', type=int, nargs='+', default=None,
                        help=f"Override the Paclitaxel n_cov sweep (default: "
                            f"{DEFAULT_SWEEPS['paclitaxel']}).")
    parser.add_argument('--theo_n_noise_cov', type=int, nargs='+', default=None,
                        help=f"Override the Theophylline n_noise_cov sweep (default: "
                            f"{DEFAULT_SWEEPS['theophylline']}).")
    parser.add_argument('--quini_n_cov', type=int, nargs='+', default=None,
                        help=f"Override the Quinidine n_cov sweep (default: "
                            f"{DEFAULT_SWEEPS['quinidine']}).")
    parser.add_argument('--warf_n_cov', type=int, nargs='+', default=None,
                        help=f"Override the Warfarin n_cov sweep (default: "
                            f"{DEFAULT_SWEEPS['warfarin']}).")
    parser.add_argument('--theo_noise_source', choices=['iid', 'correlated', 'tgca'], default=None,
                        help="Noise source for Theophylline's injected covariates (default: None, "
                             "meaning use theophylline.py's own default 'iid'). Set to 'tgca' to "
                             "use real RNAseq columns from TGCA_genes.csv instead of synthetic "
                             "lognormal noise.")
    parser.add_argument('--iters', type=int, default=15,
                        help="Training iterations per run (default: 15 -- a fast, "
                            "qualitative-trend setting, NOT a converged fit; raise for a "
                            "real result).")
    parser.add_argument('--iters_burn_in', type=int, default=5)
    parser.add_argument('--solver', default='SCIP',
                        help="cvxpy solver for the covariate-selection MIQP step (default: "
                            "SCIP, the free/no-license-cap option -- see solver_utils.py).")
    parser.add_argument('--allow_incompatible_solver', action='store_true')
    parser.add_argument('--timeout', type=int, default=600,
                        help="Per-run subprocess timeout in seconds (default: 600).")
    parser.add_argument('--n_seeds', type=int, default=5,
                        help="Number of random seeds to run per (drug, n_cov) condition "
                             "(default: 5). Seeds used are 1, 2, ..., n_seeds.")
    parser.add_argument('--out_dir', default=os.path.join(ROOT, 'Plots', 'stress_test_results'))
    parser.add_argument('--plot_dir', default=None,
                        help="If set, pass --plot_dir to each subprocess so convergence figures "
                             "(popParam.pdf, beta.pdf) are saved per condition under this directory. "
                             "Default: None (no figures). Tip: use the same value as --out_dir "
                             "to co-locate figures with the CSV/JSON summary.")
    parser.add_argument('--standardise_C', action='store_true',
                        help="Pass --standardise_C to each subprocess: divides each covariate "
                             "column of C_regression by its across-subject SD before the MIQP. "
                             "Use to test whether selection instability is scale-driven.")
    parser.add_argument('--miqp_every', type=int, default=1,
                        help="Run the covariate-selection MIQP only every N-th iteration in all "
                             "three datasets (default: 1 = every iteration). A value of 5 or 10 "
                             "cuts wall-clock by ~5-10x, which makes Paclitaxel tractable under "
                             "SCIP. See also --pacli_miqp_every for a dataset-specific override.")
    parser.add_argument('--pacli_miqp_every', type=int, default=None,
                        help="Override --miqp_every specifically for Paclitaxel (default: use "
                             "--miqp_every). E.g. --miqp_every 1 --pacli_miqp_every 10 runs "
                             "Tacrolimus/Theophylline at full resolution but speeds up Paclitaxel.")
    parser.add_argument('--skip_ll', action='store_true',
                        help="Pass --skip_ll to every subprocess: skips the post-training "
                             "log-likelihood / EBE computation. Strongly recommended when using "
                             "SCIP or when sweeping many conditions -- LogLikelihood_sample runs "
                             "~1100 ODE forward passes per subject and dominates wall-clock for "
                             "Paclitaxel (z_dim=6). IC columns in the summary table will be NaN.")
    parser.add_argument('--dry_run', action='store_true',
                        help="Print the planned run table without actually launching any "
                            "subprocess.")
    args = parser.parse_args()

    sweeps = {
        'tacrolimus': args.tacro_n_cov or DEFAULT_SWEEPS['tacrolimus'],
        'paclitaxel': args.pacli_n_cov or DEFAULT_SWEEPS['paclitaxel'],
        'theophylline': args.theo_n_noise_cov or DEFAULT_SWEEPS['theophylline'],
        'quinidine': args.quini_n_cov or DEFAULT_SWEEPS['quinidine'],
        'warfarin': args.warf_n_cov or DEFAULT_SWEEPS['warfarin'],
    }
    seeds = list(range(1, args.n_seeds + 1))

    rows = []
    for dataset in args.datasets:
        n_batch = args.theo_n_batch if (dataset == 'theophylline' and args.theo_n_batch) else args.n_batch
        for sweep_value in sweeps[dataset]:
            miqp_every = (args.pacli_miqp_every if (dataset == 'paclitaxel' and args.pacli_miqp_every)
                          else args.miqp_every)
            for seed in seeds:
                if args.dry_run:
                    rows.append(dict(dataset=dataset, n_batch=n_batch, sweep_value=sweep_value,
                                     seed=seed, solver=args.solver, miqp_every=miqp_every,
                                     theo_noise_source=args.theo_noise_source,
                                     status='dry_run'))
                    continue
                print(f"[run_stress_test] {dataset}: n_batch={n_batch} sweep_value={sweep_value} "
                     f"seed={seed}/{args.n_seeds} miqp_every={miqp_every} solver={args.solver} ...", flush=True)
                
                row = run_one(dataset, n_batch, sweep_value, args.iters, args.iters_burn_in,
                             args.solver, args.allow_incompatible_solver, args.timeout,
                             seed=seed, plot_dir=args.plot_dir,
                             standardise_C=args.standardise_C,
                             miqp_every=miqp_every, skip_ll=args.skip_ll,
                             theo_noise_source=args.theo_noise_source)

                print(f"  -> status={row['status']}"
                     + (f" n_selected={row.get('n_selected')}/{row.get('M')}" if row['status'] == 'ok' else
                        f" error={row.get('error', '')[:300]}"), flush=True)
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path  = os.path.join(args.out_dir, 'run_stress_test_all_datasets.csv')
    json_path = os.path.join(args.out_dir, 'run_stress_test_all_datasets.json')
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    summary_path = os.path.join(args.out_dir, 'summary_table.csv')
    df_summary   = save_summary_table(rows, summary_path)

    agg_path  = os.path.join(args.out_dir, 'aggregated_table.csv')
    df_agg    = save_aggregated_table(rows, agg_path)

    print('')
    print('=== PER-SEED SUMMARY TABLE ===')
    print(df_summary.to_string(index=False))
    print('')
    print('=== AGGREGATED TABLE (across seeds) ===')
    print(df_agg.to_string(index=False))
    print(f"\nWrote results to:\n  {csv_path}\n  {json_path}\n  {summary_path}\n  {agg_path}")


if __name__ == "__main__":
    main()
    
