"""
experiments.py
--------------
Experiment runners (F, G, H, H_ddpm, H_sizes, N, R2, CFG, CTN) and the
multi-seed aggregation helper.

Each run_* function accepts a master_gen (DataGenerator with shared W_y/W_u)
and returns a result dict ready for reporting.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from config import (
    SEED_BASE, SEEDS, N_SEEDS_G, N_SEEDS_H, N_SEEDS_N, N_SEEDS_R2,
    EPOCHS_MAIN, EPOCHS_SWEEP,
    NOISE_DIMS_G, OOD_SCALES, GATE_COEFFS, ND_H, ND_F,
    TRAIN_SIZES, LR, D_MODEL, DATA_DIM, DEVICE,
    LATENT_DIMS_R2L, BETAS_R2L, ND_R2L, N_SEEDS_R2L,
    NOISE_CONFIGS_G, NOISE_CONFIGS_H, NOISE_CONFIGS_R2,
    TRUE_COV_NAMES,
)
from data import DataGenerator, get_sobol_result
from models import DiffusionSchedule, ControlNetBlock
from training import (
    train_ldm, train_ldm_mine, train_cvae, r2_train_and_test,
    _tick, _done,
)
from ood import ood_sensitivity_step, ood_sensitivity_full_ddpm
from metrics import spearman_corr
from sobol import attention_alignment_sobol


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── Multi-seed aggregation ────────────────────────────────────────────────────

def _multi_seed_run(seeds, fn, aggregate_keys):
    """
    Run fn(seed) for each seed, return {key: (mean, std)} for each aggregate key.
    """
    raw = defaultdict(list)
    for seed in seeds:
        set_seed(seed)
        r = fn(seed)
        for k in aggregate_keys:
            raw[k].append(r[k])
    return {k: (float(np.mean(raw[k])), float(np.std(raw[k]))) for k in aggregate_keys}


# ── Experiment F: gate-coefficient sweep ──────────────────────────────────────

def run_F(master_gen):
    print(f"\n-- Exp F: gate-coeff sweep  [noise_dim={ND_F}]")
    gd  = master_gen.make_loaders(ND_F)
    res = {}
    for c in GATE_COEFFS:
        _tick(f"coeff={c}")
        r = train_ldm_mine(gd, noise_dim=ND_F, gate_sup_coeff=c,
                           attention_type='sparsemax', epochs=EPOCHS_SWEEP)
        res[c] = r
        _done()
    return res


# ── Experiment G: sparsity mapping ───────────────────────────────────────────

def run_G(master_gen):
    """
    5 attention types x 3 PK noise configurations (clinical-only, RNAseq-only,
    combined), 5 seeds.

    Tests whether zero_true ever selectively identifies the least Sobol-
    important true covariate as expendable while preserving the dominant one
    -- a pharmacologically meaningful sparse-attention finding not testable
    in the linear synthetic model. The dominant covariate (expected: SNP,
    ~90% per the literature-derived prior) and least-important one (expected:
    Albumin, ~0.7%) are determined dynamically from data.get_sobol_result()
    rather than hardcoded, so this also serves as an empirical check of that
    prior against the actual generated population.

    Returns {attn: {config_label: {metric: (m,s), 'zero_per_slot': [(m,s) x n_true],
                                   'alignment': {obs_dominance_ratio, exp_dominance_ratio,
                                                dominant_is_top1, spearman_rho,
                                                calibration} (m,s) per field}}}.
    TRUE_COV_NAMES gives the per-slot order.
    """
    print("\n-- Exp G: sparsity mapping  [5 seeds, all attention types, PK noise configs]")
    attn_types = ['softmax', 'sparsemax', 'entmax12', 'entmax15', 'entmax175']
    agg_keys   = ['lift_true', 'zero_true', 'zero_noise', 'H_norm']
    sobol      = get_sobol_result()
    S_rel      = sobol['S_i_relative']
    dom_name   = max(S_rel, key=S_rel.get)
    least_name = min(S_rel, key=S_rel.get)
    dom_idx, least_idx = TRUE_COV_NAMES.index(dom_name), TRUE_COV_NAMES.index(least_name)
    print(f"  (dominant covariate: {dom_name} S_rel={S_rel[dom_name]:.3f}; "
          f"least important: {least_name} S_rel={S_rel[least_name]:.3f})")
    results    = {a: {} for a in attn_types}

    for a in attn_types:
        print(f"  {a}")
        for nd, group in NOISE_CONFIGS_G:
            label = f"{group}(nd={nd})"
            per_scalar = {k: [] for k in agg_keys}
            per_slot   = []
            per_align  = defaultdict(list)
            for seed in N_SEEDS_G:
                set_seed(seed)
                gen = DataGenerator(seed=seed + 2000,
                                    fixed_W_y=master_gen.W_y,
                                    fixed_W_u=master_gen.W_u)
                gd  = gen.make_loaders(nd, noise_group=group)
                r   = train_ldm(gd, noise_dim=nd, epochs=EPOCHS_SWEEP, attention_type=a)
                for k in agg_keys:
                    per_scalar[k].append(r[k])
                per_slot.append(r['zero_per_slot'])
                align = attention_alignment_sobol(r['attn_per_token'][:len(TRUE_COV_NAMES)],
                                                  sobol, TRUE_COV_NAMES)
                for k, v in align.items():
                    if isinstance(v, (int, float, bool)):
                        per_align[k].append(v)
            stats = {k: (float(np.mean(per_scalar[k])), float(np.std(per_scalar[k])))
                     for k in agg_keys}
            slot_arr = np.stack(per_slot)   # (seeds, n_true)
            stats['zero_per_slot'] = list(zip(slot_arr.mean(0).tolist(), slot_arr.std(0).tolist()))
            stats['alignment'] = {k: (float(np.nanmean(v)), float(np.nanstd(v)))
                                  for k, v in per_align.items()}
            results[a][label] = stats
            print(f"    {label}: lift={stats['lift_true'][0]:.3f}+/-{stats['lift_true'][1]:.3f}  "
                  f"zero_{least_name}={stats['zero_per_slot'][least_idx][0]:.3f}  "
                  f"zero_{dom_name}={stats['zero_per_slot'][dom_idx][0]:.3f}  "
                  f"dominance_ratio={stats['alignment']['obs_dominance_ratio'][0]:.2f}")
    return results


# ── Experiment H: OOD perturbation — full DDPM + single-step ─────────────────

def run_H(master_gen):
    """
    Central PK experiment. 3 key models x 4 PK noise configurations (none,
    clinical nd=3, RNAseq nd=250, combined nd=253) x 5 seeds, full stochastic
    DDPM (T=100, shared-noise fix so NS_ddpm@scale=1 == 0 by construction).

    Tests whether OOD protection scales gracefully from correlated low-dim
    clinical noise (sex, weight, race) to uncorrelated high-dim transcriptomic
    noise. If MINE+softmax maintains NS_ddpm ~ 0 at nd=253 (combined), that
    validates the architecture for a realistic mixed PK+omics noise regime.

    Returns {model: {config_label: {'ns': {scale: (mean,std)},
                                    'alignment': {field: (mean,std)}}}}.
    (alignment is omitted for the noise_dim=0 'none' config -- no true-slot
    attention to evaluate distinctly from noise there.)
    """
    print("\n-- Exp H: OOD perturbation (full DDPM, PK noise configs)  [5 seeds]")
    model_configs = [
        ('softmax',               False, 'softmax',   'kv'),
        ('MINE+softmax',          True,  'softmax',   'kv'),
        ('MINE+sparsemax(vonly)', True,  'sparsemax', 'v_only'),
    ]
    results = {name: {} for name, *_ in model_configs}
    for name, use_mine, attn_type, gate_mode in model_configs:
        print(f"  {name}")
        for nd, group in NOISE_CONFIGS_H:
            label = f"{group}(nd={nd})"
            per_ns    = {s: [] for s in OOD_SCALES}
            per_align = defaultdict(list)
            for seed in N_SEEDS_H:
                set_seed(seed)
                gen = DataGenerator(seed=seed + 2000,
                                    fixed_W_y=master_gen.W_y,
                                    fixed_W_u=master_gen.W_u)
                gd = gen.make_loaders(nd, noise_group=group)
                if use_mine:
                    r = train_ldm_mine(gd, noise_dim=nd, epochs=EPOCHS_MAIN,
                                       attention_type=attn_type, gate_mode=gate_mode)
                else:
                    r = train_ldm(gd, noise_dim=nd, epochs=EPOCHS_MAIN,
                                  attention_type=attn_type, gate_mode=gate_mode)
                ood = ood_sensitivity_full_ddpm(r['model'], r['sched'], gen, nd,
                                               OOD_SCALES, noise_group=group)
                for s in OOD_SCALES:
                    per_ns[s].append(ood[s]['norm_sensitivity'])
                if nd > 0:
                    align = attention_alignment_sobol(r['attn_per_token'][:len(TRUE_COV_NAMES)],
                                                      get_sobol_result(), TRUE_COV_NAMES)
                    for k, v in align.items():
                        if isinstance(v, (int, float, bool)):
                            per_align[k].append(v)
            entry = {'ns': {s: (float(np.mean(per_ns[s])), float(np.std(per_ns[s])))
                           for s in OOD_SCALES}}
            if per_align:
                entry['alignment'] = {k: (float(np.nanmean(v)), float(np.nanstd(v)))
                                      for k, v in per_align.items()}
            results[name][label] = entry
            print(f"    {label}: ns_ddpm@100={entry['ns'][100][0]:.4f}")
    return results


# ── Experiment H_sizes: training-size sweep ───────────────────────────────────

def run_H_sizes(master_gen):
    """3 models x 5 sizes x 5 seeds. Returns {train_size: {model: {ns10, ns100, lift1}}}."""
    print(f"\n-- Exp H_sizes  [noise_dim={ND_H}, 5 seeds x 5 sizes]")
    model_configs = [
        ('softmax',               False, 'softmax',  'kv'),
        ('MINE+softmax',          True,  'softmax',  'kv'),
        ('MINE+sparsemax',        True,  'sparsemax', 'kv'),
        ('MINE+sparsemax(vonly)', True,  'sparsemax', 'v_only'),
    ]
    all_results = {ts: defaultdict(list) for ts in TRAIN_SIZES}
    for seed in SEEDS:
        set_seed(seed)
        gen = DataGenerator(seed=seed + 2000,
                            fixed_W_y=master_gen.W_y,
                            fixed_W_u=master_gen.W_u)
        sched = DiffusionSchedule()
        for ts in TRAIN_SIZES:
            print(f"  seed={seed}  train_size={ts}")
            gd = gen.make_loaders(ND_H, train_size=ts)
            for name, use_mine, attn, gm in model_configs:
                _tick(name)
                if use_mine:
                    r = train_ldm_mine(gd, noise_dim=ND_H, epochs=EPOCHS_MAIN,
                                       attention_type=attn, gate_mode=gm)
                else:
                    r = train_ldm(gd, noise_dim=ND_H, epochs=EPOCHS_MAIN,
                                  attention_type=attn, gate_mode=gm)
                
                ood = ood_sensitivity_full_ddpm(r['model'], r['sched'], gen, ND_H,
                                               scales=[10, 100],
                                               n_steps=20, eta=0.0)  # deterministic DDIM
                align = attention_alignment_sobol(r['attn_per_token'][:len(TRUE_COV_NAMES)],
                                                  get_sobol_result(), TRUE_COV_NAMES)
                all_results[ts][name].append({
                    'ns10' : ood[10]['norm_sensitivity'],
                    'ns100': ood[100]['norm_sensitivity'],
                    'dominance_ratio': align['obs_dominance_ratio'],
                    'dominant_is_top1': align['dominant_is_top1'],
                })
                _done()

    agg = {}
    for ts in TRAIN_SIZES:
        agg[ts] = {}
        for name, _, _, _ in model_configs:
            vals = all_results[ts][name]
            agg[ts][name] = {
                'ns10' : (np.mean([v['ns10']  for v in vals]),
                          np.std( [v['ns10']  for v in vals])),
                'ns100': (np.mean([v['ns100'] for v in vals]),
                          np.std( [v['ns100'] for v in vals])),
                'dominance_ratio': (np.mean([v['dominance_ratio'] for v in vals]),
                                   np.std( [v['dominance_ratio'] for v in vals])),
                'dominant_is_top1_frac': float(np.mean([v['dominant_is_top1'] for v in vals])),
            }
    return agg


# ── Experiment N: MINE+softmax baseline ──────────────────────────────────────

def run_N(master_gen):
    """
    MINE+softmax across noise_dims, 3 seeds.

    Per-slot diagnostic: instead of just the aggregate lift_true, track the
    learned per-slot true-token attention share (pw_true analogue,
    attn_per_token[:n_true]). Two complementary alignment checks against the
    Sobol ground truth (data.get_sobol_result(); computed directly from the
    generated population rather than the literature-derived Var(alpha*log(X/ref))
    figures, which it should approximately recover -- see sobol.py and the
    project_PK_ground_truth_importance memory):
      - spearman_rho_variance: rank correlation vs the Sobol S_i_relative
        ordering.
      - alignment: the tiered diagnostic (attention_alignment_sobol) --
        primary dominance ratio (dominant covariate vs mean of the rest),
        secondary dominant-is-top1 check, tertiary rank correlation.
    """
    print("\n-- Exp N: MINE+softmax baseline")
    agg_keys = ['zero_noise', 'H_norm', 'mine_true', 'mine_noise']
    sobol    = get_sobol_result()
    true_cov_importance = [sobol['S_i_relative'][n] for n in TRUE_COV_NAMES]
    res = {}
    for nd in NOISE_DIMS_G:
        per_scalar = {k: [] for k in agg_keys}
        per_slot   = []
        per_rho    = []
        per_align  = defaultdict(list)
        for seed in N_SEEDS_N:
            set_seed(seed)
            gen = DataGenerator(seed=seed + 2000,
                                fixed_W_y=master_gen.W_y,
                                fixed_W_u=master_gen.W_u)
            gd  = gen.make_loaders(nd)
            r   = train_ldm_mine(gd, noise_dim=nd, epochs=EPOCHS_SWEEP,
                                 attention_type='softmax', gate_mode='kv')
            for k in agg_keys:
                per_scalar[k].append(r[k])
            slot = r['attn_per_token'][:len(TRUE_COV_NAMES)]
            per_slot.append(slot)
            per_rho.append(spearman_corr(slot, true_cov_importance))
            align = attention_alignment_sobol(slot, sobol, TRUE_COV_NAMES)
            for k, v in align.items():
                if isinstance(v, (int, float, bool)):
                    per_align[k].append(v)
        stats = {k: (float(np.mean(per_scalar[k])), float(np.std(per_scalar[k])))
                 for k in agg_keys}
        slot_arr = np.stack(per_slot)
        stats['per_slot_attn']        = list(zip(slot_arr.mean(0).tolist(), slot_arr.std(0).tolist()))
        stats['spearman_rho_variance'] = (float(np.nanmean(per_rho)), float(np.nanstd(per_rho)))
        stats['alignment'] = {k: (float(np.nanmean(v)), float(np.nanstd(v)))
                              for k, v in per_align.items()}
        res[nd] = stats
        print(f"  nd={nd}: dominance_ratio={stats['alignment']['obs_dominance_ratio'][0]:.2f}  "
              f"rho_sobol={stats['alignment']['spearman_rho'][0]:.3f}")
    return res


# ── Experiment R2: CVAE R2 diagnostic ────────────────────────────────────────

def run_R2(master_gen):
    """
    Simplified CVAE train vs test R2: 3 PK noise configurations (none, nd=0;
    clinical, nd=3; combined, nd=253), fixed latent_dim=16, 3 seeds.
    Confirms CVAE robustness in the PK context (expected: excess_R2 ~ 0 once
    noise_dim is large) without the full noise_dim sweep.
    """
    print("\n-- Diag R2: CVAE train vs test R2 (PK noise configs, latent_dim=16)")
    res = {}
    for nd, group in NOISE_CONFIGS_R2:
        label = f"{group}(nd={nd})"
        row = {}
        for lbl, kw in [("CVAE", {}), ("CVAE+CondDrop", dict(cond_dropout_p=0.5))]:
            metrics_list = []
            for seed in N_SEEDS_R2:
                set_seed(seed)
                gen = DataGenerator(seed=seed + 2000,
                                    fixed_W_y=master_gen.W_y,
                                    fixed_W_u=master_gen.W_u)
                gd     = gen.make_loaders(nd, noise_group=group)
                cvae_r = train_cvae(gd, noise_dim=nd, latent_dim=16, **kw)
                metrics_list.append(r2_train_and_test(cvae_r, gd))
            agg = {}
            for k in metrics_list[0]:
                vals   = [m[k] for m in metrics_list]
                agg[k] = (float(np.mean(vals)), float(np.std(vals)))
            row[lbl] = agg
        res[label] = row
    return res


# ── Experiment R2L: CVAE latent-dim x beta sweep ─────────────────────────────

def run_R2_latent(master_gen):
    """
    Characterise the CVAE by sweeping two axes independently, 3 seeds each.

    Latent sweep  — vary latent_dim over LATENT_DIMS_R2L, fix beta=0.01.
    Beta sweep    — vary beta over BETAS_R2L, fix latent_dim=LATENT_DIMS_R2L[-1].

    Returns:
        {
          'latent_sweep': {latent_dim: {metric: (mean, std), ...}},
          'beta_sweep':   {beta:       {metric: (mean, std), ...}},
        }
    """
    print(f"\n-- Exp R2L: CVAE latent-dim x beta sweep  [noise_dim={ND_R2L}, 3 seeds]")
    agg_keys  = ['r2_excess', 'n_active_dims', 'final_recon',
                 'pool_weights_true', 'pool_weights_noise']
    beta_fixed   = 0.01
    latent_fixed = LATENT_DIMS_R2L[-1]   # largest latent_dim for beta sweep

    def _run_one(seed, latent_dim, beta):
        gen = DataGenerator(seed=seed + 3000,
                            fixed_W_y=master_gen.W_y,
                            fixed_W_u=master_gen.W_u)
        gd  = gen.make_loaders(ND_R2L)
        r   = train_cvae(gd, noise_dim=ND_R2L, beta=beta, latent_dim=latent_dim)
        return {
            'r2_excess':          r['r2_excess'],
            'n_active_dims':      float(r['n_active_dims']),
            'final_recon':        r['final_recon'],
            'pool_weights_true':  r['pool_weights_true'],
            'pool_weights_noise': r['pool_weights_noise'],
        }

    # ── Latent sweep ──────────────────────────────────────────────────────────
    print(f"  latent sweep (beta={beta_fixed})")
    latent_sweep = {}
    for ld in LATENT_DIMS_R2L:
        _tick(f"latent_dim={ld}")
        raw = {k: [] for k in agg_keys}
        for seed in N_SEEDS_R2L:
            set_seed(seed)
            r = _run_one(seed, ld, beta_fixed)
            for k in agg_keys:
                raw[k].append(r[k])
        latent_sweep[ld] = {k: (float(np.mean(raw[k])), float(np.std(raw[k])))
                            for k in agg_keys}
        _done()
        print(f"    excess_r2={latent_sweep[ld]['r2_excess'][0]:.3f}  "
              f"n_act={latent_sweep[ld]['n_active_dims'][0]:.1f}  "
              f"recon={latent_sweep[ld]['final_recon'][0]:.4f}")

    # ── Beta sweep ────────────────────────────────────────────────────────────
    print(f"  beta sweep (latent_dim={latent_fixed})")
    beta_sweep = {}
    for beta in BETAS_R2L:
        _tick(f"beta={beta}")
        raw = {k: [] for k in agg_keys}
        for seed in N_SEEDS_R2L:
            set_seed(seed)
            r = _run_one(seed, latent_fixed, beta)
            for k in agg_keys:
                raw[k].append(r[k])
        beta_sweep[beta] = {k: (float(np.mean(raw[k])), float(np.std(raw[k])))
                            for k in agg_keys}
        _done()
        print(f"    excess_r2={beta_sweep[beta]['r2_excess'][0]:.3f}  "
              f"n_act={beta_sweep[beta]['n_active_dims'][0]:.1f}  "
              f"recon={beta_sweep[beta]['final_recon'][0]:.4f}")

    return {'latent_sweep': latent_sweep, 'beta_sweep': beta_sweep}


# ── Experiment CFG: Classifier-Free Guidance ──────────────────────────────────

def run_CFG(master_gen):
    """CFG baseline at guidance scales 1.0–5.0."""
    print(f"\n-- Exp CFG  [noise_dim={ND_H}]")
    gd = master_gen.make_loaders(ND_H)
    _tick("Train CFG model")
    r = train_ldm(gd, noise_dim=ND_H, epochs=EPOCHS_MAIN,
                  attention_type='softmax', use_cfg=True, cfg_prob=0.1)
    _done()
    sched = DiffusionSchedule()
    r['model'].eval()
    res = {}
    for gs in [1.0, 1.5, 2.0, 3.0, 5.0]:
        _tick(f"guidance={gs}")
        wrapper = lambda x, y, t, gs=gs: r['model'](x, y, t, guidance_scale=gs)
        res[gs] = ood_sensitivity_step(wrapper, sched, master_gen, ND_H,
                                        scales=[1, 5, 10, 25, 50, 100])
        _done()
    return {'cfg_model': r, 'ood_by_guidance': res}


# ── Experiment CTN: ControlNet baseline ──────────────────────────────────────

def run_CTN(master_gen):
    """Simplified ControlNet adapter trained on top of a frozen base LDM."""
    print(f"\n-- Exp CTN: ControlNet  [noise_dim={ND_H}]")
    gd = master_gen.make_loaders(ND_H)
    _tick("Train base LDM")
    base = train_ldm(gd, noise_dim=ND_H, epochs=EPOCHS_SWEEP)
    _done()

    cnet  = ControlNetBlock(base['model']).to(DEVICE)
    opt   = optim.Adam(cnet.parameters(), lr=LR)
    sched = DiffusionSchedule()
    base['model'].eval()
    cnet.train()

    for ep in range(EPOCHS_SWEEP):
        for X, Y in gd['train']:
            X, Y     = X.to(DEVICE), Y.to(DEVICE)
            t        = torch.randint(0, sched.T, (X.shape[0],), device=DEVICE)
            x_t, eps = sched.q_sample(X, t)
            with torch.no_grad():
                h_b = (base['model'].input_proj(x_t).unsqueeze(1) +
                       base['model'].t_emb(t).unsqueeze(1))
                tok  = base['model'].per_cond(Y)
                h_ca, _, _, _ = base['model'].cross_attn(h_b, tok)
                base_pred = base['model'].output_proj((h_b + h_ca).squeeze(1))
            ctrl_out = cnet(x_t, Y, t)
            loss = F.mse_loss(base_pred + ctrl_out, eps)
            opt.zero_grad(); loss.backward(); opt.step()

    cnet.eval()

    def ctrl_wrapper(x_t, y, t):
        with torch.no_grad():
            h_b  = (base['model'].input_proj(x_t).unsqueeze(1) +
                    base['model'].t_emb(t).unsqueeze(1))
            tok  = base['model'].per_cond(y)
            h_ca, attn, gates, _ = base['model'].cross_attn(h_b, tok)
            bp   = base['model'].output_proj((h_b + h_ca).squeeze(1))
            co   = cnet(x_t, y, t)
            return bp + co, attn, gates, torch.tensor(0.0, device=DEVICE)

    ood = ood_sensitivity_step(ctrl_wrapper, sched, master_gen, ND_H)
    return {'controlnet': cnet, 'ood': ood}
