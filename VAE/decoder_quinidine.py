"""
decoder_quinidine.py
----------------------

NEW, additive decoder for the Quinidine dense PK case study (does NOT modify
any existing function in VAE/decoder.py).

Structural model
-----------------
2-compartment model with first-order absorption and an absorption lag time,
matching the parameter set Ayral et al. 2021 (COSSAC paper, Table 1) report
for "Quinidine PK": z_dim = 6 latent PK parameters per subject,
[Tlag, ka, Cl, V1, Q, V2].

States y = [Ad, A1, A2] (depot, central, peripheral amounts), integrated in
lag-shifted time t' = max(t - Tlag, 0):

    dAd/dt' = -ka*Ad
    dA1/dt' = ka*Ad - (Cl/V1 + Q/V1)*A1 + (Q/V2)*A2
    dA2/dt' = (Q/V1)*A1 - (Q/V2)*A2
    Ad(0) = Dose,  A1(0) = A2(0) = 0

C1(t) = A1(t)/V1. For t < Tlag this evaluates at t' = 0, which returns the
initial condition A1 = 0 -- so pre-lag concentrations are correctly zero
without any extra masking (unlike decoder_tacrolimus.py's explicit mask,
here the lag is handled purely by clamping the ODE eval time).

NOTE on dose: confirmed by the user as a single fixed dose of 400 (mg) for
every subject (Main/quinidine.py's --dose, default 400.0) -- unlike
Tacrolimus (fixed D=300) and Paclitaxel (per-subject dose derived from
BSA), this is a constant across subjects, not per-subject-varying.
"""

import torch
import torchode as to


def _make_ode_term():
    def f(t, y, z):
        Ad = y[:, 0]
        A1 = y[:, 1]
        A2 = y[:, 2]

        ka = z[:, 0]
        Cl = z[:, 1]
        V1 = z[:, 2]
        Q  = z[:, 3]
        V2 = z[:, 4]

        dAd = -ka * Ad
        dA1 = ka * Ad - (Cl / V1 + Q / V1) * A1 + (Q / V2) * A2
        dA2 = (Q / V1) * A1 - (Q / V2) * A2

        return torch.stack([dAd, dA1, dA2], dim=-1)

    return to.ODETerm(f, with_args=True)


def Decoder_quinidine(z_normal, time, h, dose, atol=1e-7, rtol=1e-6, max_steps=2000):
    """
    z_normal : [nbatch, 6]  raw (pre-h-transform) latent PK parameters, order
                            [Tlag, ka, Cl, V1, Q, V2].
    time     : [nbatch, T]  observation times (hours).
    h        : callable     positivity transform (e.g. torch.exp).
    dose     : [nbatch]     dose tensor (constant 400mg for every subject --
                            see module docstring).
    max_steps : int         cap on adaptive-solver steps per subject (see
                            note below on why this matters here).

    Returns
    -------
    pred_x : [nbatch, T, 1]  predicted central-compartment concentration
             C1(t) = A1(t)/V1 at each observation time. Subjects where the
             ODE solve doesn't converge within max_steps get NaN predictions
             (propagates to a non-finite ELBO for that iteration, which the
             training loop already skips via `if torch.isfinite(elbo)`).

    Note on clamping and max_steps: unlike decoder_tacrolimus.py (closed-form,
    always stable) and decoder_paclitaxel.py (states are clamped but its
    tested priors keep rate constants bounded), this model divides by V1/V2
    to form Cl/V1, Q/V1, Q/V2 rate constants. Before training stabilizes, an
    untrained LSTM encoder can emit extreme raw z_normal values (e.g. a very
    negative pre-exp V1), which after h()=exp produces a near-zero V1 and
    hence enormous rate constants -- an arbitrarily stiff system that made
    torchode's adaptive step controller spiral to unbounded memory use
    instead of failing during this project's early burn-in testing (see
    Main/quinidine.py's module docstring / commit history). Clamping z to a
    physiologically generous range and capping solver steps turns that
    failure mode into a graceful NaN instead of an OOM/hang.
    """
    nbatch = dose.shape[0]
    z = h(z_normal)
    Tlag = z[:, 0].clamp(0.0, 5.0)
    ka   = z[:, 1].clamp(0.01, 50.0)
    Cl   = z[:, 2].clamp(0.01, 200.0)
    V1   = z[:, 3].clamp(0.5, 500.0)
    Q    = z[:, 4].clamp(0.01, 200.0)
    V2   = z[:, 5].clamp(0.5, 500.0)

    t_shifted = (time - Tlag.unsqueeze(1)).clamp(min=0.0)

    term = _make_ode_term()
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller, max_steps=max_steps)

    y0 = torch.zeros(nbatch, 3, device=z.device, dtype=z.dtype)
    y0[:, 0] = dose
    problem = to.InitialValueProblem(y0=y0, t_eval=t_shifted)
    z_ode = torch.stack([ka, Cl, V1, Q, V2], dim=1)
    sol = solver.solve(problem, args=z_ode)

    A1 = sol.ys[:, :, 1]  # [nbatch, T]
    C1 = A1 / V1.unsqueeze(1)

    # sol.status != 0 (per torchode convention) marks subjects that hit
    # max_steps without converging -- mark those predictions NaN rather than
    # silently returning a truncated/incorrect solution.
    if hasattr(sol, 'status'):
        failed = sol.status.to(torch.bool)
        if failed.any():
            C1 = C1.clone()
            C1[failed] = float('nan')

    return C1.unsqueeze(-1)
