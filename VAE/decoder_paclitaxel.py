"""
decoder_paclitaxel.py
-----------------------

NEW, additive decoder for the Paclitaxel case study (does NOT modify any
existing function in VAE/decoder.py).

Structural model
-----------------
Matches the actual generative simulator in
conditioning_limits_Paclitaxel/paclitaxel_popPK_joerger2006_v3.py
(Joerger et al. 2006, Clin Cancer Res 12:2150): a 3-compartment model with
saturable (Michaelis-Menten) elimination from the central compartment and
saturable MM distribution to the 1st peripheral compartment, plus linear
distribution to a 2nd peripheral compartment:

    State y = [A1, A2, A3]  (amounts, umol)
        A1: central compartment    A2: 1st peripheral   A3: 2nd peripheral
    C1 = A1/V1 ,  C3 = A3/V3
    EL    = VMEL * C1 / (KMEL + C1)        (saturable elimination, central)
    TR_12 = VMTR * C1 / (KMTR + C1)        (saturable transfer central->P1)
    TR_21 = K21  * A2                       (first-order P1 -> central)
    TR_13 = Q    * C1                       (linear central -> P2)
    TR_31 = Q    * C3                       (linear P2 -> central)

    dA1/dt = rate_in(t) - EL - TR_12 + TR_21 - TR_13 + TR_31
    dA2/dt = TR_12 - TR_21
    dA3/dt = TR_13 - TR_31

dosed as a constant-rate 3-hour IV infusion: rate_in = dose_umol / T_INF for
t in [0, T_INF], 0 afterwards (matches paclitaxel_popPK_joerger2006_v3.py's
`pk_odes` / `run_simulation`).

z_dim = 6 estimated latent PK parameters per subject: [V1, V3, VMEL, VMTR,
KMTR, Q] (mirrors the paper's free covariate-explained quantities; K21 and
KMEL are also subject to IIV in the simulator but, to keep z_dim tractable
for the VAE-nlme covariate-selection step, are held at fixed population
typical values -- consistent with the paper's smaller relative IIV/role
for K21 and the fact that KMEL is a Michaelis constant, not a clearance,
so it carries comparatively little identifiable inter-individual signal
from a single-dose 48h profile). This is a documented modelling choice, not
an architectural requirement, and easily extended to a larger z_dim if
later analyses want to free K21/KMEL too.

rate_in(t) is implemented as a smooth (but very steep) sigmoid gate rather
than a hard discontinuity, since torchode's adaptive-step solvers handle a
smooth gate far more robustly under autodiff than a true `if`; the gate
width is tiny relative to T_INF so the bias introduced is negligible
(steep transition over ~1e-2 h around t = T_INF, against a 48h horizon).
"""

#########################################################
# Import
#########################################################
import torch
import torchode as to

T_INF_DEFAULT = 3.0  # hours, 3-hour IV infusion (Joerger 2006)


def _rate_in(t, dose_umol, t_inf, gate_sharpness=200.0):
    """Smooth approximation of the constant-rate-infusion indicator:
    rate_in(t) = dose_umol/t_inf for t in [0, t_inf], else 0."""
    gate = torch.sigmoid(gate_sharpness * (t_inf - t))
    return (dose_umol / t_inf) * gate


def _make_ode_term(dose_umol, t_inf):
    """
    Returns an ODETerm closure over the per-subject dose / infusion duration.

    args passed at solve time: z = [V1, V3, VMEL, VMTR, KMTR, Q, K21, KMEL]
    (always 8 columns -- when K21/KMEL are fixed rather than estimated,
    Decoder_paclitaxel appends them as constant columns before calling the
    solver, so this closure doesn't need to know which mode is active).
    """

    def f(t, y, z):
        A1 = y[:, 0]
        A2 = y[:, 1]
        A3 = y[:, 2]

        V1 = z[:, 0]
        V3 = z[:, 1]
        VMEL = z[:, 2]
        VMTR = z[:, 3]
        KMTR = z[:, 4]
        Q = z[:, 5]
        K21 = z[:, 6]
        KMEL = z[:, 7]

        C1 = A1.clamp(min=0.0) / V1
        C3 = A3.clamp(min=0.0) / V3

        EL = VMEL * C1 / (KMEL + C1)
        TR_12 = VMTR * C1 / (KMTR + C1)
        TR_21 = K21 * A2.clamp(min=0.0)
        TR_13 = Q * C1
        TR_31 = Q * C3

        rate_in = _rate_in(t, dose_umol, t_inf)

        dA1 = rate_in - EL - TR_12 + TR_21 - TR_13 + TR_31
        dA2 = TR_12 - TR_21
        dA3 = TR_13 - TR_31

        return torch.stack([dA1, dA2, dA3], dim=-1)

    return to.ODETerm(f, with_args=True)


def Decoder_paclitaxel(z_normal, time, h, dose, K21=0.209, KMEL=0.047, t_inf=T_INF_DEFAULT,
                       estimate_k21_kmel=False, atol=1e-7, rtol=1e-6):
    """
    z_normal : [nbatch, 6 or 8]  raw (pre-h-transform) latent PK parameters,
                            order [V1, V3, VMEL, VMTR, KMTR, Q] (z_dim=6), or
                            [V1, V3, VMEL, VMTR, KMTR, Q, K21, KMEL] (z_dim=8)
                            when `estimate_k21_kmel=True`.
    time     : [nbatch, T]  observation times (hours), e.g.
                            [0,0.5,1,2,3,3.5,4,5,6,8,12,24,48].
    h        : callable     positivity transform (e.g. torch.exp).
    dose     : [nbatch]     per-subject dose in umol (175 mg/m^2 * BSA,
                            converted via MW 853.9 g/mol -- see
                            data_loading.load_two_file_wide's dose_fn usage
                            in Main/paclitaxel.py).
    K21, KMEL : float       fixed population-typical values for the
                            P1->central rate constant and the MM elimination
                            constant, used ONLY when `estimate_k21_kmel=False`
                            (the default -- see module docstring for the
                            rationale). Ignored when `estimate_k21_kmel=True`
                            (K21/KMEL are then read from z_normal instead).
    estimate_k21_kmel : bool
                            If True, K21 and KMEL become subject-specific
                            estimated latent parameters (z_dim=8) instead of
                            fixed population constants (z_dim=6, default) --
                            the full model, at the cost of 2 extra latent
                            dimensions for the VAE/covariate-selection step
                            to estimate per subject.
    t_inf    : float        infusion duration in hours (3.0 per Joerger 2006).

    Returns
    -------
    pred_x : [nbatch, T, 1]  predicted central-compartment concentration
             C1(t) = A1(t)/V1 at each observation time.
    """
    nbatch = dose.shape[0]
    z = h(z_normal)
    V1 = z[:, 0]

    if estimate_k21_kmel:
        z_ode = z  # K21, KMEL read from z[:, 6], z[:, 7] inside the ODE term
    else:
        K21_t = torch.full((nbatch,), float(K21), device=z.device, dtype=z.dtype)
        KMEL_t = torch.full((nbatch,), float(KMEL), device=z.device, dtype=z.dtype)
        z_ode = torch.cat([z, K21_t.unsqueeze(1), KMEL_t.unsqueeze(1)], dim=1)

    term = _make_ode_term(dose, t_inf)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)

    y0 = torch.zeros(nbatch, 3, device=z.device, dtype=z.dtype)
    problem = to.InitialValueProblem(y0=y0, t_eval=time)
    sol = solver.solve(problem, args=z_ode)

    A1 = sol.ys[:, :, 0]  # [nbatch, T]
    C1 = A1 / V1.unsqueeze(1)
    pred_x = C1.unsqueeze(-1)  # [nbatch, T, 1]

    return pred_x
