"""
decoder_warfarin.py
---------------------

NEW, additive decoder for the Warfarin PK case study (does NOT modify any
existing function in VAE/decoder.py).

Structural model
-----------------
1-compartment model with first-order absorption and an absorption lag time
(the classic O'Reilly 1968 warfarin design, PK-only -- PD/INR rows have been
excluded from the source data). Parameter set [Tlag, ka, Cl, V] (z_dim=4)
matches the PK-relevant subset of the parameters Ayral et al. 2021 (COSSAC
paper, Table 1) report for their joint "Warfarin PK/PD" model ("8 - Tlag,
ka, V, Cl, R0, kout, Imax, IC50"): this script only estimates the 4 PK
parameters, since the PD (R0, kout, Imax, IC50) rows are not used here.

    C(t) = Dose/V * ka/(ka - ke) * (exp(-ke*(t-Tlag)) - exp(-ka*(t-Tlag)))
           for t >= Tlag, else 0,   ke = Cl/V

Unlike decoder_tacrolimus.py (fixed population-constant ka and Tlag),
here BOTH ka and Tlag are per-subject estimated latent parameters, and dose
is a per-subject tensor computed on the fly as 1.5 * WT (see
functions_warfarin.load_warfarin) -- matches the real recorded doses in the
raw NONMEM file almost exactly (~60-153 mg across the 32 subjects) -- not a
single fixed scalar.
"""

#########################################################
# Import
#########################################################
import torch


def Decoder_warfarin(z_normal, time, h, dose):
    """
    z_normal : [nbatch, 4]  raw (pre-h-transform) latent PK parameters, order
                            [Tlag, ka, Cl, V].
    time     : [nbatch, T]  observation times (hours).
    h        : callable     positivity transform (e.g. torch.exp).
    dose     : [nbatch]     per-subject dose (real, varies by subject).

    Returns
    -------
    pred_x : [nbatch, T, 1]  predicted concentration at each observation time.
    """
    nbatch = z_normal.shape[0]
    z = h(z_normal)
    Tlag = z[:, 0]
    ka   = z[:, 1]
    Cl   = z[:, 2]
    V    = z[:, 3]
    ke   = Cl / V

    Tlag_b = Tlag.view(nbatch, 1)
    ka_b   = ka.view(nbatch, 1)
    ke_b   = ke.view(nbatch, 1)
    V_b    = V.view(nbatch, 1)
    dose_b = dose.view(nbatch, 1)

    delta_t = time - Tlag_b
    mask    = (delta_t >= 0).to(time.dtype)
    delta_t = delta_t * mask

    # Bateman function; switch to L'Hopital limit when |ka - ke| < min_gap
    # to avoid dividing by a near-zero denominator (mirrors decoder_tacrolimus.py).
    min_gap = 0.02
    diff    = ka_b - ke_b
    regular = diff.abs() >= min_gap

    contrib_reg = (dose_b * ka_b / (V_b * diff)
                   * (torch.exp(-ke_b * delta_t) - torch.exp(-ka_b * delta_t)))
    contrib_lim = dose_b * ka_b / V_b * delta_t * torch.exp(-ka_b * delta_t)

    contrib = torch.where(regular, contrib_reg, contrib_lim) * mask

    return contrib.unsqueeze(-1)  # [nbatch, T, 1]
