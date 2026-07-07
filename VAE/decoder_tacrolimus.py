"""
decoder_tacrolimus.py
----------------------

NEW, additive decoder for the Tacrolimus case study (does NOT modify any
existing function in VAE/decoder.py).

Structural model (ground truth confirmed by the user, overrides the earlier
q12h-superposition guess inferred from Tacrolimus_parameters.csv -- that file
turned out not to be the generator behind Tacrolimus_data.csv):

    C(t) = D/V * ka/(ka - ke) * (exp(-ke*(t - t0)) - exp(-ka*(t - t0)))   for t >= t0
    C(t) = 0                                                              for t <  t0

with a SINGLE dose D = 300 (fixed, same for every subject), a FIXED
absorption rate ka = 0.502 1/h (not estimated, not per-subject), and a FIXED
absorption lag time t0 = 0.346 h (not estimated). Only ke and V are
subject-specific latent PK parameters (z_dim = 2: [ke, V]).
"""

#########################################################
# Import
#########################################################
import torch

#########################################################
# Fixed (population-constant) structural parameters
#########################################################
DOSE_DEFAULT = 300.0
KA_DEFAULT = 0.502
LAG_DEFAULT = 0.346


#########################################################
# Decoder
#########################################################
def Decoder_tacrolimus(z_normal, time, h, dose=DOSE_DEFAULT, ka=KA_DEFAULT, t0=LAG_DEFAULT):
    """
    z_normal : [nbatch, 2]  raw (pre-h-transform) latent PK parameters, order
                            [ke, V].
    time     : [nbatch, T]  observation times (hours).
    h        : callable     positivity transform (e.g. torch.exp).
    dose     : float        single dose amount, fixed at 300 for all subjects
                            (ground truth; kept as a default arg, not a
                            per-subject tensor, since it does not vary).
    ka       : float        fixed (not estimated) absorption rate, 0.502 1/h.
    t0       : float        fixed (not estimated) absorption lag time, 0.346 h.

    Returns
    -------
    pred_x : [nbatch, T, 1]  predicted concentration at each observation time:
             C(t) = dose/V * ka/(ka-ke) * (exp(-ke*(t-t0)) - exp(-ka*(t-t0)))
             for t >= t0, else 0.
    """
    nbatch = z_normal.shape[0]
    z  = h(z_normal)
    ke = z[:, 0]
    V  = z[:, 1]

    delta_t = time - t0          # [nbatch, T]
    mask    = (delta_t >= 0).to(time.dtype)
    delta_t = delta_t * mask     # zero out pre-lag times

    ke_b = ke.view(nbatch, 1)
    V_b  = V.view(nbatch, 1)

    # Bateman function; switch to L'Hôpital limit when |ka - ke| < min_gap
    # to avoid dividing by a near-zero denominator.
    # Previous guard (shift ke by +eps) was wrong: it made the denominator
    # -eps (still near-zero, same sign issue) rather than a safe value.
    min_gap = 0.02               # ~4% of ka; well outside float noise
    diff    = ka - ke_b          # [nbatch, 1]
    regular = diff.abs() >= min_gap

    contrib_reg = (dose * ka / (V_b * diff)
                   * (torch.exp(-ke_b * delta_t) - torch.exp(-ka * delta_t)))
    contrib_lim = dose * ka / V_b * delta_t * torch.exp(-ka * delta_t)

    contrib = torch.where(regular, contrib_reg, contrib_lim) * mask

    return contrib.unsqueeze(-1)  # [nbatch, T, 1]
