"""
Created on 16.07.2025

@author: Jan Rohleff

Functions for visualization of the convergence of the VAE-nlme model.
"""

import math
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Generic convergence plots — work for any dataset / z_dim / n_cov
# ---------------------------------------------------------------------------

def _to_np(t):
    return t.detach().numpy() if hasattr(t, 'detach') else np.asarray(t)


def _add_iter_marks(ax, iters_burn_in, k_alpha, k_beta, total_iters,
                    kl_iter, gamma_iter, iters, fontsize=7):
    """Shade burn-in and add K_alpha / K_beta lines. Returns (l1, l2, l3)."""
    l1 = ax.axvspan(0, iters_burn_in, facecolor=(0.83, 0.83, 0.83, 0.5), alpha=0.4)
    l2 = ax.axvline(x=k_alpha, linestyle='dashed', color='green')
    l3 = ax.axvline(x=k_beta,  linestyle='dashed', color='red')
    xticks  = [iters_burn_in, k_alpha, k_beta, total_iters]
    xlabels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ymin, ymax = ax.get_ylim()
    dy = (ymax - ymin) or 1.0
    for x, lbl in zip(xticks, xlabels):
        ax.text(x, ymin - dy / 14, lbl, ha='center', va='top', fontsize=fontsize)
    return l1, l2, l3


def _save_or_show(save_path):
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.9)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plotConvergence_pop(elbo_iter, a_iter, z_pop_iter, omega_pop_iter,
                        iters, kl_iter, gamma_iter, iters_burn_in,
                        param_names, save_path=None):
    """Structural PK parameter convergence for any dataset / z_dim.

    param_names : list of z_dim strings, e.g. ['ke', 'V'] or ['ka', 'ke', 'V'].
    Panels (left→right, top→bottom): z_pop structural (h-space), ω, residual a, ELBO.
    save_path   : if given the figure is saved (PDF/PNG) and closed; otherwise plt.show().
    """
    elbo_np  = _to_np(elbo_iter)
    elbo_np  = np.hstack([np.nan * np.zeros(iters_burn_in), elbo_np])
    a_np     = _to_np(a_iter)
    zpop_np  = _to_np(z_pop_iter)
    omega_np = _to_np(omega_pop_iter)

    z_dim = len(param_names)
    panels = (
        [(zpop_np[:, i],  rf'${param_names[i]}_{{pop}}$')      for i in range(z_dim)] +
        [(omega_np[:, i], rf'$\omega_{{{param_names[i]}}}$')   for i in range(z_dim)] +
        [(a_np,           r'$a$'),
         (elbo_np,        r'$-\mathcal{L}_\psi(x)$')]
    )

    ncols = 3
    nrows = math.ceil(len(panels) / ncols)
    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=False,
                            figsize=(4 * ncols, 3 * nrows))
    axs_flat = np.array(axs).flatten()

    total_iters = iters + iters_burn_in
    k_alpha     = kl_iter    + iters_burn_in
    k_beta      = gamma_iter + iters_burn_in

    l1 = l2 = l3 = None
    for i, (data, title) in enumerate(panels):
        ax = axs_flat[i]
        ax.plot(data)
        ax.set_title(title, size=12)
        ax.set_xlim(0, total_iters)
        l1, l2, l3 = _add_iter_marks(ax, iters_burn_in, k_alpha, k_beta,
                                      total_iters, kl_iter, gamma_iter, iters)

    for j in range(len(panels), len(axs_flat)):
        axs_flat[j].axis('off')

    legend_ax = axs_flat[len(panels)] if len(panels) < len(axs_flat) else axs_flat[-1]
    legend_ax.axis('off')
    if l1 and l2 and l3:
        legend_ax.legend([l1, l2, l3], ['Burn in', r'$K_\alpha$', r'$K_\beta$'],
                         loc='center', fontsize='large')

    _save_or_show(save_path)


def plotConvergence_covariate(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
                              z_dim, n_cov, param_names, cov_names,
                              max_cov=8, save_path=None):
    """Beta-coefficient convergence traces in a (z_dim × min(n_cov, max_cov)) grid.

    Each subplot shows the post-burn-in trace of beta[param_p, cov_k].
    Panels where the final beta is zero (set to 0 by MIQP) are shaded grey.
    If n_cov > max_cov only the first max_cov covariates are shown.

    z_pop_iter layout (columns):
      0 … z_dim-1          : h(z_pop structural params) — not used here
      z_dim + p*n_cov + k  : beta for parameter p, covariate k

    save_path : if given the figure is saved (PDF/PNG) and closed; otherwise plt.show().
    """
    zpop_np = _to_np(z_pop_iter)

    n_show  = min(n_cov, max_cov)
    nrows   = z_dim
    ncols   = n_show

    if ncols == 0 or nrows == 0:
        return

    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=False,
                            figsize=(max(6, 2.2 * ncols), 2.2 * nrows))
    # normalise to 2-D array
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs[np.newaxis, :]
    elif ncols == 1:
        axs = axs[:, np.newaxis]

    post_bi  = zpop_np[iters_burn_in:]   # shape [iters, z_dim + M]
    x_range  = iters
    k_alpha  = kl_iter
    k_beta   = gamma_iter
    xticks   = [0, kl_iter, gamma_iter, iters]
    xlabels  = ['0', str(kl_iter), str(gamma_iter), str(iters)]

    for p, pname in enumerate(param_names):
        for k in range(n_show):
            col_idx = z_dim + p * n_cov + k
            data    = post_bi[:, col_idx]
            cname   = cov_names[k] if k < len(cov_names) else str(k)
            ax      = axs[p, k]

            ax.plot(data, linewidth=0.8)
            ax.set_title(rf'$\beta_{{{pname}}}^{{{cname}}}$', size=9)
            ax.set_xlim(0, x_range)

            if data[-1] == 0:
                ymin, ymax = ax.get_ylim()
                ax.axhspan(ymin, ymax, facecolor=(0.83, 0.83, 0.83), alpha=0.4, zorder=0)
                ax.set_ylim(ymin, ymax)

            ax.axvline(x=k_alpha, linestyle='dashed', color='green', linewidth=0.7)
            ax.axvline(x=k_beta,  linestyle='dashed', color='red',   linewidth=0.7)

            ax.set_xticks(xticks)
            ax.set_xticklabels([])
            if p == nrows - 1:          # bottom row: add tick labels
                ymin, ymax = ax.get_ylim()
                dy = (ymax - ymin) or 1.0
                for x, lbl in zip(xticks, xlabels):
                    ax.text(x, ymin - dy / 12, lbl, ha='center', va='top', fontsize=6)

        axs[p, 0].set_ylabel(pname, fontsize=9)

    if n_cov > max_cov:
        fig.suptitle(f'Beta traces — first {max_cov} of {n_cov} covariates shown',
                     fontsize=10, y=1.01)

    _save_or_show(save_path)


def plotConvergence_beta(z_pop, z_dim, n_cov, param_names, cov_names,
                         save_path=None):
    """Heatmap of the final beta matrix (z_dim rows × n_cov cols).

    Selected (nonzero) cells are coloured RdBu_r; zeroed-out cells are grey.
    Figure width scales with n_cov (capped at 24 inches).

    save_path : if given the figure is saved (PDF/PNG) and closed; otherwise plt.show().
    """
    beta_np = _to_np(z_pop[z_dim:]) if n_cov > 0 else np.zeros((z_dim, 0))
    beta    = beta_np.reshape(z_dim, n_cov) if n_cov > 0 else beta_np

    fig_w = max(6, min(n_cov * 0.4 + 1.5, 28))
    fig_h = max(2, z_dim * 0.7 + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    selected = np.abs(beta) > 1e-9
    beta_col = np.where(selected, beta, np.nan)
    vmax = np.nanmax(np.abs(beta_col)) if np.any(selected) else 1.0

    im = ax.imshow(beta_col, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.imshow(~selected, aspect='auto', cmap='Greys', vmin=0, vmax=1, alpha=0.35)

    ax.set_yticks(range(z_dim))
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_xticks(range(n_cov))
    ax.set_xticklabels(cov_names if cov_names else [str(k) for k in range(n_cov)],
                       rotation=90, fontsize=7)
    ax.set_xlabel('Covariate')
    ax.set_ylabel('PK parameter')
    n_sel = int(selected.any(axis=0).sum())
    ax.set_title(f'Beta matrix — {n_sel}/{n_cov} covariates selected  '
                 f'(grey = zeroed by MIQP)')
    plt.colorbar(im, ax=ax, label='beta value', shrink=0.8)

    _save_or_show(save_path)

def printOutput_theo(z_pop, omega_pop, a, b, z_dim, nbatch, n_tot, h, names, LL_lin, LL_is):
    ln_N = torch.log(torch.tensor(nbatch))
    ln_n_tot = torch.log(n_tot)
    z_pop_h = h(z_pop)
    print('')
    print('#############################################')
    print('ESTIMATION OF THE POPULATION PARAMETERS')
    print('#############################################')
    print('')
    print('Fixed Effects:')
    print(f'{"ka_pop:":<15} {z_pop_h[0]:>10.2f}')
    print(f'{"ke_pop:":<15} {z_pop_h[1]:>10.2f}')
    print(f'{"V_pop:":<15} {z_pop_h[2]:>10.2f}')
    count = 0
    for k in range(z_dim, len(z_pop)):
        if z_pop[k] != 0:
            print(f'{names[k-z_dim] + ":":<15} {z_pop[k]:>10.2f}')
            count += 1
    print('')
    print('Standard Deviation of the Random Effects:')
    print(f'{"omega_ka:":<15} {omega_pop[0].sqrt():>10.2f}')
    print(f'{"omega_ke:":<15} {omega_pop[1].sqrt():>10.2f}')
    print(f'{"omega_V:":<15} {omega_pop[2].sqrt():>10.2f}')

    print('')
    print('Error Model Parameters:')
    if a != 0:
        print(f'{"a:":<15} {a:>10.2f}')
    if b != 0:
        print(f'{"b:":<15} {b:>10.2f}')

    print('')
    print('#########################################################')
    print('ESTIMATION OF THE LOG LIKELIHOOD and INFORMATION CRITERIA')
    print('#########################################################')

    print(f'{"":<45} {"Linearization:":>30}  {"Importance Sampling:":>30}')
    print('-' * 135)

    # OFV
    print(f'{"-2Log likelihood (OFV):":<45}'f'{2*LL_lin:>30.2f}{2*LL_is:>30.2f}')

    # AIC
    print(f'{"Akaike Information Criteria (AIC):":<45}'
      f'{2*LL_lin + 2*(2*z_dim + 1 + count):>30.2f}'
      f'{2*LL_is + 2*(2*z_dim + 1 + count):>30.2f}')

    # BIC
    print(f'{"Bayesian Information Criteria (BIC):":<45}'
      f'{2*LL_lin + ln_N*(2*z_dim + 1 + count):>30.2f}'
      f'{2*LL_is + ln_N*(2*z_dim + 1 + count):>30.2f}')

    # BICc
    print(f'{"Corrected B.I. Criteria (BICc):":<45}'
      f'{2*LL_lin + ln_N*(z_dim + count) + ln_n_tot*(z_dim + 1):>30.2f}'
      f'{2*LL_is + ln_N*(z_dim + count) + ln_n_tot*(z_dim + 1):>30.2f}')


def plotConvergence_pop_theo(elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
                             save_path='../Plots/theophylline_convergence_popParam.pdf'):
    """Thin wrapper around plotConvergence_pop for the theophylline case study."""
    return plotConvergence_pop(elbo_iter, a_iter, z_pop_iter, omega_pop_iter,
                               iters, kl_iter, gamma_iter, iters_burn_in,
                               param_names=['ka', 'ke', 'V'], save_path=save_path)


def plotConvergence_covariate_theo(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
                                   save_path='../Plots/theophylline_convergence_covariate.pdf'):
    """Thin wrapper around plotConvergence_covariate for the theophylline base case (2 covariates)."""
    return plotConvergence_covariate(
        z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in,
        z_dim=3, n_cov=2, param_names=['ka', 'ke', 'V'],
        cov_names=['weight', 'sex'], max_cov=8, save_path=save_path)

def plotConvergence_pop_theo_multiple(elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in): 
    elbo_iter = elbo_iter.detach().numpy()
    elbo_iter = np.hstack([np.nan*np.zeros(100), elbo_iter])
    a_iter = a_iter.detach().numpy()
    z_pop_iter = z_pop_iter.detach().numpy()
    omega_pop_iter = omega_pop_iter.detach().numpy()

    fig, axs = plt.subplots(3, 3, sharex=False, sharey=False)
    fig.set_size_inches(10, 6)
    xmax = iters
    burn_in = iters_burn_in
    k_alpha = kl_iter + iters_burn_in
    k_beta = gamma_iter + iters_burn_in
    xticks = [iters_burn_in, kl_iter + iters_burn_in, gamma_iter+ iters_burn_in, iters+ iters_burn_in]
    xtick_labels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    plot_data = [
    (z_pop_iter[:, 0], r'$k_{a,pop}$'),
    (z_pop_iter[:, 1], r'$k_{e,pop}$'),
    (z_pop_iter[:, 2], r'$V_{pop}$'),
    (omega_pop_iter[:, 0], r'$\omega_{k_a}$'),
    (omega_pop_iter[:, 1], r'$\omega_{k_e}$'),
    (omega_pop_iter[:, 2], r'$\omega_{V}$'),
    (a_iter, r'$a$'),
    (elbo_iter, r'$-\mathcal{L}_\psi(x)$')
    ]

    for i, (data, title) in enumerate(plot_data):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        if data is not None:
            ax.plot(data)
            ax.set_title(title, size = 14)
            ax.set_xlim(0, xmax)

            l1 = ax.axvspan(0, burn_in, facecolor=(0.83, 0.83, 0.83, 0.5), alpha=0.4)
            l2 = ax.axvline(x=k_alpha, linestyle='dashed', color='green')
            l3 = ax.axvline(x=k_beta, linestyle='dashed', color='red')

            ax.set_xticks(xticks)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            offset_text = (ymax - ymin) / 16
            offset_label = (ymax - ymin) / 5
            for x, label in zip(xticks, xtick_labels):
                ax.text(x, ymin - offset_text, label, ha='center', va='top')
            ax.text(210, ymin - 1.2 * offset_label, 'Iterations', ha='center', va='top', fontsize=11)


    plt.tight_layout()
    axs[2,2].axis('off')
    axs[2, 2].legend([l1, l2, l3],['Burn in',r'$K_\alpha$',r'$K_\beta$'], loc='best', fontsize = 'x-large')

    plt.subplots_adjust(hspace = 1)
    plt.savefig('../Plots/theophylline_multiple_convergence_popParam.pdf', dpi=500)
    plt.show()

def plotConvergence_covariate_theo_multiple(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in): 
    z_pop_iter = z_pop_iter.detach().numpy()

    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False)
    fig.set_size_inches(10, 4)
    xmax = iters
    k_alpha = kl_iter 
    k_beta = gamma_iter 
    xticks = [0, kl_iter, gamma_iter, iters]
    xtick_labels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    plot_data = [
    (z_pop_iter[iters_burn_in:, 3], r'$\beta_{k_a}^{w}$'),
    (z_pop_iter[iters_burn_in:, 5], r'$\beta_{k_e}^{w}$'),
    (z_pop_iter[iters_burn_in:, 7], r'$\beta_{V}^{w}$'),
    (z_pop_iter[iters_burn_in:, 4], r'$\beta_{k_a}^{sex}$'),
    (z_pop_iter[iters_burn_in:, 6], r'$\beta_{k_e}^{sex}$'),
    (z_pop_iter[iters_burn_in:, 8], r'$\beta_{V}^{sex}$')
    ]

    for i, (data, title) in enumerate(plot_data):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        if data is not None:
            ax.plot(data)
            ax.set_title(title, size = 14)
            ax.set_xlim(0, xmax)
            if data[-1] == 0:
                ymin, ymax = ax.get_ylim()
                ax.axhspan(ymin, ymax, facecolor=(0.83, 0.83, 0.83), alpha=0.4, zorder=0)
                ax.set_ylim(ymin, ymax)

            # vertikale Linien überall
            ax.axvline(x=k_alpha, linestyle='dashed', color='green')
            ax.axvline(x=k_beta, linestyle='dashed', color='red')

            ax.set_xticks(xticks)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            offset_text = (ymax - ymin) / 16
            offset_label = (ymax - ymin) / 5
            for x, label in zip(xticks, xtick_labels):
                ax.text(x, ymin - offset_text, label, ha='center', va='top')
            ax.text(160, ymin - 1.2 * offset_label, 'Iterations', ha='center', va='top', fontsize=11)


    plt.tight_layout()

    plt.subplots_adjust(hspace = 1)
    plt.savefig('../Plots/theophylline_multiple_convergence_covariate.pdf', dpi=1000)
    plt.show()

def printOutput_neonates(z_pop, omega_pop, a, b, z_dim, nbatch, n_tot, h, names, LL_lin_mu, LL_is):
    ln_N = torch.log(torch.tensor(nbatch))
    ln_n_tot = torch.log(n_tot)
    z_pop_h = h(z_pop)
    print('')
    print('#############################################')
    print('ESTIMATION OF THE POPULATION PARAMETERS')
    print('#############################################')
    print('')
    print('Fixed Effects:')
    print(f'{"W0_pop:":<15} {z_pop_h[0]:>10.2f}')
    print(f'{"kin_pop:":<15} {z_pop_h[1]:>10.2f}')
    print(f'{"Tlag_pop:":<15} {z_pop_h[2]:>10.2f}')
    print(f'{"kout_pop:":<15} {z_pop_h[3]:>10.2f}')
    print(f'{"T50_pop:":<15} {z_pop_h[4]:>10.2f}')
    count = 0
    for k in range(z_dim, len(z_pop)):
        if z_pop[k] != 0:
            print(f'{names[k-z_dim] + ":":<15} {z_pop[k]:>10.2f}')
            count += 1
    print('')
    print('Standard Deviation of the Random Effects:')
    print(f'{"omega_W0:":<15} {omega_pop[0].sqrt():>10.2f}')
    print(f'{"omega_kin:":<15} {omega_pop[1].sqrt():>10.2f}')
    print(f'{"omega_Tlag:":<15} {omega_pop[2].sqrt():>10.2f}')
    print(f'{"omega_kout:":<15} {omega_pop[3].sqrt():>10.2f}')
    print(f'{"omega_T50:":<15} {omega_pop[4].sqrt():>10.2f}')

    print('')
    print('Error Model Parameters:')
    if a != 0:
        print(f'{"a:":<15} {a:>10.2f}')
    if b != 0:
        print(f'{"b:":<15} {b:>10.2f}')

    print('')
    print('#########################################################')
    print('ESTIMATION OF THE LOG LIKELIHOOD and INFORMATION CRITERIA')
    print('#########################################################')

    print(f'{"":<45} {"Linearization:":>30}{"Importance Sampling:":>30}')
    print('-' * 135)

    # OFV
    print(f'{"-2Log likelihood (OFV):":<45}'f'{2*LL_lin_mu:>30.2f}{2*LL_is:>30.2f}')

    # AIC
    print(f'{"Akaike Information Criteria (AIC):":<45}'
      f'{2*LL_lin_mu + 2*(2*z_dim + 1 + count):>30.2f}'
      f'{2*LL_is + 2*(2*z_dim + 1 + count):>30.2f}')

    # BIC
    print(f'{"Bayesian Information Criteria (BIC):":<45}'
      f'{2*LL_lin_mu + ln_N*(2*z_dim + 1 + count):>30.2f}'
      f'{2*LL_is + ln_N*(2*z_dim + 1 + count):>30.2f}')

    # BICc
    print(f'{"Corrected B.I. Criteria (BICc):":<45}'
      f'{2*LL_lin_mu + ln_N*(z_dim + count) + ln_n_tot*(z_dim + 1):>30.2f}'
      f'{2*LL_is + ln_N*(z_dim + count) + ln_n_tot*(z_dim + 1):>30.2f}')
    
def plotConvergence_pop_neonates(elbo_iter, a_iter, z_pop_iter, omega_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in): 
    elbo_iter = elbo_iter.detach().numpy()
    elbo_iter = np.hstack([np.nan*np.zeros(100), elbo_iter])
    a_iter = a_iter.detach().numpy()
    z_pop_iter = z_pop_iter.detach().numpy()
    omega_pop_iter = omega_pop_iter.detach().numpy()

    fig, axs = plt.subplots(3, 4, sharex=False, sharey=False)
    fig.set_size_inches(10, 6)
    xmax = iters
    burn_in = iters_burn_in
    k_alpha = kl_iter + iters_burn_in
    k_beta = gamma_iter + iters_burn_in
    xticks = [iters_burn_in, kl_iter + iters_burn_in, gamma_iter+ iters_burn_in, iters+ iters_burn_in]
    xtick_labels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    plot_data = [
    (z_pop_iter[:, 0], r'$W_{0,pop}$'),
    (z_pop_iter[:, 1], r'$k_{in,pop}$'),
    (z_pop_iter[:, 2], r'$T_{lag,pop}$'),
    (z_pop_iter[:, 3], r'$k_{out,pop}$'),
    (z_pop_iter[:, 4], r'$T_{50,pop}$'),
    (omega_pop_iter[:, 0], r'$\omega_{W_0}$'),
    (omega_pop_iter[:, 1], r'$\omega_{k_{in}}$'),
    (omega_pop_iter[:, 2], r'$\omega_{T_{lag}}$'),
    (omega_pop_iter[:, 3], r'$\omega_{k_{out}}$'),
    (omega_pop_iter[:, 4], r'$\omega_{T_{50}}$'),
    (a_iter, r'$a$'),
    (elbo_iter, r'$-\mathcal{L}_\psi(x)$')
    ]

    for i, (data, title) in enumerate(plot_data):
        row, col = divmod(i, 4)
        ax = axs[row, col]

        if data is not None:
            ax.plot(data)
            ax.set_title(title, size = 14)
            ax.set_xlim(0, xmax)

            l1 = ax.axvspan(0, burn_in, facecolor=(0.83, 0.83, 0.83, 0.5), alpha=0.4)
            l2 = ax.axvline(x=k_alpha, linestyle='dashed', color='green')
            l3 = ax.axvline(x=k_beta, linestyle='dashed', color='red')

            ax.set_xticks(xticks)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            offset_text = (ymax - ymin) / 16
            offset_label = (ymax - ymin) / 5
            for x, label in zip(xticks, xtick_labels):
                ax.text(x, ymin - offset_text, label, ha='center', va='top')
            ax.text(210, ymin - 1.2 * offset_label, 'Iterations', ha='center', va='top', fontsize=11)


    plt.tight_layout()
    #axs[2,2].axis('off')
    axs[2, 3].legend([l1, l2, l3],['Burn in',r'$K_\alpha$',r'$K_\beta$'], loc='best', fontsize = '12')

    plt.subplots_adjust(hspace = 1)
    plt.savefig('Plots/neonates_convergence_popParam.pdf', dpi=500)
    plt.show()

def plotConvergence_covariate_neonates(z_pop_iter, iters, kl_iter, gamma_iter, iters_burn_in): 
    z_pop_iter = z_pop_iter.detach().numpy()

    fig, axs = plt.subplots(5, 5, sharex=False, sharey=False)
    fig.set_size_inches(18, 8)
    xmax = iters
    k_alpha = kl_iter 
    k_beta = gamma_iter 
    xticks = [0, kl_iter, gamma_iter, iters]
    xtick_labels = ['0', str(kl_iter), str(gamma_iter), str(iters)]
    plot_data = [
    (z_pop_iter[iters_burn_in:, 5], r'$\beta_{W_0,sex}$'),
    (z_pop_iter[iters_burn_in:, 6], r'$\beta_{W_0,DelM}$'),
    (z_pop_iter[iters_burn_in:, 7], r'$\beta_{W_0,GA}$'),
    (z_pop_iter[iters_burn_in:, 8], r'$\beta_{W_0,Mage}$'),
    (z_pop_iter[iters_burn_in:, 9], r'$\beta_{W_0,Para_2}$'),
    (z_pop_iter[iters_burn_in:, 10], r'$\beta_{k_{in},sex}$'),
    (z_pop_iter[iters_burn_in:, 11], r'$\beta_{k_{in},DelM}$'),
    (z_pop_iter[iters_burn_in:, 12], r'$\beta_{k_{in},GA}$'),
    (z_pop_iter[iters_burn_in:, 13], r'$\beta_{k_{in},Mage}$'),
    (z_pop_iter[iters_burn_in:, 14], r'$\beta_{k_{in},Para_2}$'),
    (z_pop_iter[iters_burn_in:, 15], r'$\beta_{T_{lag}, sex}$'),
    (z_pop_iter[iters_burn_in:, 16], r'$\beta_{T_{lag}, DelM}$'),
    (z_pop_iter[iters_burn_in:, 17], r'$\beta_{T_{lag}, GA}$'),
    (z_pop_iter[iters_burn_in:, 18], r'$\beta_{T_{lag}, Mage}$'),
    (z_pop_iter[iters_burn_in:, 19], r'$\beta_{T_{lag}, Para_2}$'),
    (z_pop_iter[iters_burn_in:, 20], r'$\beta_{k_{out}, sex}$'),
    (z_pop_iter[iters_burn_in:, 21], r'$\beta_{k_{out}, DelM}$'),
    (z_pop_iter[iters_burn_in:, 22], r'$\beta_{k_{out}, GA}$'),
    (z_pop_iter[iters_burn_in:, 23], r'$\beta_{k_{out}, Mage}$'),
    (z_pop_iter[iters_burn_in:, 24], r'$\beta_{k_{out}, Para_2}$'),
    (z_pop_iter[iters_burn_in:, 25], r'$\beta_{T_{50}, sex}$'),
    (z_pop_iter[iters_burn_in:, 26], r'$\beta_{T_{50}, DelM}$'),
    (z_pop_iter[iters_burn_in:, 27], r'$\beta_{T_{50}, GA}$'),
    (z_pop_iter[iters_burn_in:, 28], r'$\beta_{T_{50}, Mage}$'),
    (z_pop_iter[iters_burn_in:, 29], r'$\beta_{T_{50}, Para_2}$'),

    ]

    for i, (data, title) in enumerate(plot_data):
        row, col = divmod(i, 5)
        ax = axs[row, col]

        if data is not None:
            ax.plot(data)
            ax.set_title(title, size = 14)
            ax.set_xlim(0, xmax)
            if data[-1] == 0:
                ymin, ymax = ax.get_ylim()
                ax.axhspan(ymin, ymax, facecolor=(0.83, 0.83, 0.83), alpha=0.4, zorder=0)
                ax.set_ylim(ymin, ymax)

            # vertikale Linien überall
            ax.axvline(x=k_alpha, linestyle='dashed', color='green')
            ax.axvline(x=k_beta, linestyle='dashed', color='red')

            ax.set_xticks(xticks)
            ax.set_xticklabels([])

            ymin, ymax = ax.get_ylim()
            offset_text = (ymax - ymin) / 16
            offset_label = (ymax - ymin) / 5
            for x, label in zip(xticks, xtick_labels):
                ax.text(x, ymin - offset_text, label, ha='center', va='top')
            ax.text(160, ymin - 1.2 * offset_label, 'Iterations', ha='center', va='top', fontsize=11)


    plt.tight_layout()

    plt.subplots_adjust(hspace = 1)
    plt.savefig('Plots/neonates_convergence_covariate.pdf', dpi=1000)
    plt.show()
