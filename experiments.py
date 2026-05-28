#!/usr/bin/env python3
"""
Experiments for the multi-worker ICML paper.

This file generates figures for the paper on tight convergence
analysis of error feedback algorithms in distributed optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from lyapunov import has_lyapunov, bisection
from pepit_helpers import worst_case_performance
from plotting import contour_plot, line_plot, standard_textbox, set_matplotlib_style
from utils import dask_grid_compute, dask_parallel_map
from theory_helpers import optimal_step_size

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
LABEL_SIZE = 20
TICK_SIZE = 20
L_FIXED = 1.0
KAPPAS = [5, 10, 100]
GRID_RESOLUTION = 100  # Coarser grid for faster computation
EPSILON_VALS = np.linspace(0.01, 0.99, GRID_RESOLUTION)
TOL = 1e-4  # Bisection tolerance
ROOT_IMAG_TOL = 1e-5  # Treat tiny imaginary parts as numerical noise

Path("figures").mkdir(exist_ok=True)


# ------------------------------------------------------------------------------
# Polynomial Rate Formula (Empirical Law 4.3, n=2)
# ------------------------------------------------------------------------------

def polynomial_coeffs_n2(epsilon, L1, mu1, L2, mu2):
    """
    Return coefficients of the cubic Q(ρ) for the n=2 heterogeneous empirical law.
    """
    s = np.sqrt(epsilon)
    r_s = (1 - s)**2 / (1 + s)

    Sigma1, Sigma2 = L1 + mu1, L2 + mu2
    Delta1, Delta2 = L1 - mu1, L2 - mu2

    K1 = (Delta2**2 * Sigma1 + Delta1**2 * Sigma2) / (Sigma1 * Sigma2 * (Sigma1 + Sigma2))
    K2 = (Delta1 + Delta2)**2 / (Sigma1 + Sigma2)**2

    # Coefficients for ρ³ - B*ρ² + C*ρ - D = 0
    B = s * (2 + s) + r_s * (s * K1 + K2)
    C = s**2 * (1 + 2*s + r_s * (K1 + s * K2))
    D = s**4

    return [1.0, -B, C, -D]


def polynomial_roots_n2(epsilon, L1, mu1, L2, mu2):
    """
    Compute all roots of Q(ρ) for the n=2 heterogeneous empirical law.
    """
    coeffs = polynomial_coeffs_n2(epsilon, L1, mu1, L2, mu2)
    return np.roots(coeffs)


def polynomial_rate_n2(epsilon, L1, mu1, L2, mu2):
    """
    Compute the optimal rate for n=2 heterogeneous case from the polynomial Q(ρ).

    The polynomial is given by:
        Q(ρ) = ρ³ - [s(2+s) + r(s)(sK₁ + K₂)]ρ² + s²[1+2s + r(s)(K₁ + sK₂)]ρ - s⁴

    where s = √ε, r(s) = (1-s)²/(1+s), and K₁, K₂ depend on L and μ parameters.

    Returns the largest real root.
    """
    roots = polynomial_roots_n2(epsilon, L1, mu1, L2, mu2)
    real_mask = np.abs(roots.imag) <= ROOT_IMAG_TOL
    real_roots = roots.real[real_mask]

    if len(real_roots) == 0:
        return np.nan
    return np.max(real_roots)


def homogeneous_optimal_rate(epsilon, L, mu):
    """
    Compute the optimal rate for homogeneous case (same L, μ for all workers).
    
    This is the closed-form rate from Theorem 1:
        ρ* = √ε + ((1-√ε)/2) * ((κ-1)/(κ+1))² * Ψ(κ,ε)
    
    where Ψ(κ,ε) = 1 - √ε + √[(1+√ε)² + √ε * 16κ/(κ-1)²]
    """
    sqrt_eps = np.sqrt(epsilon)
    kappa = L / mu
    
    if kappa == 1:  # Edge case: no curvature gap
        return sqrt_eps
    
    kappa_term = ((kappa - 1) / (kappa + 1))**2
    Psi = 1 - sqrt_eps + np.sqrt((1 + sqrt_eps)**2 + sqrt_eps * 16 * kappa / (kappa - 1)**2)
    
    return sqrt_eps + ((1 - sqrt_eps) / 2) * kappa_term * Psi


def optimal_step_size_heterogeneous(Ls, mus, epsilon):
    """
    Compute the optimal step size for heterogeneous workers.
    
    η* = (2n / Σᵢ(Lᵢ + μᵢ)) * (1 - √ε) / (1 + √ε)
    """
    n = len(Ls)
    sqrt_eps = np.sqrt(epsilon)
    return (2 * n / np.sum(np.array(Ls) + np.array(mus))) * (1 - sqrt_eps) / (1 + sqrt_eps)


def richtarik_distributed_rate(epsilon, Ls, mus):
    """
    Compute the EF21 rate from Richtarik et al. in the distributed PL setting.

    The theorem gives
        rho = 1 - gamma * mu
    with the largest admissible step size
        gamma = min{ 1 / (L + L_tilde * sqrt(2 beta / theta)), theta / (2 mu) },
    where L is the smoothness of the averaged objective, mu is its PL constant,
    L_tilde = sqrt((1/n) sum_i L_i^2), theta = 1 - sqrt(epsilon),
    and beta = epsilon / (1 - sqrt(epsilon)).

    For the regularity-heterogeneous quadratic settings used in this repo, we use
    L = mean(L_i) and mu = mean(mu_i).
    """
    eps = np.asarray(epsilon, dtype=float)
    if np.any((eps < 0) | (eps > 1)):
        raise ValueError("epsilon must belong to [0, 1].")

    L_bar = float(np.mean(Ls))
    mu_bar = float(np.mean(mus))
    L_tilde = float(np.sqrt(np.mean(np.square(Ls))))

    sqrt_eps = np.sqrt(eps)
    theta = 1.0 - sqrt_eps

    rates = np.ones_like(eps, dtype=float)
    valid_mask = theta > 0
    if np.any(valid_mask):
        theta_valid = theta[valid_mask]
        beta_valid = eps[valid_mask] / theta_valid
        gamma_pl = theta_valid / (2.0 * mu_bar)
        gamma_smooth = 1.0 / (
            L_bar + L_tilde * np.sqrt(2.0 * beta_valid / theta_valid)
        )
        gamma = np.minimum(gamma_pl, gamma_smooth)
        rates[valid_mask] = 1.0 - gamma * mu_bar

    if np.isscalar(epsilon):
        return float(rates)
    return rates


# ------------------------------------------------------------------------------
# Figure 2: Polynomial Bifurcation Plot (Empirical Law 4.3, n=2)
# ------------------------------------------------------------------------------

def generate_polynomial_bifurcation_plot():
    """
    Generate bifurcation plots of the cubic roots for the n=2 empirical law.

    Settings:
    - L1 = L2 = 1
    - mu1 = 0.9
    - mu2 in {1.0, 0.1, 0.01}
    - epsilon in [0, 1] with resolution 100
    """
    print("Generating Figure 2: polynomial_bifurcation.pdf ...")

    set_matplotlib_style()

    epsilons = np.linspace(0.0, 1.0, GRID_RESOLUTION)

    L1 = L2 = 1.0
    mu1 = 0.9
    mu2_vals = [1.0, 0.1, 0.01]

    root_colors = ['#E07A5F', '#3D405B', '#81B29A']
    root_labels = [r'Root 1', r'Root 2', r'Root 3']

    fig, axes = plt.subplots(1, len(mu2_vals), figsize=(15, 5), dpi=150, sharey=True)

    for col_idx, mu2 in enumerate(mu2_vals):
        ax = axes[col_idx]
        roots_by_index = np.full((len(epsilons), 3), np.nan)

        # Reference: rho = epsilon (draw first so it stays in the background)
        ax.plot(epsilons, epsilons, color='#9E9E9E', linewidth=2.0,
                linestyle=(0, (4, 2)), alpha=0.7,
                label=r'$\rho=\epsilon$' if col_idx == len(mu2_vals) - 1 else "")

        for idx, eps in enumerate(epsilons):
            roots = polynomial_roots_n2(eps, L1, mu1, L2, mu2)
            if np.any(np.abs(roots.imag) > ROOT_IMAG_TOL):
                print(f"Error: complex roots at epsilon={eps:.4f}, mu2={mu2}. Roots: {roots}")
                continue
            real_roots = np.sort(roots.real)
            roots_by_index[idx, :] = real_roots

        for root_idx in range(3):
            ax.plot(epsilons, roots_by_index[:, root_idx], color=root_colors[root_idx],
                    linewidth=2.5, label=root_labels[root_idx] if col_idx == len(mu2_vals) - 1 else "")

        ax.set_xlabel(r'$\epsilon$', fontsize=LABEL_SIZE)
        if col_idx == 0:
            ax.set_ylabel(r'$\rho$', fontsize=LABEL_SIZE)

        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, linestyle=':', alpha=0.5)

        kappa2 = L2 / mu2
        ax.text(**standard_textbox(f'$\\kappa_2 = {kappa2:g}$',
                                   {'x': 0.95, 'y': 0.05, 'ha': 'right', 'va': 'bottom'}),
               transform=ax.transAxes)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=4, fontsize=18, frameon=False)

    fig.tight_layout()
    fig.savefig('figures/polynomial_bifurcation.pdf', bbox_inches='tight')
    plt.close(fig)
    print("✓ Generated figures/polynomial_bifurcation.pdf")


# ------------------------------------------------------------------------------
# Figure 1: Empirical Law vs Theorem Rates (Appendix)
# ------------------------------------------------------------------------------

def generate_empirical_law_vs_theorem_rates():
    """
    Compare the n=2 empirical-law rate with homogeneous rates using worst-case
    and averaged parameters.

    Settings:
    - L1 = L2 = 1
    - mu1 = 1, mu2 in {1, 0.1, 0.001}
    - epsilon in [0, 1] with resolution 100
    """
    print("Generating Figure 1: empirical_law_vs_theorem_rates.pdf ...")

    set_matplotlib_style()

    epsilons = np.linspace(0.0, 1.0, GRID_RESOLUTION)

    L1 = L2 = 1.0
    mu1 = 1.0
    mu2_vals = [1.0, 0.1, 0.001]

    fig, axes = plt.subplots(1, len(mu2_vals), figsize=(15, 5), dpi=150, sharey=True)

    for col_idx, mu2 in enumerate(mu2_vals):
        ax = axes[col_idx]

        # Empirical law: largest real root of the cubic
        conj_rates = np.array([polynomial_rate_n2(eps, L1, mu1, L2, mu2) for eps in epsilons])

        # Theorem 1: worst-case homogeneous rate using max L and min mu
        L_worst = max(L1, L2)
        mu_worst = min(mu1, mu2)
        thm1_rates = np.array([homogeneous_optimal_rate(eps, L_worst, mu_worst) for eps in epsilons])

        # Averaged-parameter homogeneous rate used for the linear-compressor comparison
        L_bar = 0.5 * (L1 + L2)
        mu_bar = 0.5 * (mu1 + mu2)
        thm2_rates = np.array([homogeneous_optimal_rate(eps, L_bar, mu_bar) for eps in epsilons])

        ax.plot(epsilons, conj_rates, color='#E07A5F', linewidth=2.5,
                linestyle=(0, (6, 2)), label='Heterogeneous' if col_idx == len(mu2_vals) - 1 else "")
        ax.plot(epsilons, thm1_rates, color='#3D405B', linewidth=2.5,
                linestyle=(0, (3, 2)), label='Homogeneous' if col_idx == len(mu2_vals) - 1 else "")
        ax.plot(epsilons, thm2_rates, color='#81B29A', linewidth=2.5,
                linestyle=(0, (1, 1)), label=r'Linear $\mathcal{C}(\cdot)$' if col_idx == len(mu2_vals) - 1 else "")

        ax.set_xlabel(r'$\epsilon$', fontsize=LABEL_SIZE)
        if col_idx == 0:
            ax.set_ylabel(r'$\rho$', fontsize=LABEL_SIZE)

        ax.set_ylim(bottom=0.0, top=1.0)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, linestyle=':', alpha=0.5)

        ax.text(**standard_textbox(rf'$\mu_2 = {mu2:g}$',
                                   {'x': 0.95, 'y': 0.05, 'ha': 'right', 'va': 'bottom'}),
               transform=ax.transAxes)

        # Zoom inset on last panel: compare linear-compressor rate vs heterogeneous rate.
        if col_idx == len(mu2_vals) - 1:
            inset_ax = inset_axes(ax, width="30%", height="30%", loc="center right", borderpad=0.6)
            inset_ax.plot(epsilons, conj_rates, color='#E07A5F', linewidth=2.0)
            inset_ax.plot(epsilons, thm2_rates, color='#81B29A', linestyle=(0, (1, 1)), linewidth=2.0)

            inset_ax.set_xlim(0.43, 0.53)
            inset_ax.set_ylim(0.77, 0.82)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.tick_params(length=0)
            inset_ax.grid(False)
            mark_inset(ax, inset_ax, loc1=1, loc2=3, fc="none", ec="0.5", linewidth=1.0)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=3, fontsize=16, frameon=False)

    fig.tight_layout()
    fig.savefig('figures/empirical_law_vs_theorem_rates.pdf', bbox_inches='tight')
    plt.close(fig)
    print("✓ Generated figures/empirical_law_vs_theorem_rates.pdf")


def generate_L_heterogeneous_richtarik_log_complexity():
    """
    Plot log(rho_Richtarik) / log(rho_star) for the distributed Richtarik EF21
    rate versus the empirical-law rate, with shared strong-convexity constants
    and one varying smoothness constant.

    Settings:
    - L1 = 1
    - L2 in {1, 5, 500}
    - mu1 = mu2 = 0.5
    - epsilon in [0.01, 0.99] with resolution 100
    """
    print("Generating Figure 3: richtarik_multiagent_log_complexity_L_heterogeneity.pdf ...")

    set_matplotlib_style()

    epsilons = EPSILON_VALS
    L1 = 1.0
    L2_vals = [1.0, 5.0, 500.0]
    mu1 = mu2 = 0.5

    ratio_by_L2 = {}
    global_min, global_max = np.inf, -np.inf

    for L2 in L2_vals:
        empirical_rates = np.array(
            [polynomial_rate_n2(eps, L1, mu1, L2, mu2) for eps in epsilons]
        )
        richtarik_rates = richtarik_distributed_rate(epsilons, [L1, L2], [mu1, mu2])

        ratio = np.full_like(empirical_rates, np.nan, dtype=float)
        valid_mask = (
            (empirical_rates > 0)
            & (empirical_rates < 1)
            & (richtarik_rates > 0)
            & (richtarik_rates < 1)
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio[valid_mask] = (
                np.log(richtarik_rates[valid_mask])
                / np.log(empirical_rates[valid_mask])
            )

        ratio_by_L2[L2] = ratio

        finite_mask = np.isfinite(ratio)
        if finite_mask.any():
            global_min = min(global_min, np.nanmin(ratio[finite_mask]))
            global_max = max(global_max, np.nanmax(ratio[finite_mask]))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = 1.0, 1.1

    span = global_max - global_min
    padding = 0.05 * span if span > 1e-6 else 0.02
    y_lower = max(0.0, global_min - padding)
    y_upper = global_max + padding

    fig, axes = plt.subplots(1, len(L2_vals), figsize=(15, 4.1), dpi=150, sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    ylabel = r"$\log \rho_{\text{Richt\'arik}} / \log \rho_{\star}$"

    for ax, L2 in zip(axes_array, L2_vals):
        ratio = ratio_by_L2[L2]
        line_plot(
            [(epsilons, ratio, {'color': 'blue', 'linewidth': 3.0, 'zorder': 3})],
            ax=ax,
            plt_legend=False,
            xlabel=r'$\epsilon$',
            ylabel=ylabel if ax is axes_array[0] else None,
            label_size=LABEL_SIZE,
            tick_size=TICK_SIZE,
            return_plt=True,
        )
        ax.set_title(rf'$L^{{(2)}} = {L2:g}$', fontsize=LABEL_SIZE, pad=10)
        ax.set_ylim(y_lower, y_upper)

        finite_mask = np.isfinite(ratio)
        if finite_mask.any():
            min_y = float(np.nanmin(ratio))

            ax.axhline(
                min_y,
                color='#009E73',
                linestyle='--',
                linewidth=1.3,
                alpha=0.75,
                zorder=2,
            )

            right_ax = ax.twinx()
            right_ax.set_ylim(y_lower, y_upper)
            right_ax.set_yticks([min_y])
            right_ax.set_yticklabels([f'{min_y:.2f}'], color='#009E73', fontsize=TICK_SIZE)
            right_ax.tick_params(axis='y', colors='#009E73', length=0, pad=3)
            right_ax.set_xticks([])
            right_ax.grid(False)
            for spine in right_ax.spines.values():
                spine.set_visible(False)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig('figures/richtarik_multiagent_log_complexity_L_heterogeneity.pdf', bbox_inches='tight')
    plt.close(fig)
    print("✓ Generated figures/richtarik_multiagent_log_complexity_L_heterogeneity.pdf")






# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate experiments for ICML multi-worker paper'
    )
    parser.add_argument(
        'experiment', 
        type=str, 
        help='Name of experiment to run or "all"'
    )
    args = parser.parse_args()
    
    experiment_functions = {
        "Figure 1": generate_empirical_law_vs_theorem_rates,
        "Figure 2": generate_polynomial_bifurcation_plot,
        "Figure 3": generate_L_heterogeneous_richtarik_log_complexity,
                    }
    
    if args.experiment == "all":
        print("Running all experiments...")
        for name, func in experiment_functions.items():
            func()
            print(f"✓ Completed {name}\n")
        print("All experiments completed!")
    elif args.experiment not in experiment_functions:
        print(f"Error: Unknown experiment '{args.experiment}'")
        print("Available experiments:")
        for exp in experiment_functions.keys():
            print(f"  - {exp}")
        print("  - all (runs all experiments)")
        return
    else:
        experiment_functions[args.experiment]()


if __name__ == "__main__":
    main()
