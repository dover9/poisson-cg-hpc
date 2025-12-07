#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter  # unused; safe to remove

from utils import RESULTS_DIR, savefig


def main():
    df = pd.read_csv(RESULTS_DIR / "results_phase4.csv")

    # -------------------------
    # Strong scaling for largest N
    # -------------------------
    N_strong = 4097
    dfN = df[df["N"] == N_strong].copy()
    dfN["t_iter"] = dfN["runtime_sec"] / dfN["iters"]
    dfN = dfN.sort_values("nprocs")

    print(f"\nStrong scaling (N = {N_strong}):")
    print(dfN[["N", "nprocs", "iters", "t_cg_total", "t_iter"]].to_string(index=False))

    t_ms = 1e3 * dfN["t_iter"]

    plt.figure(figsize=(6, 4))
    plt.plot(dfN["nprocs"], t_ms, marker="o")

    ax = plt.gca()
    ax.set_title(f"Strong scaling of CG (N={N_strong})")
    ax.set_xlabel("Number of GPUs (P)")
    ax.set_ylabel("Time per iteration (ms)")

    # GPU tick labels
    ax.set_xticks([1, 2, 4, 8])

    # custom y-ticks
    ax.set_yticks([0.4, 0.6, 1.0, 2.0])
    ax.set_ylim(0.35, 2.05)

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    savefig("strong_scaling_time_per_iter.png")
    plt.show()

    # -------------------------
    # Figure 3: Speedup & efficiency
    # -------------------------
    T1 = dfN.loc[dfN["nprocs"] == 1, "t_iter"].iloc[0]
    dfN["speedup"] = T1 / dfN["t_iter"]
    dfN["efficiency"] = dfN["speedup"] / dfN["nprocs"]

    print(f"\nSpeedup / Efficiency (N = {N_strong}):")
    print(dfN[["nprocs", "t_iter", "speedup", "efficiency"]].to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.plot(dfN["nprocs"], dfN["speedup"], "o-", label="Speedup")
    plt.plot(dfN["nprocs"], dfN["nprocs"], "k--", label="Ideal speedup")
    plt.xlabel("Number of GPUs (P)")
    plt.ylabel(r'Speedup $S(P) = \frac{T(1)}{T(P)}$')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.title(f"Strong scaling speedup (N={N_strong})")
    plt.xticks(dfN["nprocs"], dfN["nprocs"])
    plt.legend()
    plt.tight_layout()
    savefig("strong_scaling_speedup.png")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(dfN["nprocs"], dfN["efficiency"], "o-")
    plt.xlabel("Number of GPUs (P)")
    plt.ylabel(r'Parallel efficiency $\eta(P) = \frac{T(1)}{P\,T(P)}$')
    plt.title(f"Parallel efficiency of CG (N={N_strong})")
    plt.xticks(dfN["nprocs"], dfN["nprocs"])
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.figtext(
        0.5, -0.10,
        r'$T(P)$ = wall-clock time per CG iteration for fixed $N=4097$',
        ha='center', fontsize=10
    )
    plt.tight_layout()
    savefig("strong_scaling_efficiency.png")
    plt.show()

    # -------------------------
    # Runtime breakdown for N = 4097
    # -------------------------
    components = ["t_halo", "t_applyA", "t_axpy_u", "t_axpy_r", "t_update_p", "t_dot"]

    # Filter to N_strong and sort by number of GPUs
    dfN = df[df["N"] == N_strong].copy().sort_values("nprocs")

    # Use sum of components as the denominator
    total_comp = dfN[components].sum(axis=1)
    frac = dfN[components].div(total_comp, axis=0)

    print(f"\nRuntime breakdown (normalized component fractions) for N = {N_strong}:")
    print(frac.assign(nprocs=dfN["nprocs"]).to_string(index=False))

    cmap = plt.get_cmap("tab10")
    colors = {
        "t_dot":      cmap(0),
        "t_applyA":   cmap(1),
        "t_axpy_u":   cmap(2),
        "t_halo":     cmap(3),
        "t_update_p": cmap(4),
        "t_axpy_r":   "#c6dbef",   # light blue
    }

    # Figure: Pie chart for N = 4097, P = 8
    row_p8 = frac[dfN["nprocs"] == 8].iloc[0]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, _ = ax.pie(
        row_p8.values,
        colors=[colors[c] for c in components],
        startangle=90,
    )
    ax.set_title(f"Runtime breakdown of CG iteration (N={N_strong}, P=8)")
    ax.legend(
        wedges,
        components,
        title="Components",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
    )
    ax.axis("equal")
    plt.tight_layout()
    savefig("breakdown_pie_P8.png")
    plt.show()

    # Figure: Stacked bar vs ranks 
    x = np.arange(len(dfN))
    bottom = np.zeros(len(dfN))

    fig, ax = plt.subplots(figsize=(7, 4))
    for comp in components:
        ax.bar(x, frac[comp], bottom=bottom, label=comp, color=colors[comp])
        bottom += frac[comp].values

    ax.set_xticks(x)
    ax.set_xticklabels(dfN["nprocs"])
    ax.set_xlabel("Number of GPUs (P)")
    ax.set_ylabel("Fraction of CG iteration time")
    ax.set_title(f"Runtime breakdown of CG iterations vs GPUs (N={N_strong})")
    ax.legend(title="Components")
    plt.tight_layout()
    savefig("breakdown_stacked.png")
    plt.show()

if __name__ == "__main__":
    main()
