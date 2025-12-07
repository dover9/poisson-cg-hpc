#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import RESULTS_DIR, savefig


def best_time_for_phase(df, N):
    """Return (best_P, best_runtime) for a given phase and grid size N."""
    dfN = df[df["N"] == N]
    if dfN.empty:
        raise ValueError(f"No rows with N={N} in this phase.")
    group = dfN.groupby("nprocs")["runtime_sec"].mean()
    P_best = group.idxmin()        # P with smallest runtime
    T_best = group.loc[P_best]
    return int(P_best), float(T_best)


def main():
    # -------------------------
    # Convergence plot: iteration count vs grid size (Phase 1)
    # -------------------------
    df = pd.read_csv(RESULTS_DIR / "results_phase1.csv")

    df_small = df[df["N"].isin([257, 513, 1025, 2049, 4097])]
    df_small = df_small.sort_values("N")

    plt.figure(figsize=(6, 4))
    plt.plot(df_small["N"], df_small["iters"], marker="o")

    plt.xlabel("Grid size $N$")
    plt.ylabel("CG iteration count")
    plt.title("CG convergence: iterations vs grid size (CPU)")
    plt.grid(True, linewidth=0.3, linestyle="--", alpha=0.6)

    plt.tight_layout()
    savefig("convergence_iters_vs_N.png")
    plt.show()

    # -------------------------
    # "Look how far we've come" speedup plot across phases
    # -------------------------
    df1 = pd.read_csv(RESULTS_DIR / "results_phase1.csv")  # CPU serial
    df2 = pd.read_csv(RESULTS_DIR / "results_phase2.csv")  # MPI (CPU)
    df3 = pd.read_csv(RESULTS_DIR / "results_phase3.csv")  # GPU offload
    df4 = pd.read_csv(RESULTS_DIR / "results_phase4.csv")  # MPI + GPU

    N_demo = 4097

    # Phase 1: serial CPU baseline
    df1N = df1[df1["N"] == N_demo]
    if df1N.empty:
        raise ValueError(f"No rows with N={N_demo} in results_phase1.csv")
    T1 = df1N["runtime_sec"].iloc[0]

    P2, T2 = best_time_for_phase(df2, N_demo)
    P3, T3 = best_time_for_phase(df3, N_demo)
    P4, T4 = best_time_for_phase(df4, N_demo)

    labels = [
        "Phase 1:\nCPU serial",
        f"Phase 2:\nMPI CPU (P={P2})",
        f"Phase 3:\nGPU (P={P3})",
        f"Phase 4:\nMPI+GPU (P={P4})",
    ]

    speedups = [
        1.0,          # baseline
        T1 / T2,
        T1 / T3,
        T1 / T4,
    ]

    plt.figure(figsize=(7, 4))
    x = np.arange(len(labels))
    bars = plt.bar(x, speedups)

    plt.xticks(x, labels)
    plt.ylabel("Speedup vs Phase 1 (CPU serial)")
    plt.title(f"End-to-end speedup across phases (N={N_demo})")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars with “X.X×”
    for xi, s in zip(x, speedups):
        plt.text(xi, s + 0.3, f"{s:.1f}×", ha="center", va="bottom", fontsize=9)

    plt.ylim(0, max(speedups) * 1.25)

    plt.tight_layout()
    savefig("speedup_by_phase_N4097.png")
    plt.show()


if __name__ == "__main__":
    main()