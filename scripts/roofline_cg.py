#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import RESULTS_DIR, savefig

# Machine constants
PEAK_GFLOPS = 45.3 * 1000  # = 45,300 GFLOP/s for MI210
PEAK_GBS    = 1600         # GB/s

def main():
    # Load CSV from results/
    df = pd.read_csv(RESULTS_DIR / "results_phase4.csv")

    # FLOP/byte model for 2D 5-point stencil
    # - 4 adds/subs + 2 muls = 6 FLOPs per interior point
    # - 5 loads + 1 store of doubles = 48 bytes per interior point
    FLOPS_PER_POINT = 6.0
    BYTES_PER_POINT = 48.0

    N      = df["N"].astype(int)
    iters  = df["iters"].astype(int)
    t_applyA = df["t_applyA"].astype(float)  # total apply_A time over all iterations

    # Interior points (global)
    points_interior = (N - 2) ** 2

    # FLOPs & bytes per apply_A call (global)
    flops_per_call  = FLOPS_PER_POINT * points_interior
    bytes_per_call  = BYTES_PER_POINT * points_interior

    # Totals over all iterations
    df["points_interior"]    = points_interior
    df["flops_total_applyA"] = flops_per_call * iters
    df["bytes_total_applyA"] = bytes_per_call * iters

    # Arithmetic intensity (FLOPs / Byte)
    df["AI_applyA"] = df["flops_total_applyA"] / df["bytes_total_applyA"]

    # Performance (global)
    df["GFLOPs_applyA"] = df["flops_total_applyA"] / (1e9 * t_applyA)
    df["GBs_applyA"]    = df["bytes_total_applyA"] / (1e9 * t_applyA)

    # Convert to per-GPU/per-rank if nprocs present
    if "nprocs" in df.columns:
        df["GFLOPs_applyA"] /= df["nprocs"]
        df["GBs_applyA"]    /= df["nprocs"]

    # Keep only the most performant run per N (highest per-GPU GFLOP/s)
    df_best = (
        df.sort_values("GFLOPs_applyA", ascending=False)
          .groupby("N", as_index=False)
          .first()
    )

    # Print summary
    cols_to_show = ["N", "iters", "AI_applyA", "GFLOPs_applyA", "GBs_applyA"]
    if "nprocs" in df.columns:
        cols_to_show.insert(1, "nprocs")

    print("\n=== Phase 4 apply_A roofline metrics ===")
    print(df_best[cols_to_show].to_string(index=False))

    # Use only single-GPU runs for the roofline plot
    if "nprocs" in df.columns:
        df_plot = df[df["nprocs"] == 1].copy()
    else:
        df_plot = df.copy()

    ai_vals = df_plot["AI_applyA"].values
    ai_min  = ai_vals.min() / 10.0
    ai_max  = ai_vals.max() * 10.0
    ai = np.logspace(np.log10(ai_min), np.log10(ai_max), 200)

    roof_bw = PEAK_GBS * ai

    plt.figure(figsize=(7, 5))

    # Bandwidth roof
    plt.loglog(ai, roof_bw, "--", label=f"BW roof ({PEAK_GBS} GB/s)")

    # One color (and legend entry) per N
    unique_N = sorted(df_plot["N"].unique())
    for Nval in unique_N:
        sub = df_plot[df_plot["N"] == Nval]
        plt.loglog(
            sub["AI_applyA"],
            sub["GFLOPs_applyA"],
            "o",
            label=f"N={int(Nval)}"
        )

    plt.xlabel("Arithmetic Intensity (FLOPs / Byte)")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title("Roofline for apply_A (P=1 runs)")
    plt.legend()
    plt.tight_layout()

    # Save + (optionally) show
    savefig("roofline_applyA.png")
    plt.show()

    # Save augmented CSV back into results/
    out_csv = RESULTS_DIR / "results_phase4_with_roofline.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

if __name__ == "__main__":
    main()
