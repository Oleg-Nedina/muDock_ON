"""
plotter_roofline.py
Generates the publication-quality Application Roofline Model (muDock in toto)
using a dual-view layout:
  - Left Panel: Global Accelerator Capacity (0-100% of A100 Peak) showing hardware ceilings.
  - Right Panel: Detailed Operational Zoom (0-22% BW, 0-14% SM) showing exact backend comparison.
Produces: roofline_application_big.pdf and ncu_roofline.pdf
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Amdahl kernel time weights from full application profiling:
# adt_score (scoring) is ~93.2%, geom_transform is ~6.2%, genetic is ~0.6%
WEIGHTS = {
    "adt_score":      0.932,
    "geom_transform": 0.062,
    "genetic":        0.006,
}

BACKEND_STYLES = {
    "CUDA 1W":        {"color": "#E74C3C", "marker": "o", "label": "CUDA (1 Worker)"},
    "CUDA 2W":        {"color": "#922B21", "marker": "s", "label": "CUDA (2 Workers)"},
    "Alpaka CUDA 1W": {"color": "#3498DB", "marker": "^", "label": "Alpaka CUDA (1 Worker)"},
    "Alpaka CUDA 2W": {"color": "#1B4F72", "marker": "D", "label": "Alpaka CUDA (2 Workers)"},
}

def plot_dual_roofline(csv_path: str = "roofline_metrics_big.csv", out_path: str = "roofline_application_big.pdf"):
    if not Path(csv_path).exists():
        print(f"[-] File non trovato: {csv_path}", file=sys.stderr)
        return

    df = pd.read_csv(csv_path)

    bw_df = df[df["Metric"] == "Memory Bandwidth (%)"].groupby(["Backend", "Kernel"])["Value"].mean().reset_index().rename(columns={"Value": "BW%"})
    sm_df = df[df["Metric"] == "SM Compute Throughput (%)"].groupby(["Backend", "Kernel"])["Value"].mean().reset_index().rename(columns={"Value": "SM%"})
    merged = bw_df.merge(sm_df, on=["Backend", "Kernel"])

    if merged.empty:
        print("[-] Dati insufficienti per il Roofline.")
        return

    # Compute application-level weighted average for each backend
    app_rows = []
    for backend, grp in merged.groupby("Backend"):
        total_w = sum(WEIGHTS.get(k, 1.0) for k in grp["Kernel"])
        if total_w > 0:
            app_bw = sum(row["BW%"] * WEIGHTS.get(row["Kernel"], 1.0) for _, row in grp.iterrows()) / total_w
            app_sm = sum(row["SM%"] * WEIGHTS.get(row["Kernel"], 1.0) for _, row in grp.iterrows()) / total_w
        else:
            app_bw = grp["BW%"].mean()
            app_sm = grp["SM%"].mean()
        app_rows.append({"Backend": backend, "BW%": app_bw, "SM%": app_sm})

    app_df = pd.DataFrame(app_rows)

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "figure.dpi":       300,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "axes.grid":        True,
        "grid.alpha":       0.35,
        "grid.linestyle":   "--",
        "axes.axisbelow":   True,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ─────────────────────────────────────────────────────────────────────────────
    # PANEL 1: GLOBAL ACCELERATOR CAPACITY (0 to 105%)
    # ─────────────────────────────────────────────────────────────────────────────
    ax1.axhline(100, color="crimson", linewidth=1.5, linestyle="--", label="Compute Peak: 312 TFLOPS (100%)")
    ax1.axvline(100, color="steelblue", linewidth=1.5, linestyle="--", label="HBM2e Peak: 2039 GB/s (100%)")
    x_ridge = np.linspace(0, 100, 200)
    ax1.plot(x_ridge, np.minimum(x_ridge, 100), color="gray", linewidth=1.0, linestyle=":", label="Ridge Line (y = x)")

    for _, row in app_df.iterrows():
        b = row["Backend"]
        st = BACKEND_STYLES.get(b, {"color": "black", "marker": "o", "label": b})
        ax1.scatter(row["BW%"], row["SM%"], color=st["color"], marker=st["marker"],
                    s=120, edgecolors="black", linewidths=1.0, zorder=6, label=st["label"])

    ax1.set_xlim(0, 105)
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("Memory Bandwidth Utilization (% of 2039 GB/s Peak)", fontsize=10, fontweight="bold", labelpad=8)
    ax1.set_ylabel("SM Compute Throughput (% of 312 TFLOPS Peak)", fontsize=10, fontweight="bold", labelpad=8)
    ax1.set_title("(a) Full Accelerator Dynamic Range (A100 SXM4)", fontsize=11, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, fontsize=8, framealpha=0.9)
    sns.despine(ax=ax1)

    # Indicate zoom area on Panel 1
    rect = plt.Rectangle((0, 0), 22, 14, linewidth=1.2, edgecolor="darkgreen", facecolor="none", linestyle="-.", zorder=10)
    ax1.add_patch(rect)
    ax1.annotate("Zoom Region (b) →", (23, 7), color="darkgreen", fontsize=8.5, fontweight="bold", va="center")

    # ─────────────────────────────────────────────────────────────────────────────
    # PANEL 2: ZOOMED OPERATIONAL SPACE (muDock in toto)
    # ─────────────────────────────────────────────────────────────────────────────
    # Local ridge line
    x_zoom = np.linspace(0, 25, 100)
    ax2.plot(x_zoom, x_zoom, color="gray", linewidth=1.0, linestyle=":", label="Ridge Line (y = x)")

    for _, row in app_df.iterrows():
        b = row["Backend"]
        st = BACKEND_STYLES.get(b, {"color": "black", "marker": "o", "label": b})
        ax2.scatter(row["BW%"], row["SM%"], color=st["color"], marker=st["marker"],
                    s=160, edgecolors="black", linewidths=1.2, zorder=6, label=st["label"])

        # Offset annotation so labels don't collide
        offset_y = 0.5 if "2W" in b else -0.8
        offset_x = 0.6 if "CUDA" in b and "Alpaka" not in b else 0.6
        if b == "CUDA 1W":
            offset_y = 0.4
        elif b == "Alpaka CUDA 1W":
            offset_y = -0.7

        ax2.annotate(f" {st['label']}\n ({row['BW%']:.1f}% BW, {row['SM%']:.1f}% SM)",
                    (row["BW%"], row["SM%"] + offset_y * 0.5),
                    fontsize=8.5, fontweight="bold", zorder=7)

    ax2.set_xlim(0, 22)
    ax2.set_ylim(0, 14)
    ax2.set_xlabel("Memory Bandwidth Utilization (% of 2039 GB/s Peak)", fontsize=10, fontweight="bold", labelpad=8)
    ax2.set_ylabel("SM Compute Throughput (% of 312 TFLOPS Peak)", fontsize=10, fontweight="bold", labelpad=8)
    ax2.set_title("(b) Operational Zoom — muDock in toto (BIG Dataset)", fontsize=11, fontweight="bold", pad=12)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5, framealpha=0.9)
    sns.despine(ax=ax2)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    # Also overwrite ncu_roofline.pdf for seamless integration
    fig.savefig("ncu_roofline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved Dual-Panel Application Roofline → {out_path}")
    print(f"  ✔  Saved Dual-Panel Application Roofline → ncu_roofline.pdf")

def main():
    plot_dual_roofline("roofline_metrics_big.csv", "roofline_application_big.pdf")

if __name__ == "__main__":
    main()
