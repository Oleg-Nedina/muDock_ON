"""
profiler_roofline_big.py
Dedicated Roofline Profiler for muDock in toto on the BIG dataset (400k ligands).
Stresses the GPU under full multi-ligand batch load and measures realistic
operational intensity (Memory Bandwidth % and SM Compute Throughput %).
Uses --launch-count 25 and --time_limit_sec 60 so it finishes in ~3-4 minutes total
without timing out.
Produces: roofline_metrics_big.csv + roofline_application_big.pdf
"""

import os
import io
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROTEIN = "/work/onedina/muDock_ON/data/1fkb/1fkb_pocket.pdbqt"
# HEAVY BIG DATASET to fully saturate GPU memory bandwidth and SMs
LIGAND  = "/work/onedina/muDock_ON/data/big_1/big.adtmol2"

BACKENDS = [
    ("CUDA 1W",          "/work/onedina/muDock_ON/build/cuda/application/muDock",        "CUDA:GPU:0"),
    ("CUDA 2W",          "/work/onedina/muDock_ON/build/cuda/application/muDock",        "CUDA:GPU:0:2:4000000000"),
    ("Alpaka CUDA 1W",   "/work/onedina/muDock_ON/build/alpaka-cuda/application/muDock", "ALPAKA:GPU:0"),
    ("Alpaka CUDA 2W",   "/work/onedina/muDock_ON/build/alpaka-cuda/application/muDock", "ALPAKA:GPU:0:2:4000000000"),
]

# Workload configuration on the big dataset
POPULATION   = 20
GENERATIONS  = 100
LAUNCH_COUNT = 25    # Profile first 25 heavy multi-ligand kernel launches
TIME_LIMIT   = 60    # muDock stops gracefully after 60s, preventing any runaway execution
TIMEOUT_SEC  = 180   # subprocess safety timeout

# Core Roofline metrics on NVIDIA A100
METRICS = ",".join([
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", # Memory Bandwidth (% of peak)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",                 # SM Compute Throughput (% of peak)
    "sm__warps_active.avg.pct_of_peak_sustained_active",               # Achieved Occupancy (%)
    "launch__registers_per_thread",                                     # Registers per thread
])

# Hardware Peak Ratings for A100-SXM4-40GB
PEAK_FLOPS_TFLOPS = 312.0
PEAK_BW_GBPS      = 2039.0

# Amdahl kernel time weights from full application profiling:
# adt_score (scoring) is ~93.2%, geom_transform is ~6.2%, genetic is ~0.6%
WEIGHTS = {
    "adt_score":      0.932,
    "geom_transform": 0.062,
    "genetic":        0.006,
}

# ─── PLOT STYLE ──────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":             10,
    "axes.titlesize":        12,
    "axes.titleweight":      "bold",
    "axes.labelsize":        10.5,
    "axes.labelweight":      "bold",
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       8.5,
    "legend.title_fontsize": 9,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "axes.grid":             True,
    "grid.alpha":            0.35,
    "grid.linestyle":        "--",
    "axes.axisbelow":        True,
})

def banner(title: str, width: int = 68) -> None:
    print(f"\n╔{'═' * (width - 2)}╗")
    print(f"║  {title:<{width - 4}}║")
    print(f"╚{'═' * (width - 2)}╝")

def section(msg: str) -> None: print(f"\n  ▶  {msg}")
def ok(msg: str)      -> None: print(f"  ✔  {msg}")
def warn(msg: str)    -> None: print(f"  ⚠  {msg}", file=sys.stderr)


# ─── NCU RUNNER ──────────────────────────────────────────────────────────────
def run_ncu_roofline(binary: str, backend_name: str, use_flag: str) -> str:
    cmd = [
        "ncu", "--csv", "--page", "details",
        "--launch-count", str(LAUNCH_COUNT),
        "--metrics", METRICS,
        binary,
        "--use",            use_flag,
        "--protein",        PROTEIN,
        "--ligand",         LIGAND,
        "--population",     str(POPULATION),
        "--generations",    str(GENERATIONS),
        "--search",         "genetic",
        "--seed",           "42",
        "--time_limit_sec", str(TIME_LIMIT),
    ]
    print(f"  ncu (BIG)  {backend_name:<16}  Pop:{POPULATION} Gen:{GENERATIONS} (limit {TIME_LIMIT}s) …", flush=True)
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT_SEC)
        return res.stdout
    except subprocess.TimeoutExpired:
        warn(f"NCU timed out after {TIMEOUT_SEC}s for {backend_name}")
        return ""


def parse_ncu_output(output: str, backend: str) -> list[dict]:
    lines = output.splitlines()
    csv_start = next(
        (i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")),
        None,
    )
    if csv_start is None:
        warn(f"No CSV data found in NCU output for {backend}")
        return []

    df = pd.read_csv(io.StringIO("\n".join(lines[csv_start:])))
    records = []

    KERNEL_MAP = {
        "adt_score":      ["adt_score", "calc_energy"],
        "genetic":        ["genetic", "iterate", "finalize", "initialize"],
        "geom_transform": ["geom_transform", "apply_alpaka", "apply"],
    }
    METRIC_MAP = {
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "Memory Bandwidth (%)",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed":                 "SM Compute Throughput (%)",
        "sm__warps_active.avg.pct_of_peak_sustained_active":               "Achieved Occupancy (%)",
        "launch__registers_per_thread":                                     "Registers / Thread",
    }

    for _, row in df.iterrows():
        kname = str(row.get("Kernel Name", "")).lower()
        mname = str(row.get("Metric Name", ""))
        mval  = str(row.get("Metric Value", "0")).replace(",", "")

        k_clean = next(
            (k for k, keywords in KERNEL_MAP.items() if any(kw in kname for kw in keywords)),
            None,
        )
        m_clean = METRIC_MAP.get(mname)
        if k_clean is None or m_clean is None:
            continue

        try:
            val = float(mval)
        except ValueError:
            val = 0.0

        records.append({
            "Backend": backend,
            "Kernel":  k_clean,
            "Metric":  m_clean,
            "Value":   val,
        })

    return records


# ─── PLOTTER (APPLICATION ROOFLINE IN TOTO) ──────────────────────────────────
def plot_roofline_in_toto(df: pd.DataFrame, out_path: str = "roofline_application_big.pdf") -> None:
    section("Generating Application Roofline Model (muDock in toto on BIG dataset)…")

    bw_df = df[df["Metric"] == "Memory Bandwidth (%)"].groupby(["Backend", "Kernel"])["Value"].mean().reset_index().rename(columns={"Value": "BW%"})
    sm_df = df[df["Metric"] == "SM Compute Throughput (%)"].groupby(["Backend", "Kernel"])["Value"].mean().reset_index().rename(columns={"Value": "SM%"})
    merged = bw_df.merge(sm_df, on=["Backend", "Kernel"])

    if merged.empty:
        warn("Not enough data to generate Roofline plot.")
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

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # Roofline ceiling lines
    ax.axhline(100, color="crimson", linewidth=1.5, linestyle="--", label="Compute ceiling (100% SM Peak: 312 TFLOPS)")
    ax.axvline(100, color="steelblue", linewidth=1.5, linestyle="--", label="Bandwidth ceiling (100% HBM Peak: 2039 GB/s)")
    
    # Memory-bound ridge line: y = x
    x_ridge = np.linspace(0, 100, 200)
    ax.plot(x_ridge, np.minimum(x_ridge, 100), color="gray", linewidth=1.0,
            linestyle=":", label="Memory-Bound Ridge (y = x)")

    BACKEND_STYLES = {
        "CUDA 1W":        {"color": "#E74C3C", "marker": "o", "label": "CUDA (1 Worker)"},
        "CUDA 2W":        {"color": "#922B21", "marker": "s", "label": "CUDA (2 Workers)"},
        "Alpaka CUDA 1W": {"color": "#3498DB", "marker": "^", "label": "Alpaka CUDA (1 Worker)"},
        "Alpaka CUDA 2W": {"color": "#1B4F72", "marker": "D", "label": "Alpaka CUDA (2 Workers)"},
    }

    # Plot Application In-Toto data points
    for _, row in app_df.iterrows():
        b = row["Backend"]
        style = BACKEND_STYLES.get(b, {"color": "black", "marker": "o", "label": b})
        ax.scatter(row["BW%"], row["SM%"],
                   color=style["color"], marker=style["marker"], s=160, zorder=6,
                   edgecolors="black", linewidths=1.2, label=style["label"])
        
        # Position annotation with slight offset
        ax.annotate(f"  {style['label']}\n  ({row['BW%']:.1f}% BW, {row['SM%']:.1f}% SM)",
                    (row["BW%"], row["SM%"]),
                    xytext=(8, -5), textcoords="offset points",
                    fontsize=9, fontweight="bold", zorder=7)

    # Max limits with margin
    max_x = max(105, app_df["BW%"].max() * 1.3)
    max_y = max(105, app_df["SM%"].max() * 1.3)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    
    ax.set_xlabel("Memory Bandwidth Utilization (% of 2039 GB/s A100 Peak)", labelpad=8)
    ax.set_ylabel("SM Compute Throughput (% of 312 TFLOPS A100 Peak)", labelpad=8)
    ax.set_title("Application Roofline Model — muDock in toto\n(Heavy Multi-Ligand BIG Dataset on NVIDIA A100)", pad=14)
    ax.legend(title="Application Backend", loc="upper left", frameon=True, fontsize=8.5, framealpha=0.9)
    sns.despine(ax=ax)
    plt.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    ok(f"Saved Application Roofline → {out_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    banner("DEDICATED ROOFLINE PROFILER — muDock in toto on BIG Dataset")

    all_records = []
    for name, bin_path, use_flag in BACKENDS:
        if not os.path.exists(bin_path):
            warn(f"Executable not found → {bin_path} (skipping {name})")
            continue

        raw = run_ncu_roofline(bin_path, name, use_flag)
        recs = parse_ncu_output(raw, name)
        all_records.extend(recs)

    if not all_records:
        warn("No profiling data collected. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    csv_path = "roofline_metrics_big.csv"
    df.to_csv(csv_path, index=False)
    ok(f"Saved raw metrics → {csv_path}")

    # Generate the Application Roofline
    plot_roofline_in_toto(df, "roofline_application_big.pdf")
    # Also overwrite ncu_roofline.pdf for seamless integration
    plot_roofline_in_toto(df, "ncu_roofline.pdf")

if __name__ == "__main__":
    main()
