"""
profiler_ncu.py
Extracts NCU runtime hardware counters (Occupancy, Mem Bandwidth, SM Throughput,
Registers/Thread, Uniform Branch) for each kernel, produces:
1. Application-level Roofline Model (ncu_roofline.pdf) representing muDock in toto.
2. Kernel-level bar charts for each hardware metric (ncu_<kernel>_<metric>.pdf).
Produces: ncu_metrics_configs.csv + ncu_roofline.pdf + ncu_<kernel>_<metric>.pdf
"""

import os
import io
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROTEIN = "/work/onedina/muDock_ON/data/1fkb/1fkb_pocket.pdbqt"
# Single ligand dataset so NCU microarchitectural profiling completes in minutes without timing out
LIGAND  = "/work/onedina/muDock_ON/data/1fkb/1fkb_ligand.adtmol2"

BACKENDS = [
    ("CUDA 1W",          "/work/onedina/muDock_ON/build/cuda/application/muDock",        "CUDA:GPU:0"),
    ("CUDA 2W",          "/work/onedina/muDock_ON/build/cuda/application/muDock",        "CUDA:GPU:0:2:4000000000"),
    ("Alpaka CUDA 1W",   "/work/onedina/muDock_ON/build/alpaka-cuda/application/muDock", "ALPAKA:GPU:0"),
    ("Alpaka CUDA 2W",   "/work/onedina/muDock_ON/build/alpaka-cuda/application/muDock", "ALPAKA:GPU:0:2:4000000000"),
]

CONFIGS = [
    (20,  100),   # Small:  2 k evals
    (50,  200),   # Medium: 10 k evals
    (100, 400),   # Big:    40 k evals
]

# Robust NCU metrics supported across all NVIDIA Ampere A100 kernels
METRICS = ",".join([
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__registers_per_thread",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
])

# Peak hardware values (A100 SXM4)
PEAK_FLOPS_TFLOPS  = 312.0   # FP32 TFLOPS  (A100 SXM4)
PEAK_BW_GBPS       = 2039.0  # Memory bandwidth GB/s (A100 SXM4)

# ─── PLOT STYLE ──────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       8,
    "legend.title_fontsize": 9,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "axes.grid":             True,
    "grid.alpha":            0.4,
    "grid.linestyle":        "--",
    "axes.axisbelow":        True,
})

PALETTE      = sns.color_palette("deep", n_colors=len(BACKENDS))
BACKEND_CLRS = {name: PALETTE[i] for i, (name, _, _) in enumerate(BACKENDS)}

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def banner(title: str, width: int = 66) -> None:
    print(f"\n╔{'═' * (width - 2)}╗")
    print(f"║  {title:<{width - 4}}║")
    print(f"╚{'═' * (width - 2)}╝")

def section(msg: str) -> None: print(f"\n  ▶  {msg}")
def ok(msg: str)      -> None: print(f"  ✔  {msg}")
def warn(msg: str)    -> None: print(f"  ⚠  {msg}", file=sys.stderr)


def run_ncu(binary: str, backend_name: str, use_flag: str, pop: int, gen: int) -> str:
    # --launch-count 20 captures deterministic metrics rapidly without waiting for hundreds of iterations
    cmd = [
        "ncu", "--csv", "--page", "details",
        "--launch-count", "20",
        "--metrics", METRICS,
        binary,
        "--use",         use_flag,
        "--protein",     PROTEIN,
        "--ligand",      LIGAND,
        "--population",  str(pop),
        "--generations", str(gen),
        "--search",      "genetic",
        "--seed",        "42",
    ]
    print(f"  ncu  {backend_name:<16}  Pop:{pop:<4}  Gen:{gen:<4} …", flush=True)
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        return res.stdout
    except subprocess.TimeoutExpired:
        warn(f"NCU run timed out after 10 minutes for {backend_name}")
        return ""


def parse_ncu_csv(output: str, backend: str, pop: int, gen: int) -> list[dict]:
    lines = output.splitlines()
    csv_start = next(
        (i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")),
        None,
    )
    if csv_start is None:
        warn(f"No CSV data found for {backend} (Pop:{pop} Gen:{gen})")
        return []

    df = pd.read_csv(io.StringIO("\n".join(lines[csv_start:])))
    records = []

    KERNEL_MAP = {
        "adt_score":      ["adt_score", "calc_energy"],
        "genetic":        ["genetic", "iterate", "finalize", "initialize"],
        "geom_transform": ["geom_transform", "apply_alpaka", "apply"],
    }
    METRIC_MAP = {
        "sm__warps_active.avg.pct_of_peak_sustained_active":          "Achieved Occupancy (%)",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed":            "SM Compute Throughput (%)",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "Memory Bandwidth (%)",
        "launch__registers_per_thread":                                "Registers / Thread",
        "smsp__sass_average_branch_targets_threads_uniform.pct":       "Uniform Branch (%)",
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
            "Backend":           backend,
            "Total Evaluations": pop * gen,
            "Kernel":            k_clean,
            "Metric":            m_clean,
            "Value":             val,
        })

    return records


# ─── ROOFLINE (APPLICATION IN TOTO) ──────────────────────────────────────────
def plot_roofline(df: pd.DataFrame) -> None:
    section("Generating application-level roofline plot (muDock in toto)…")

    # Time weights for kernels based on profiling (93.2% adt_score, 6.2% geom_transform, 0.6% genetic)
    WEIGHTS = {"adt_score": 0.932, "geom_transform": 0.062, "genetic": 0.006}

    bw_df  = df[df["Metric"] == "Memory Bandwidth (%)"][["Backend", "Kernel", "Value"]].rename(columns={"Value": "BW%"})
    sm_df  = df[df["Metric"] == "SM Compute Throughput (%)"][["Backend", "Kernel", "Value"]].rename(columns={"Value": "SM%"})
    merged = bw_df.merge(sm_df, on=["Backend", "Kernel"])

    if merged.empty:
        warn("Not enough data for roofline plot")
        return

    # Compute application-level weighted average per Backend
    app_records = []
    for backend, group in merged.groupby("Backend"):
        total_w = sum(WEIGHTS.get(k, 1.0) for k in group["Kernel"])
        if total_w > 0:
            app_bw = sum(row["BW%"] * WEIGHTS.get(row["Kernel"], 1.0) for _, row in group.iterrows()) / total_w
            app_sm = sum(row["SM%"] * WEIGHTS.get(row["Kernel"], 1.0) for _, row in group.iterrows()) / total_w
        else:
            app_bw = group["BW%"].mean()
            app_sm = group["SM%"].mean()
        app_records.append({"Backend": backend, "BW%": app_bw, "SM%": app_sm})

    app_df = pd.DataFrame(app_records)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Roofline ceiling lines
    ax.axhline(100, color="crimson", linewidth=1.5, linestyle="--", label="Compute ceiling (100% SM Peak)")
    ax.axvline(100, color="steelblue", linewidth=1.5, linestyle="--", label="Bandwidth ceiling (100% HBM Peak)")
    import numpy as np
    x_ridge = np.linspace(0, 100, 200)
    ax.plot(x_ridge, np.minimum(x_ridge, 100), color="gray", linewidth=1.0,
            linestyle=":", label="Ridge line (BW-bound region)")

    # Distinct styles for Backends
    BACKEND_STYLES = {
        "CUDA 1W":        {"color": "#E74C3C", "marker": "o", "label": "CUDA (1 Worker)"},
        "CUDA 2W":        {"color": "#C0392B", "marker": "s", "label": "CUDA (2 Workers)"},
        "Alpaka CUDA 1W": {"color": "#3498DB", "marker": "^", "label": "Alpaka CUDA (1 Worker)"},
        "Alpaka CUDA 2W": {"color": "#2980B9", "marker": "D", "label": "Alpaka CUDA (2 Workers)"},
    }

    # Plot Application In-Toto data points
    for _, row in app_df.iterrows():
        b = row["Backend"]
        style = BACKEND_STYLES.get(b, {"color": "black", "marker": "o", "label": b})
        ax.scatter(row["BW%"], row["SM%"],
                   color=style["color"], marker=style["marker"], s=140, zorder=5,
                   edgecolors="black", linewidths=1.0, label=style["label"])
        ax.annotate(f" {style['label']}\n ({row['BW%']:.1f}% BW, {row['SM%']:.1f}% SM)",
                    (row["BW%"] + 1.0, row["SM%"] + 0.5),
                    fontsize=8.5, fontweight="bold", zorder=6)

    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Memory Bandwidth Utilization (% of 2039 GB/s A100 Peak)", labelpad=8)
    ax.set_ylabel("SM Compute Throughput (% of 312 TFLOPS A100 Peak)", labelpad=8)
    ax.set_title("Application Roofline Model — muDock in toto (CUDA vs Alpaka CUDA)", fontweight="bold", pad=12)
    ax.legend(title="Application Backend", loc="upper left", frameon=True, fontsize=8.5)
    sns.despine(ax=ax)
    plt.tight_layout()

    pdf_path = "ncu_roofline.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    ok(f"Application Roofline → {pdf_path}")


# ─── KERNEL-LEVEL METRIC BAR CHARTS ──────────────────────────────────────────
def plot_metric_bars(df: pd.DataFrame) -> None:
    section("Generating kernel-level metric bar charts…")

    df = df.copy()
    df["Config"] = df["Total Evaluations"].apply(
        lambda x: "Small\n(2k)" if x < 5000 else ("Medium\n(10k)" if x < 50000 else "Big\n(40k)")
    )
    config_order = ["Small\n(2k)", "Medium\n(10k)", "Big\n(40k)"]

    metrics = df["Metric"].unique()
    kernels = df["Kernel"].unique()

    for metric in metrics:
        for kernel in kernels:
            sub = df[(df["Metric"] == metric) & (df["Kernel"] == kernel)]
            if sub.empty or sub["Value"].dropna().empty or sub["Value"].sum() == 0:
                continue

            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            present = [o for o in config_order if o in sub["Config"].values]
            sns.barplot(
                data=sub, x="Config", y="Value", hue="Backend",
                ax=ax, order=present,
                errorbar=None,
                edgecolor="black", linewidth=0.7,
                palette=PALETTE,
            )

            # Value labels
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:.1f}",
                        (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8,
                    )

            k_title = kernel.replace('_', ' ').title()
            ax.set_title(f"NCU Profiling: {k_title} — {metric}", fontsize=11, fontweight="bold", pad=12)
            ax.set_ylabel(metric, labelpad=8)
            ax.set_xlabel("Workload Size (Population × Generations)", labelpad=8)

            if "%" in metric:
                ax.set_ylim(0, 115)
                ax.axhline(100, color="crimson", linewidth=0.8, linestyle="--", alpha=0.6)
            else:
                ax.set_ylim(bottom=0)
                ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
                
            ax.legend(title="Backend", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=8)
            sns.despine(ax=ax)
            plt.tight_layout()

            safe_metric = metric.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '').replace('/', 'per').lower()
            fname = f"ncu_{kernel}_{safe_metric}.pdf"
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            
    ok("Generated all NCU single-metric PDFs.")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main() -> None:
    banner("NCU RUNTIME HARDWARE PROFILER — CUDA vs Alpaka CUDA")

    all_records: list[dict] = []

    for name, bin_path, use_flag in BACKENDS:
        if not os.path.exists(bin_path):
            warn(f"Executable not found → {bin_path}  (skipping {name})")
            continue
        for pop, gen in CONFIGS:
            raw    = run_ncu(bin_path, name, use_flag, pop, gen)
            recs   = parse_ncu_csv(raw, name, pop, gen)
            all_records.extend(recs)

    if not all_records:
        warn("No data parsed. Did ncu run successfully?")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    # Average multiple kernel invocations of the same type within one run
    df = (df.groupby(["Backend", "Total Evaluations", "Kernel", "Metric"])
            .mean(numeric_only=True)
            .reset_index())

    csv_path = "ncu_metrics_configs.csv"
    df.to_csv(csv_path, index=False)
    ok(f"Results → {csv_path}")

    plot_roofline(df)
    plot_metric_bars(df)


if __name__ == "__main__":
    main()
