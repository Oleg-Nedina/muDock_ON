"""
profiler_host_metrics.py
Measures Peak RAM (RSS) and Host CPU utilization for all backends (CPU and GPU).
Sets OMP_NUM_THREADS=32 in the subprocess environment so Alpaka-OMP uses all cores.
Iterates over Small, Medium, and Big configurations, 3 runs each.
Produces: host_metrics_variance.csv + host_metrics_analysis_{dataset}.pdf
"""

import subprocess
import re
import csv
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASETS = {
    "cpu500": {
        "PROTEIN": "/work/onedina/muDock_ON/data/1fkb/1fkb_pocket.pdbqt",
        "LIGAND":  "/work/onedina/muDock_ON/data/cpu_500lig.adtmol2",
    },
}

# OMP_NUM_THREADS injected into every subprocess environment
OMP_THREADS = 32
ENV = {**os.environ, "OMP_NUM_THREADS": str(OMP_THREADS)}

BACKENDS = [
    ("CPP Serial",       "/work/onedina/muDock_ON/build/cpp/application/muDock",          "CPP:CPU:0"),
    ("CPP OMP 8W",       "/work/onedina/muDock_ON/build/cpp-omp/application/muDock",       "CPP:CPU:0-7"),
    ("CPP OMP 32W",      "/work/onedina/muDock_ON/build/cpp-omp/application/muDock",       "CPP:CPU:0-31"),
    ("Alpaka Serial 1W", "/work/onedina/muDock_ON/build/alpaka-serial/application/muDock", "ALPAKA:CPU:0"),
    ("Alpaka Serial 8W", "/work/onedina/muDock_ON/build/alpaka-serial/application/muDock", "ALPAKA:CPU:0-7"),
    ("Alpaka OMP 1W",    "/work/onedina/muDock_ON/build/alpaka-omp/application/muDock",    "ALPAKA:CPU:0"),
    ("Alpaka OMP 8W",    "/work/onedina/muDock_ON/build/alpaka-omp/application/muDock",    "ALPAKA:CPU:0-7"),
    ("Alpaka OMP 32W",   "/work/onedina/muDock_ON/build/alpaka-omp/application/muDock",    "ALPAKA:CPU:0-31"),
    ("CUDA 1W",          "/work/onedina/muDock_ON/build/cuda/application/muDock",          "CUDA:GPU:0"),
    ("CUDA 2W",          "/work/onedina/muDock_ON/build/cuda/application/muDock",          "CUDA:GPU:0:2:4000000000"),
    ("Alpaka CUDA 1W",   "/work/onedina/muDock_ON/build/alpaka-cuda/application/muDock",   "ALPAKA:GPU:0"),
    ("Alpaka CUDA 2W",   "/work/onedina/muDock_ON/build/alpaka-cuda/application/muDock",   "ALPAKA:GPU:0:2:4000000000"),
]

CONFIGS = [
    (5, 10),   # 500 ligands × 5 pop × 10 gen — minimal, fast on CPU
]

WARMUP_RUNS = 1
RUNS        = 3
TIMEOUT_SEC = None

# ─── PLOT STYLE ──────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "xtick.labelsize":       8,
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

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def banner(title: str, width: int = 66) -> None:
    print(f"\n╔{'═' * (width - 2)}╗")
    print(f"║  {title:<{width - 4}}║")
    print(f"╚{'═' * (width - 2)}╝")

def section(msg: str) -> None: print(f"\n  ▶  {msg}")
def ok(msg: str)      -> None: print(f"  ✔  {msg}")
def warn(msg: str)    -> None: print(f"  ⚠  {msg}", file=sys.stderr)


def extract_host_metrics(time_output: str):
    """Parse /usr/bin/time -v stderr output for RSS, CPU%, and Context Switches."""
    rss = cpu = None
    ctx_switches = 0
    for line in time_output.splitlines():
        if "Maximum resident set size (kbytes):" in line:
            m = re.search(r"(\d+)", line)
            if m:
                rss = int(m.group(1)) / 1024.0   # → MB
        elif "Percent of CPU this job got:" in line:
            m = re.search(r"(\d+)%", line)
            if m:
                cpu = float(m.group(1))
        elif "Voluntary context switches:" in line or "Involuntary context switches:" in line:
            m = re.search(r"(\d+)", line)
            if m:
                ctx_switches += int(m.group(1))
    return rss, cpu, ctx_switches


# ─── PROFILING ───────────────────────────────────────────────────────────────
def run_host_profiling() -> list[dict]:
    banner(f"HOST METRICS PROFILER  —  OMP_NUM_THREADS={OMP_THREADS}  |  {WARMUP_RUNS} Warmup + {RUNS} Runs")
    results: list[dict] = []
    csv_path   = "host_metrics_variance.csv"
    fieldnames = ["Dataset", "Backend", "Total Evaluations", "Run",
                  "Peak RSS (MB)", "CPU Utilization (%)", "Context Switches"]

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for ds_name, paths in DATASETS.items():
        section(f"Dataset: {ds_name.upper()}")
        for name, bin_path, use_flag in BACKENDS:
            if not Path(bin_path).exists():
                warn(f"Executable not found → {bin_path}  (skipping {name})")
                continue

            for pop, gen in CONFIGS:
                cmd = [
                    "/usr/bin/time", "-v",
                    bin_path,
                    "--protein",     paths["PROTEIN"],
                    "--ligand",      paths["LIGAND"],
                    "--use",         use_flag,
                    "--population",  str(pop),
                    "--generations", str(gen),
                    "--search",      "genetic",
                    "--seed",        "42",
                ]

                # Warmup (discarded)
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=ENV, timeout=TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    print("  ⚠ Warmup Timeout!")
                    continue

                rss_runs, cpu_runs = [], []
                print(f"  {name:<24}  Pop:{pop:<4} Gen:{gen:<5} | ",
                      end="", flush=True)

                for r_idx in range(RUNS):
                    try:
                        res = subprocess.run(
                            cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=ENV, timeout=TIMEOUT_SEC
                        )
                        rss, cpu, ctx = extract_host_metrics(res.stderr)
                        if rss is not None and cpu is not None:
                            rss_runs.append(rss)
                            cpu_runs.append(cpu)
                            row = {
                                "Dataset":             ds_name,
                                "Backend":             name,
                                "Total Evaluations":   pop * gen,
                                "Run":                 r_idx + 1,
                                "Peak RSS (MB)":       rss,
                                "CPU Utilization (%)": cpu,
                                "Context Switches":    ctx,
                            }
                            results.append(row)
                            with open(csv_path, "a", newline="") as f:
                                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                            print(".", end="", flush=True)
                        else:
                            print("x", end="", flush=True)
                    except subprocess.TimeoutExpired:
                        print("T", end="", flush=True)

                if rss_runs:
                    print(f"  RSS: {sum(rss_runs)/len(rss_runs):6.1f} MB"
                          f"   CPU: {sum(cpu_runs)/len(cpu_runs):5.1f}%"
                          f"   Ctx: {ctx}")
                else:
                    print()

    ok(f"Results → {csv_path}")
    return results


# ─── PLOTTING ────────────────────────────────────────────────────────────────
def _config_label(total_evals: int) -> str:
    if total_evals < 5000:
        return "Small\n(2k evals)"
    if total_evals < 50000:
        return "Medium\n(10k evals)"
    return "Big\n(40k evals)"


def plot_host_metrics(df: pd.DataFrame) -> None:
    section("Generating IEEE-style PDF plots…")

    df = df.copy()
    df["Config"] = df["Total Evaluations"].apply(_config_label)
    config_order = ["Small\n(2k evals)", "Medium\n(10k evals)", "Big\n(40k evals)"]

    n_backends = df["Backend"].nunique()
    palette    = sns.color_palette("deep", n_colors=n_backends)

    for ds_name in DATASETS:
        sub = df[df["Dataset"] == ds_name]
        if sub.empty:
            warn(f"No data for dataset '{ds_name}' — skipping plot")
            continue

        present_order = [o for o in config_order if o in sub["Config"].values]

        import plot_utils
        
        for metric, ylabel, unit, expl, gloss in zip(
            ["Peak RSS (MB)",       "CPU Utilization (%)", "Context Switches"],
            ["Peak RSS",            "CPU Utilization",     "Context Switches"],
            ["MB",                  "%",                   "count"],
            [
                "Measures the maximum amount of physical RAM consumed by the process.",
                "Measures the percentage of CPU time utilized (100% = 1 core fully loaded).",
                "Total context switches (voluntary + involuntary) indicating thread scheduling overhead."
            ],
            [
                {"Peak RSS": "Peak Resident Set Size, measuring actual RAM usage.", "Zero-Overhead": "CUDA and Alpaka show identical usage."},
                {"CPU Util": "High values mean multi-threading is working. Low values mean thread starvation."},
                {"Context Switch": "OS switching CPU out from a thread. High numbers indicate thread thrashing or pipeline stalling."}
            ]
        ):
            if metric not in sub.columns:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 6.0))
            sns.barplot(
                data=sub, x="Config", y=metric, hue="Backend",
                ax=ax, order=present_order,
                errorbar="sd",
                capsize=0.05, err_kws={"linewidth": 1.0},
                edgecolor="black", linewidth=0.7,
                palette=palette,
            )
            ax.set_title(f"Host Resource Usage: {ylabel} ({ds_name.upper()})", fontsize=11, fontweight="bold", pad=12)
            ax.set_xlabel("Workload Size (Population × Generations)", labelpad=8)
            ax.set_ylabel(f"{ylabel} ({unit})", labelpad=8)
            ax.set_ylim(bottom=0)
            
            # Value annotations on bars
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:.0f}",
                        (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, rotation=90,
                    )
            
            ax.legend(title="Backend", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=8)
            sns.despine(ax=ax)
            plt.tight_layout()
            
            plot_utils.add_glossary(fig, expl, gloss)
            
            fname = f"host_{ylabel.lower().replace(' ', '_')}_{ds_name}.pdf"
            fig.savefig(fname)
            plt.close(fig)
            ok(f"Plot → {fname}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_host_profiling()
    if results:
        plot_host_metrics(pd.DataFrame(results))
    else:
        warn("No data collected. Exiting.")
        sys.exit(1)
