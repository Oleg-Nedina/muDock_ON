"""
plotter_macro_cpu_no_omp.py
Plots separating Serial CPU backends:
1. Serial Family: CPP Serial vs Alpaka Serial (1W, 8W)
Produces separate, clear plots for Throughput and Latency with clean layout.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASETS = ["cpu500", "small", "mid", "big"]

SERIAL_BACKENDS = [
    "CPP Serial",
    "Alpaka Serial (1 Worker)",
    "Alpaka Serial (8 Workers)",
]

# ─── PLOT STYLE ──────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.titleweight":      "bold",
    "axes.labelsize":        10,
    "axes.labelweight":      "bold",
    "xtick.labelsize":       8.5,
    "ytick.labelsize":       9,
    "legend.fontsize":       8,
    "legend.title_fontsize": 8.5,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "axes.grid":             True,
    "grid.alpha":            0.4,
    "grid.linestyle":        "--",
    "axes.axisbelow":        True,
})

def ok(msg: str)   -> None: print(f"  ✔  Saved → {msg}")
def warn(msg: str) -> None: print(f"  ⚠  {msg}", file=sys.stderr)

def _config_label(row: pd.Series) -> str:
    base_evals = row["Population"] * row["Generations"]
    if base_evals <= 100:
        return f"Workload ({base_evals} evals/lig)\nPop {row['Population']}, Gen {row['Generations']}"
    if base_evals < 10000:
        return "Small Workload\n(2k evals: Pop 20, Gen 100)"
    if base_evals < 100000:
        return "Medium Workload\n(10k evals: Pop 50, Gen 200)"
    return "Big Workload\n(40k evals: Pop 100, Gen 400)"

def _annotate_bars(ax, fmt="{:.1f}", log=False):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            y_pos = h * 1.12 if log else h + (ax.get_ylim()[1] * 0.02)
            val_str = f"{h:,.0f}" if h >= 100 else fmt.format(h)
            ax.annotate(val_str, (p.get_x() + p.get_width() / 2, y_pos),
                        ha="center", va="bottom", fontsize=7.5, rotation=90)


# ─── SERIAL PLOTS ────────────────────────────────────────────────────────────
def plot_serial_group(df: pd.DataFrame, ds_name: str) -> None:
    df_s = df[df["Backend"].isin(SERIAL_BACKENDS)].copy()
    if df_s.empty:
        return

    present = [b for b in SERIAL_BACKENDS if b in df_s["Backend"].unique()]
    palette = sns.color_palette("muted", n_colors=len(present))

    df_s["Config"] = df_s.apply(_config_label, axis=1)
    cfg_order = list(dict.fromkeys(df_s["Config"]))

    out_dir = Path("scripts/plot_no_omp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Throughput Plot (Serial)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(data=df_s, x="Config", y="Throughput (Evals/s)", hue="Backend",
                ax=ax, order=cfg_order, hue_order=present, errorbar="sd",
                capsize=0.04, err_kws={"linewidth": 0.8}, edgecolor="black", linewidth=0.7,
                palette=palette)
    ax.set_title(f"CPU Serial Execution — Throughput Comparison ({ds_name.upper()} Dataset)", pad=12)
    ax.set_xlabel("Workload Configuration", labelpad=8)
    ax.set_ylabel("Throughput (Evaluations / second)", labelpad=8)
    _annotate_bars(ax, fmt="{:,.1f}")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    ax.legend(title="Serial Backend", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    sns.despine(ax=ax)

    plt.tight_layout()
    out_tp = out_dir / f"cpu_serial_throughput_{ds_name}.pdf"
    fig.savefig(out_tp, bbox_inches="tight")
    plt.close(fig)
    ok(out_tp)

    # 2. Latency Plot (Serial)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(data=df_s, x="Config", y="Time (s)", hue="Backend",
                ax=ax, order=cfg_order, hue_order=present, errorbar="sd",
                capsize=0.04, err_kws={"linewidth": 0.8}, edgecolor="black", linewidth=0.7,
                palette=palette)
    ax.set_title(f"CPU Serial Execution — Wall-Clock Latency ({ds_name.upper()} Dataset)", pad=12)
    ax.set_xlabel("Workload Configuration", labelpad=8)
    ax.set_ylabel("Total Latency (seconds)", labelpad=8)
    _annotate_bars(ax, fmt="{:.2f}s")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.legend(title="Serial Backend", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    sns.despine(ax=ax)

    plt.tight_layout()
    out_lat = out_dir / f"cpu_serial_latency_{ds_name}.pdf"
    fig.savefig(out_lat, bbox_inches="tight")
    plt.close(fig)
    ok(out_lat)

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  CPU PLOTTER  —  Serial Analysis                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    for ds_name in DATASETS:
        csv_file = f"scripts/macro_results_cpu_{ds_name}.csv"
        if not Path(csv_file).exists():
            continue

        df = pd.read_csv(csv_file)
        if df.empty:
            continue

        if "Total Evaluations" not in df.columns:
            df["Total Evaluations"] = df["Population"] * df["Generations"]

        print(f"\n▶ Dataset: {ds_name.upper()}")
        print("  --- 1. Serial Analysis (CPP Serial vs Alpaka Serial) ---")
        plot_serial_group(df, ds_name)

if __name__ == "__main__":
    main()
