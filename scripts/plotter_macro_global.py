"""
plotter_macro_global.py
Generates IEEE paper-quality plots from the CSV produced by profiler_macro_global.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import matplotlib.ticker as ticker

DATASETS = ["small", "mid", "big"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

def ok(msg: str) -> None:
    print(f"  ✔  Saved → {msg}")

def warn(msg: str) -> None:
    print(f"  ⚠  {msg}", file=sys.stderr)

def plot_metric(df: pd.DataFrame, ds_name: str, metric: str, filename_prefix: str) -> str:
    df = df.copy()
    
    def _cfg(row):
        base = row['Population'] * row['Generations']
        return f"Pop:{row['Population']} Gen:{row['Generations']}"
        
    df["Config"] = df.apply(_cfg, axis=1)
    order = ["Pop:20 Gen:100", "Pop:50 Gen:200", "Pop:100 Gen:400"]
    
    df = df[df["Config"].isin(df["Config"].unique())]
    current_order = [o for o in order if o in df["Config"].values]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    sns.barplot(
        data=df, x="Config", y=metric, hue="Backend",
        ax=ax, order=current_order, errorbar="sd", 
        capsize=0.05, err_kws={'linewidth': 1}, edgecolor="black", linewidth=0.8,
        palette="deep"
    )
    
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(f"GPU Performance Scaling — Dataset: {ds_name.upper()}", pad=15)
    ax.set_xlabel("Workload Size (Evaluations)")
    
    if "Throughput" in metric:
        ax.set_yscale("log")
        ax.set_ylabel("Throughput (Evaluations/s)")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    else:
        ax.set_ylabel("Latency (s)")
        
    sns.despine(top=True, right=True)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    
    import plot_utils
    expl = f"Compares {metric} across GPU backends."
    gloss = {
        "Evaluations": "Energy scoring cycles.",
        "Workload Size": "Total evaluation count across population and generations.",
        "Alpaka CUDA": "C++ code abstracted through Alpaka running on GPU."
    }
    plot_utils.add_glossary(fig, expl, gloss)
    
    plt.tight_layout()
    out = f"macro_global_{filename_prefix}_{ds_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out

def plot_speedup_bar(df: pd.DataFrame, ds_name: str) -> str | None:
    mean_df = df.groupby(["Backend", "Population", "Generations", "Total Evaluations"])["Throughput (Evals/s)"].mean().reset_index()
    
    df_cuda = mean_df[mean_df["Backend"].str.contains("CUDA")].set_index(["Population", "Generations", "Total Evaluations"])
    df_alpaka = mean_df[mean_df["Backend"].str.contains("Alpaka")].set_index(["Population", "Generations", "Total Evaluations"])
    
    if df_cuda.empty or df_alpaka.empty: return None
        
    common = df_cuda.index.intersection(df_alpaka.index)
    if common.empty: return None
        
    speedup = (df_alpaka.loc[common, "Throughput (Evals/s)"] / df_cuda.loc[common, "Throughput (Evals/s)"]).reset_index()
    speedup.rename(columns={"Throughput (Evals/s)": "Speedup"}, inplace=True)
    
    def _cfg(row):
        base = row['Population'] * row['Generations']
        return f"Pop:{row['Population']} Gen:{row['Generations']}"
        
    speedup["Config"] = speedup.apply(_cfg, axis=1)
    order = ["Pop:20 Gen:100", "Pop:50 Gen:200", "Pop:100 Gen:400"]
    current_order = [o for o in order if o in speedup["Config"].values]
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.barplot(
        data=speedup, x="Config", y="Speedup", 
        order=current_order, ax=ax, palette="Blues_d",
        edgecolor="black", linewidth=0.8
    )
    
    ax.axhline(1.0, color="darkred", linestyle="--", linewidth=1.5, label="Parity (1.0x)")
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_title(f"Speedup: Alpaka vs CUDA — Dataset: {ds_name.upper()}", pad=15)
    ax.set_ylabel("Speedup (Alpaka / CUDA)")
    ax.set_xlabel("Workload Size (Evaluations)")
    
    sns.despine(top=True, right=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}x", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), textcoords='offset points', fontsize=9)
                    
    import plot_utils
    expl = "Demonstrates the relative performance of the Alpaka CUDA backend compared to Native CUDA."
    gloss = {
        "Speedup": "Ratio of Alpaka throughput over CUDA throughput. >1.0 means Alpaka is faster.",
        "Parity (1.0x)": "The baseline where both implementations perform identically."
    }
    plot_utils.add_glossary(fig, expl, gloss)
    
    plt.tight_layout()
    out = f"macro_global_speedup_{ds_name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out

def generate_macro_plots() -> None:
    for ds_name in DATASETS:
        csv_file = f"macro_results_global_{ds_name}.csv"
        if not Path(csv_file).exists():
            warn(f"File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        if df.empty:
            continue

        print(f"\n  ▶  Dataset: {ds_name.upper()} ({len(df)} rows)")
        ok(plot_metric(df, ds_name, "Time (s)", "latency"))
        ok(plot_metric(df, ds_name, "Throughput (Evals/s)", "throughput"))
        sp_out = plot_speedup_bar(df, ds_name)
        if sp_out: ok(sp_out)

if __name__ == "__main__":
    generate_macro_plots()
