import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["single", "small_multi"]

for ds in DATASETS:
    csv_file = f"macro_results_cpu_{ds}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_file)
    if not os.path.exists(csv_path):
        print(f"[-] {csv_file} not found in {RESULTS_DIR}, skipping.")
        continue

    ds_title = ds.replace("_", " ").title()
    df = pd.read_csv(csv_path)
    df_avg = df.groupby(['Backend', 'Population', 'Generations'])['Throughput (Evals/s)'].mean().reset_index()

    sns.set_theme(style="whitegrid")

    # 1. HEATMAP: The Cost of Unrolling
    if 'Alpaka Serial (Unroll)' in df_avg['Backend'].values and 'Alpaka Serial (No Unroll)' in df_avg['Backend'].values:
        df_alp_unroll = df_avg[df_avg['Backend'] == 'Alpaka Serial (Unroll)'].set_index(['Population', 'Generations'])
        df_alp_no = df_avg[df_avg['Backend'] == 'Alpaka Serial (No Unroll)'].set_index(['Population', 'Generations'])
        speedup_unroll = (df_alp_no['Throughput (Evals/s)'] / df_alp_unroll['Throughput (Evals/s)']).reset_index()
        heatmap_data_unroll = speedup_unroll.pivot(index="Population", columns="Generations", values="Throughput (Evals/s)")

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data_unroll, annot=True, fmt=".2f", cmap="RdYlGn", cbar_kws={'label': 'Speedup Factor'})
        plt.title(f"The Cost of Unrolling on CPU (Dataset: {ds_title})\nSpeedup: Alpaka (No-Unroll) / Alpaka (Unroll)")
        plt.savefig(os.path.join(RESULTS_DIR, f"cpu_advanced_01_unroll_penalty_heatmap_{ds}.pdf"), bbox_inches='tight')
        plt.close()

    # 2. HEATMAP: The RNG Gap
    if 'CPP Serial' in df_avg['Backend'].values and 'Alpaka Serial (No Unroll)' in df_avg['Backend'].values:
        df_cpp = df_avg[df_avg['Backend'] == 'CPP Serial'].set_index(['Population', 'Generations'])
        speedup_rng = (df_alp_no['Throughput (Evals/s)'] / df_cpp['Throughput (Evals/s)']).reset_index()
        heatmap_data_rng = speedup_rng.pivot(index="Population", columns="Generations", values="Throughput (Evals/s)")

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data_rng, annot=True, fmt=".2f", cmap="Reds_r", cbar_kws={'label': 'Performance Ratio (1.0 = Parity)'})
        plt.title(f"The RNG Gap (Philox vs MT) (Dataset: {ds_title})\nRatio: Alpaka (No-Unroll) / CPP Serial")
        plt.savefig(os.path.join(RESULTS_DIR, f"cpu_advanced_02_rng_gap_heatmap_{ds}.pdf"), bbox_inches='tight')
        plt.close()

    # 3. OVERALL BAR PLOT: Average Throughput Collapse
    plt.figure(figsize=(10, 6))
    avg_global = df_avg.groupby('Backend')['Throughput (Evals/s)'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=avg_global, x='Throughput (Evals/s)', y='Backend', hue='Backend', legend=False, palette="viridis")
    plt.title(f"Global Average Throughput (Dataset: {ds_title})")
    for index, value in enumerate(avg_global['Throughput (Evals/s)']):
        plt.text(value, index, f" {int(value)} Evals/s", va='center')
    plt.xlim(0, avg_global['Throughput (Evals/s)'].max() * 1.15)
    plt.savefig(os.path.join(RESULTS_DIR, f"cpu_advanced_03_global_averages_{ds}.pdf"), bbox_inches='tight')
    plt.close()

    print(f"[+] Generated advanced analytical plots for {ds}.")
