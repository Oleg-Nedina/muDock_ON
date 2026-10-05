import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_macro_results(csv_filename, output_filename, title):
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    if not os.path.exists(csv_path):
        print(f"[-] {csv_filename} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[-] {csv_filename} is empty.")
        return

    # Average the 3 runs for plotting
    df_avg = df.groupby(['Backend', 'Population', 'Generations'])['Throughput (Evals/s)'].mean().reset_index()

    sns.set_theme(style="whitegrid")
    
    # Create a grid of subplots: one for each Population
    populations = sorted(df_avg['Population'].unique())
    num_pops = len(populations)
    
    fig, axes = plt.subplots(1, num_pops, figsize=(5 * num_pops, 6), sharey=True)
    
    # If there's only one population, axes is not an array, make it one
    if num_pops == 1:
        axes = [axes]

    for i, pop in enumerate(populations):
        subset = df_avg[df_avg['Population'] == pop]
        sns.barplot(
            data=subset, 
            x="Generations", 
            y="Throughput (Evals/s)", 
            hue="Backend", 
            ax=axes[i], 
            palette="Set2"
        )
        axes[i].set_title(f"Population: {pop}")
        if i == 0:
            axes[i].set_ylabel("Throughput (Evals/s)")
        else:
            axes[i].set_ylabel("")
            
        # Add values on top of bars
        for p in axes[i].patches:
            height = p.get_height()
            if height > 0:
                axes[i].annotate(f'{int(height)}', 
                                 (p.get_x() + p.get_width() / 2., height), 
                                 ha='center', va='bottom', 
                                 fontsize=9, color='black', xytext=(0, 2), 
                                 textcoords='offset points')

    # Unified external legend
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=len(labels), frameon=True, fontsize=11)

    plt.suptitle(title, fontsize=16, y=1.12)
    plt.tight_layout()
    
    out_path = os.path.join(RESULTS_DIR, output_filename)
    plt.savefig(out_path, format="pdf", bbox_inches='tight')
    print(f"[+] Saved {output_filename}")
    plt.close()

if __name__ == "__main__":
    print("Generating Final Macro Benchmark Plots...")
    for ds in ["single", "small_multi"]:
        ds_title = ds.replace("_", " ").title()
        plot_macro_results(
            f"macro_results_global_{ds}.csv", 
            f"macro_plot_global_{ds}.pdf", 
            f"Global Macro Throughput (Dataset: {ds_title})"
        )
