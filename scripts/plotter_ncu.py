"""
plotter_ncu.py
Generates the Roofline plot (ncu_roofline.pdf) and metric bar charts (ncu_analysis_configs.pdf)
from the existing ncu_metrics_configs.csv file.
"""

import sys
import pandas as pd
from pathlib import Path
from profiler_ncu import plot_roofline, plot_metric_bars

def main():
    csv_path = Path("ncu_metrics_configs.csv")
    if not csv_path.exists():
        print(f"[-] File non trovato: {csv_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Caricamento dati da {csv_path.name}...")
    df = pd.read_csv(csv_path)
    plot_roofline(df)
    plot_metric_bars(df)
    print("[+] Plot generati con successo!")

if __name__ == "__main__":
    main()
