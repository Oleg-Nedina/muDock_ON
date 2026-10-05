"""
plotter_host_metrics.py
Generates Peak RSS and CPU utilization plots (host_metrics_analysis_*.pdf)
from the existing host_metrics_variance.csv file.
"""

import sys
import pandas as pd
from pathlib import Path
from profiler_host_metrics import plot_host_metrics

def main():
    csv_path = Path("host_metrics_variance.csv")
    if not csv_path.exists():
        print(f"[-] File non trovato: {csv_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Caricamento dati da {csv_path.name}...")
    df = pd.read_csv(csv_path)
    plot_host_metrics(df)
    print("[+] Plot generati con successo!")

if __name__ == "__main__":
    main()
