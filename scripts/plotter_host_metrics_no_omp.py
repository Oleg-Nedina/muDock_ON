"""
plotter_host_metrics_no_omp.py
Generates Peak RSS and CPU utilization plots for serial backends only.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Aggiungiamo la cartella scripts al sys.path per importare le dipendenze
sys.path.append(os.path.abspath("scripts"))
from profiler_host_metrics import plot_host_metrics

def main():
    csv_path = Path("scripts/host_metrics_variance.csv")
    if not csv_path.exists():
        print(f"[-] File non trovato: {csv_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Caricamento dati da {csv_path.name}...")
    df = pd.read_csv(csv_path)

    # Filtriamo solo i backend seriali (senza OMP e senza GPU)
    serial_backends = [
        "CPP Serial",
        "Alpaka Serial 1W",
        "Alpaka Serial 8W"
    ]
    df_filtered = df[df["Backend"].isin(serial_backends)].copy()

    if df_filtered.empty:
        print("[-] Nessun dato per backend seriali trovato.")
        sys.exit(1)

    # Cambiamo la directory di lavoro in modo che plot_host_metrics salvi i file qui
    os.makedirs("scripts/plot_no_omp", exist_ok=True)
    os.chdir("scripts/plot_no_omp")
    
    print("[*] Generazione dei grafici...")
    plot_host_metrics(df_filtered)
    print("[+] Plot generati con successo in scripts/plot_no_omp/")

if __name__ == "__main__":
    main()
