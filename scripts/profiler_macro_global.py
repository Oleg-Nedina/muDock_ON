"""
profiler_macro_global.py
Full macro-profiling for GPU backends: 3 configs, 1 warmup + 10 runs.
Covers small, mid, big datasets.
Produces: macro_results_global_{dataset}.csv
"""

import subprocess
import re
import csv
from pathlib import Path
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASETS = {
    "small": {
        "PROTEIN": "/work/onedina/muDock_ON/data/1fkb/1fkb_pocket.pdbqt",
        "LIGAND": "/work/onedina/muDock_ON/data/small_1/small.adtmol2",
    },
    "mid": {
        "PROTEIN": "/work/onedina/muDock_ON/data/1fkb/1fkb_pocket.pdbqt",
        "LIGAND": "/work/onedina/muDock_ON/data/mid_1/mid.adtmol2",
    },
    "big": {
        "PROTEIN": "/work/onedina/muDock_ON/data/1fkb/1fkb_pocket.pdbqt",
        "LIGAND": "/work/onedina/muDock_ON/data/big_1/big.adtmol2",
    },
}

BACKENDS = [
    ("CUDA (1 Worker)", "/work/onedina/muDock_ON/build/cuda/application/muDock", "CUDA:GPU:0"),
    ("CUDA (2 Workers)", "/work/onedina/muDock_ON/build/cuda/application/muDock", "CUDA:GPU:0:2:4000000000"),
    ("Alpaka (1 Worker)", "/work/onedina/muDock_ON/build/alpaka-cuda/application/muDock", "ALPAKA:GPU:0"),
    ("Alpaka (2 Workers)", "/work/onedina/muDock_ON/build/alpaka-cuda/application/muDock", "ALPAKA:GPU:0:2:4000000000"),
]

CONFIGS = [
    (20, 100),    # Small: 2k evals
    (50, 200),    # Medium: 10k evals
    (100, 400),   # Big: 40k evals
]

WARMUP_RUNS = 1
RUNS = 3
TIMEOUT_SEC = None  # 20 minutes timeout per run

FIELDNAMES = [
    "Backend",
    "Population",
    "Generations",
    "Run",
    "Total Evaluations",
    "Time (s)",
    "Throughput (Evals/s)",
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def banner(title: str, width: int = 68) -> None:
    print(f"\n╔{'═' * (width - 2)}╗")
    print(f"║  {title:<{width - 4}}║")
    print(f"╚{'═' * (width - 2)}╝")

def section(msg: str) -> None:
    print(f"\n  ▶  {msg}")

def ok(msg: str) -> None:
    print(f"  ✔  {msg}")

def warn(msg: str) -> None:
    print(f"  ⚠  {msg}", file=sys.stderr)

def extract_total_time(text: str) -> float | None:
    m = re.search(r"\[\s*([\d\.]+)\s*\]\s*INFO All Done!", text)
    return float(m.group(1)) if m else None

def extract_num_ligands(text: str) -> int:
    m = re.search(r"Parsing\s+(\d+)\s+compound", text)
    return int(m.group(1)) if m else 1

def progress_bar(current: int, total: int, width: int = 30) -> str:
    if total == 0:
        return f"[{'█'*width}] 100.0%"
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100.0 * current / total
    return f"[{bar}] {pct:5.1f}%  ({current}/{total})"

# ─── MAIN ────────────────────────────────────────────────────────────────────
def run_macro_benchmark() -> None:
    total_configs = len(BACKENDS) * len(CONFIGS)
    total_tasks = total_configs * RUNS

    banner(
        f"MACRO GLOBAL PROFILING  —  muDock  |  "
        f"{len(CONFIGS)} configs  "
        f"|  {WARMUP_RUNS} Warmup + {RUNS} Runs"
    )

    for ds_name, paths in DATASETS.items():
        section(f"Dataset: {ds_name.upper()}")
        print(f"     Backends     : {', '.join(n for n, *_ in BACKENDS)}")
        print(f"     Configs      : {CONFIGS}")
        print(f"     Total runs   : {total_tasks} per backend")

        results: list[dict] = []
        csv_output = f"macro_results_global_{ds_name}.csv"
        task_done = 0

        with open(csv_output, mode="w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

        for name, bin_path, use_flag in BACKENDS:
            if not Path(bin_path).exists():
                warn(f"Executable not found → {bin_path}  (skipping {name})")
                task_done += len(CONFIGS) * RUNS
                continue

            print()
            for pop, gen in CONFIGS:
                cmd = [
                    bin_path,
                    "--protein",
                    paths["PROTEIN"],
                    "--ligand",
                    paths["LIGAND"],
                    "--use",
                    use_flag,
                    "--population",
                    str(pop),
                    "--generations",
                    str(gen),
                    "--search",
                    "genetic",
                    "--seed",
                    "42",
                ]

                # ── Warmup ──
                skip = False
                for _ in range(WARMUP_RUNS):
                    try:
                        subprocess.run(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=TIMEOUT_SEC,
                        )
                    except subprocess.TimeoutExpired:
                        warn(f"Warmup timed out → {name}  Pop:{pop}  Gen:{gen}  — config skipped")
                        skip = True
                        break

                if skip:
                    task_done += RUNS
                    continue

                # ── Measurements ──
                run_times: list[float] = []
                for r_idx in range(RUNS):
                    task_done += 1
                    pb = progress_bar(task_done, total_tasks)
                    print(
                        f"  {pb}  {name:<6}  Pop:{pop:<5}  Gen:{gen:<6}  Run:{r_idx + 1}/{RUNS}",
                        end="  ",
                        flush=True,
                    )
                    try:
                        res = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=TIMEOUT_SEC,
                        )
                        t = extract_total_time(res.stdout)
                        if t is None:
                            print("→ ⚠ time not found")
                            continue

                        num_ligands = extract_num_ligands(res.stdout)
                        total_evals = num_ligands * pop * gen
                        evals_s = total_evals / t
                        run_times.append(t)
                        print(f"→ {t:.3f}s  ({evals_s:,.0f} Evals/s [L:{num_ligands}])")

                        row = {
                            "Backend": name,
                            "Population": pop,
                            "Generations": gen,
                            "Run": r_idx + 1,
                            "Total Evaluations": total_evals,
                            "Time (s)": t,
                            "Throughput (Evals/s)": evals_s,
                        }
                        results.append(row)
                        with open(csv_output, mode="a", newline="") as f:
                            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                    except subprocess.TimeoutExpired:
                        print(f"→ ⚠ timed out (>{TIMEOUT_SEC}s)")
                    except subprocess.CalledProcessError as e:
                        print(f"→ ✗ error: {e}")

                if run_times:
                    avg_t = sum(run_times) / len(run_times)
                    avg_eval = total_evals / avg_t
                    print(f"         └─ avg  {avg_t:.3f}s   {avg_eval:,.0f} Evals/s")

        if results:
            ok(f"Global profiling done for '{ds_name}'.")
            ok(f"Results → {csv_output}  ({len(results)} rows)")
        else:
            warn(f"No valid results for dataset '{ds_name}'.")

if __name__ == "__main__":
    run_macro_benchmark()
