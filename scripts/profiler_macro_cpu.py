"""
profiler_macro_cpu.py
Macro-profiling for CPU backends: 1 configuration (Pop=20, Gen=100), 1 warmup + 3 runs.
Uses the 'small' multi-ligand dataset to stress the CPU TBB pipeline with real parallelism.
Produces: macro_results_cpu_small.csv
"""

import subprocess
import re
import csv
import sys
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASETS = {
    "cpu500": {
        "PROTEIN": "/work/onedina/muDock_ON/data/1fkb/1fkb_pocket.pdbqt",
        "LIGAND":  "/work/onedina/muDock_ON/data/cpu_500lig.adtmol2",
    },
}

BACKENDS = [
    ("CPP Serial",           "/work/onedina/muDock_ON/build/cpp/application/muDock",           "CPP:CPU:0"),
    ("CPP OMP (1 Worker)",   "/work/onedina/muDock_ON/build/cpp-omp/application/muDock",       "CPP:CPU:0"),
    ("CPP OMP (8 Workers)",  "/work/onedina/muDock_ON/build/cpp-omp/application/muDock",       "CPP:CPU:0-7"),
    ("Alpaka Serial (1 Worker)", "/work/onedina/muDock_ON/build/alpaka-serial/application/muDock", "ALPAKA:CPU:0"),
    ("Alpaka Serial (8 Workers)","/work/onedina/muDock_ON/build/alpaka-serial/application/muDock", "ALPAKA:CPU:0-7"),
    ("Alpaka OMP (1 Worker)",    "/work/onedina/muDock_ON/build/alpaka-omp/application/muDock",    "ALPAKA:CPU:0"),
    ("Alpaka OMP (8 Workers)",   "/work/onedina/muDock_ON/build/alpaka-omp/application/muDock",    "ALPAKA:CPU:0-7"),
]

CONFIGS = [
    (5, 10),    # 500 ligands × 5 pop × 10 gen = 25k evals — minimal, fast on CPU
]

WARMUP_RUNS = 1
RUNS        = 3
TIMEOUT_SEC = None

FIELDNAMES = [
    "Backend", "Population", "Generations", "Run",
    "Total Evaluations", "Time (s)", "Throughput (Evals/s)",
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
    total_tasks   = total_configs * RUNS

    banner(
        f"MACRO CPU PROFILING  —  muDock  |  "
        f"{len(CONFIGS)} config  "
        f"|  {WARMUP_RUNS} Warmup + {RUNS} Runs"
    )

    for ds_name, paths in DATASETS.items():
        section(f"Dataset: {ds_name.upper()}")
        print(f"     Backends     : {', '.join(n for n, *_ in BACKENDS)}")
        print(f"     Configs      : {CONFIGS}")
        print(f"     Total runs   : {total_tasks} per backend")

        results: list[dict] = []
        csv_output = f"macro_results_cpu_{ds_name}.csv"
        task_done  = 0

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
                    "--protein",     paths["PROTEIN"],
                    "--ligand",      paths["LIGAND"],
                    "--use",         use_flag,
                    "--population",  str(pop),
                    "--generations", str(gen),
                    "--search",      "genetic",
                    "--seed",        "42",
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
                        f"  {pb}  {name:<30}  Pop:{pop:<5}  Gen:{gen:<6}  Run:{r_idx + 1}/{RUNS}",
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
                            "Backend":              name,
                            "Population":           pop,
                            "Generations":          gen,
                            "Run":                  r_idx + 1,
                            "Total Evaluations":    total_evals,
                            "Time (s)":             t,
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
                    avg_t    = sum(run_times) / len(run_times)
                    avg_eval = total_evals / avg_t
                    print(f"         └─ avg  {avg_t:.3f}s   {avg_eval:,.0f} Evals/s")

        if results:
            ok(f"Global profiling done for '{ds_name}'.")
            ok(f"Results → {csv_output}  ({len(results)} rows)")
        else:
            warn(f"No valid results for dataset '{ds_name}'.")

if __name__ == "__main__":
    run_macro_benchmark()
