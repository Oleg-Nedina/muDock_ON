#!/bin/bash
# run_all.sh
# Runs all profilers sequentially. Designed to be run inside a tmux session.

PYTHON=/home/onedina/.conda/envs/mudock_profiling/bin/python
SCRIPTS_DIR=/work/onedina/muDock_ON/scripts

cd $SCRIPTS_DIR || { echo "Directory non trovata!"; exit 1; }

echo "=========================================================="
echo "  AVVIO DELLA SUITE DI BENCHMARK GLOBALE"
echo "  I log di esecuzione saranno stampati in questo terminale."
echo "=========================================================="
echo ""

run_suite() {
    local name="$1"
    local script="$2"
    echo "=========================================================="
    echo "  -> Esecuzione: $name"
    echo "=========================================================="
    if [ -f "$script" ]; then
        $PYTHON "$script"
    else
        echo "  ⚠  Script $script non trovato, salto..."
    fi
    echo ""
}

# Macro Profilers (CPU e GPU)
run_suite "CPU Macro Profiler" "profiler_macro_cpu.py"
run_suite "CPU Macro Plotter" "plotter_macro_cpu.py"

run_suite "GPU Macro Profiler" "profiler_macro_global.py"
run_suite "GPU Macro Plotter" "plotter_macro_global.py"

# Rapid Profilers (se attivi)
# run_suite "Rapid CPU Profiler" "profiler_macro_rapid_cpu.py"
# run_suite "Rapid GPU Profiler" "profiler_macro_rapid.py"

# Host Metrics (Peak RSS / CPU Utilization)
run_suite "Host Metrics Profiler" "profiler_host_metrics.py"
run_suite "Host Metrics Plotter" "plotter_host_metrics.py"

# Micro Profilers (Nsys)
run_suite "Micro Profiler - ADT Score" "profiler_micro_adt.py"
run_suite "Micro Profiler - Genetic Search" "profiler_micro_genetic.py"
run_suite "Micro Profiler - Geom Score" "profiler_micro_geom.py"

# NCU Profiler
run_suite "NCU Hardware Counters Profiler" "profiler_ncu.py"
run_suite "NCU Plotter" "plotter_ncu.py"

# PTXAS Profiler
run_suite "PTXAS Compiler Profiler" "profiler_ptxas.py"

echo "=========================================================="
echo "  TUTTE LE SUITE DI PROFILAZIONE COMPLETATE CON SUCCESSO!"
echo "=========================================================="

