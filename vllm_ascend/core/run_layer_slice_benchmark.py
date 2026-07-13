#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer-slice benchmark runner.

Modify layer_slice_config.yaml with different slice values and run
vllm-perf-test.py iteratively.

Usage:
    python run_layer_slice_benchmark.py

Configuration:
    Edit the CONFIG section below before running.
"""

import subprocess
import shutil
import re
import sys
from pathlib import Path

# =============================================================================
# CONFIG  (edit this section before running)
# =============================================================================

# Path to layer_slice_config.yaml
CONFIG_YAML_PATH = Path(r"vllm-ascend\vllm_ascend\core\layer_slice_config.yaml")

# Path to the benchmark script
BENCHMARK_SCRIPT = Path(r"path\to\vllm-perf-test.py")

# Python interpreter to run the benchmark (None = use sys.executable)
PYTHON_EXE = None  # e.g. Path(r"C:\Python312\python.exe")

# Slice values for each length key.
# All lists must have the same length. In round i, every key is set to its
# i-th value simultaneously, then the benchmark is executed.
#
# Example (all lengths同步变化):
#   SLICE_CONFIGS = {
#       0:  [1, 2, 3, 4, 5],
#       1:  [1, 2, 3, 4, 5],
#       2:  [1, 2, 3, 4, 5],
#       3:  [1, 2, 3, 4, 5],
#       4:  [1, 2, 3, 4, 5],
#       8:  [1, 2, 3, 4, 5],
#       16: [1, 2, 3, 4, 5],
#   }
#
# Example (only length-0 changes, others fixed at 5):
#   SLICE_CONFIGS = {
#       0: [1, 2, 3, 4, 5],
#   }
#   # Keys not listed here keep their original values.

SLICE_CONFIGS = {
    0:  [1, 2, 3, 4, 5],
    1:  [1, 2, 3, 4, 5],
    2:  [1, 2, 3, 4, 5],
    3:  [1, 2, 3, 4, 5],
    4:  [1, 2, 3, 4, 5],
    8:  [1, 2, 3, 4, 5],
    16: [1, 2, 3, 4, 5],
}

# Whether to restore the original yaml after all rounds finish
RESTORE_ORIGINAL_AFTER_DONE = True

# Whether to stop the whole run if one benchmark fails
STOP_ON_ERROR = True

# =============================================================================
# END OF CONFIG
# =============================================================================


def validate_configs(configs: dict) -> int:
    """Check that all value lists have the same length."""
    if not configs:
        raise ValueError("SLICE_CONFIGS is empty.")

    lengths = {len(v) for v in configs.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"All value lists in SLICE_CONFIGS must have the same length, "
            f"but got lengths: {lengths}"
        )
    return next(iter(lengths))


def modify_yaml(yaml_path: Path, round_values: dict) -> None:
    """
    Read the yaml, replace the slice count for each key found in
    round_values, and write it back.
    """
    text = yaml_path.read_text(encoding="utf-8")
    new_lines = []

    # Regex: capture leading spaces, the key, colon, value, trailing comment/spaces
    pattern = re.compile(r"^(\s*)(\d+)(\s*:\s*)(\d+)(\s*)$")

    for line in text.splitlines(keepends=True):
        # Remove trailing newline for processing
        bare_line = line.rstrip("\n").rstrip("\r")
        m = pattern.match(bare_line)
        if m:
            key = int(m.group(2))
            if key in round_values:
                # Rebuild the line preserving indentation and comment spacing
                new_line = f"{m.group(1)}{key}{m.group(3)}{round_values[key]}{m.group(5)}\n"
                new_lines.append(new_line)
                continue
        new_lines.append(line)

    yaml_path.write_text("".join(new_lines), encoding="utf-8")


def run_benchmark(bench_script: Path, python_exe: Path) -> int:
    """Execute the benchmark script and stream stdout/stderr in real time."""
    cmd = [str(python_exe), str(bench_script)]
    print(f"[Runner] Executing: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc.wait()


def main() -> int:
    yaml_path = CONFIG_YAML_PATH.resolve()
    bench_script = BENCHMARK_SCRIPT.resolve()
    python_exe = Path(PYTHON_EXE) if PYTHON_EXE else Path(sys.executable)

    # Sanity checks
    if not yaml_path.exists():
        print(f"[Error] Config yaml not found: {yaml_path}", file=sys.stderr)
        return 1
    if not bench_script.exists():
        print(f"[Error] Benchmark script not found: {bench_script}", file=sys.stderr)
        return 1

    num_rounds = validate_configs(SLICE_CONFIGS)
    keys = sorted(SLICE_CONFIGS.keys())

    # Backup original config
    backup_path = yaml_path.with_suffix(".yaml.bak")
    shutil.copy2(yaml_path, backup_path)
    print(f"[Runner] Original config backed up to: {backup_path}")

    try:
        for i in range(num_rounds):
            # Build the value map for this round
            round_values = {k: SLICE_CONFIGS[k][i] for k in keys}

            print("\n" + "=" * 70)
            print(f"[Runner] Round {i + 1} / {num_rounds}")
            print(f"[Runner] Updating {yaml_path.name} with values:")
            for k in keys:
                print(f"         {k}: {round_values[k]}")
            print("=" * 70 + "\n")

            # Modify yaml
            modify_yaml(yaml_path, round_values)

            # Run benchmark
            ret = run_benchmark(bench_script, python_exe)
            if ret != 0:
                print(f"\n[Error] Benchmark exited with code {ret}", file=sys.stderr)
                if STOP_ON_ERROR:
                    return ret

        print("\n[Runner] All rounds finished successfully.")
    finally:
        if RESTORE_ORIGINAL_AFTER_DONE:
            shutil.copy2(backup_path, yaml_path)
            print(f"[Runner] Original config restored: {yaml_path}")
        else:
            print(f"[Runner] Keeping last modified config: {yaml_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
