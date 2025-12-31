#!/usr/bin/env python3
"""
make_fbm_table.py — FBM-specific benchmark

- Dataset: FractionalBrownianMotion
- 5 methods
- 1 seed
- Reduced cost (FBM-friendly)
"""

import os
import re
import sys
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm


# =============================
# Configuration FBM
# =============================

DATASET = "FractionalBrownianMotion"

@dataclass(frozen=True)
class MethodSpec:
    name: str
    args: Dict[str, Any]     # non-bool args only
    flags: List[str]         # store_true flags only


METHODS = [
    MethodSpec("LSTM", {"model": "lstm"}, []),
    MethodSpec("Transformer", {"model": "transformer"}, []),

    MethodSpec(
        "RFormer-G",
        {
            "model": "transformer",
            "sig_level": 2,
            "num_windows": 25,   # 🔑 FBM speed
        },
        ["use_signatures", "global_backward", "add_time"],
    ),

    MethodSpec(
        "RFormer-L",
        {
            "model": "transformer",
            "sig_level": 2,
            "num_windows": 25,
        },
        ["use_signatures", "local_tight", "add_time"],
    ),

    MethodSpec(
        "RFormer-GL",
        {
            "model": "transformer",
            "sig_level": 2,
            "num_windows": 25,
        },
        ["use_signatures", "global_backward", "local_tight", "add_time"],
    ),
]

SEEDS = [42]

DEFAULT_CONFIG = os.path.join("src", "UEA", "configs", "base.yaml")
MAIN_PY = os.path.join("src", "UEA", "main.py")
PYTHON_BIN = sys.executable

EXTRA_GLOBAL_ARGS = {
    "epoch": 15,
    "batch_size": 32,
}


# =============================
# Utils
# =============================

FINAL_ACC_RE = re.compile(r"final_test_acc:\s*([0-9]*\.?[0-9]+)")

def parse_final_acc(stdout: str) -> Optional[float]:
    m = FINAL_ACC_RE.findall(stdout)
    return float(m[-1]) if m else None


def build_cmd(method: MethodSpec, seed: int) -> List[str]:
    cmd = [
        PYTHON_BIN, MAIN_PY,
        "--config", DEFAULT_CONFIG,
        "--dataset", DATASET,
        "--n_seeds", "1",
        "--seed_base", str(seed),
    ]

    for k, v in EXTRA_GLOBAL_ARGS.items():
        cmd += [f"--{k}", str(v)]

    for k, v in method.args.items():
        cmd += [f"--{k}", str(v)]

    for fl in method.flags:
        cmd += [f"--{fl}"]

    return cmd


def run_one(method: MethodSpec, seed: int) -> float:
    log_dir = "results/logs_fbm"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"FBM__{method.name}__seed{seed}.log")
    cmd = build_cmd(method, seed)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(f"Crash: {log_path}\n{tail}")

    acc = parse_final_acc(proc.stdout)
    if acc is None:
        raise RuntimeError(f"Accuracy not found: {log_path}")

    return acc


# =============================
# Main
# =============================

def main():
    rows = []
    jobs = [(m, s) for m in METHODS for s in SEEDS]

    pbar = tqdm(jobs, desc="FBM benchmark", unit="run")
    for method, seed in pbar:
        pbar.set_postfix_str(method.name)

        try:
            acc = run_one(method, seed)
        except Exception as e:
            print(f"\n[WARN]\n{e}\n")
            acc = float("nan")

        rows.append({
            "dataset": DATASET,
            "method": method.name,
            "seed": seed,
            "acc": acc,
        })

    df = pd.DataFrame(rows)
    table = df.pivot(index="dataset", columns="method", values="acc")

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/fbm_table.csv")
    table.to_latex("results/fbm_table.tex", float_format="%.4f")

    print("\n=== FBM TABLE ===")
    print(table)


if __name__ == "__main__":
    main()