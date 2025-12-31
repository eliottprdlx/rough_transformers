#!/usr/bin/env python3
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm


# -----------------------------
# CONFIG "FINAL TABLE 2"
# -----------------------------

DATASETS = [
    "TSC_ArticularyWordRecognition",
    "TSC_HandMovementDirection",
    "TSC_UWaveGestureLibrary",
]

@dataclass(frozen=True)
class MethodSpec:
    name: str
    args: Dict[str, Any]
    flags: List[str]

METHODS = [
    MethodSpec("LSTM", {"model": "lstm"}, []),
    MethodSpec("Transformer", {"model": "transformer"}, []),

    MethodSpec(
        "RFormer-G",
        {"model": "transformer", "sig_level": 2, "num_windows": 100},
        ["use_signatures", "global_backward", "add_time"],
    ),
    MethodSpec(
        "RFormer-L",
        {"model": "transformer", "sig_level": 2, "num_windows": 100},
        ["use_signatures", "local_tight", "add_time"],
    ),
    MethodSpec(
        "RFormer-GL",
        {"model": "transformer", "sig_level": 2, "num_windows": 100},
        ["use_signatures", "global_backward", "local_tight", "add_time"],
    ),
]

SEEDS = [42]

DEFAULT_CONFIG = os.path.join("src", "UEA", "configs", "base.yaml")
MAIN_PY = os.path.join("src", "UEA", "main.py")
PYTHON_BIN = sys.executable

EXTRA_GLOBAL_ARGS = {
    "epoch": 150,        # mets 50 si tu as le temps
    "batch_size": 32,
    # optionnel: "lr": 0.0004,
}

FINAL_ACC_RE = re.compile(r"final_test_acc:\s*([0-9]*\.?[0-9]+)")


def parse_final_acc(stdout: str) -> Optional[float]:
    m = FINAL_ACC_RE.findall(stdout)
    return float(m[-1]) if m else None


def build_cmd(dataset: str, method: MethodSpec, seed: int) -> List[str]:
    cmd = [
        PYTHON_BIN, MAIN_PY,
        "--config", DEFAULT_CONFIG,
        "--dataset", dataset,
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


def run_one(dataset: str, method: MethodSpec, seed: int, log_dir: str) -> float:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset}__{method.name}__seed{seed}.log")

    cmd = build_cmd(dataset, method, seed)

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    dt = time.time() - t0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-60:])
        raise RuntimeError(f"Crash: {log_path}\n--- last lines ---\n{tail}")

    acc = parse_final_acc(proc.stdout)
    if acc is None:
        raise RuntimeError(f"Accuracy not found: {log_path}")

    return acc, dt


def format_pm(mean: float, std: float, decimals: int = 2) -> str:
    mean *= 100.0
    std *= 100.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    out_dir = "results"
    log_dir = os.path.join(out_dir, "logs_table2_final")
    os.makedirs(out_dir, exist_ok=True)

    jobs = [(d, m, s) for d in DATASETS for m in METHODS for s in SEEDS]
    rows = []

    pbar = tqdm(jobs, desc="Table2 final runs", unit="run")
    for dataset, method, seed in pbar:
        pbar.set_postfix_str(f"{dataset} | {method.name} | seed={seed}")

        try:
            acc, dt = run_one(dataset, method, seed, log_dir)
        except Exception as e:
            print(f"\n[WARN]\n{e}\n")
            acc, dt = float("nan"), float("nan")

        rows.append({
            "dataset": dataset,
            "method": method.name,
            "seed": seed,
            "acc": acc,
            "time_s": dt,
        })

        # sauvegarde intermédiaire (anti-crash)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "table2_final_raw.csv"), index=False)

    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["dataset", "method"])["acc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["pretty"] = agg.apply(lambda r: format_pm(r["mean"], 0.0 if pd.isna(r["std"]) else r["std"]), axis=1)

    pivot = agg.pivot(index="dataset", columns="method", values="pretty")
    pivot = pivot.reindex(columns=[m.name for m in METHODS])

    csv_path = os.path.join(out_dir, "table2_final.csv")
    tex_path = os.path.join(out_dir, "table2_final.tex")
    pivot.to_csv(csv_path)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(pivot.to_latex(escape=False))

    print("\n=== TABLE 2 FINAL ===")
    print(pivot)
    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {tex_path}")


if __name__ == "__main__":
    main()