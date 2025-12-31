#!/usr/bin/env python3
"""
make_table5.py — FINAL (poster-ready)

Table 5: robustness to missing data with RANDOM DROP (50%) — resampled each epoch (handled in main.py)

- 3 UEA datasets
- 5 methods
- 3 seeds
- outputs:
  - results/table5_raw.csv       (one line per run)
  - results/table5.csv           (pretty mean±std)
  - results/table5.tex           (LaTeX)
  - results/logs_table5/*.log    (stdout logs per run)
"""

import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# -----------------------------
# 1) FINAL benchmark spec
# -----------------------------

DATASETS = [
    "TSC_ArticularyWordRecognition",
    "TSC_HandMovementDirection",
    "TSC_UWaveGestureLibrary",
]

@dataclass(frozen=True)
class MethodSpec:
    name: str
    args: Dict[str, Any]      # non-bool args only
    flags: List[str]          # store_true flags only


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

# Poster-ready: 3 seeds = std stable, still manageable
SEEDS = [42, 43, 44]

DEFAULT_CONFIG = os.path.join("src", "UEA", "configs", "base.yaml")
MAIN_PY = os.path.join("src", "UEA", "main.py")
PYTHON_BIN = sys.executable

OUT_DIR = "results"
LOG_DIR = os.path.join(OUT_DIR, "logs_table5")
RAW_CSV = os.path.join(OUT_DIR, "table5_raw.csv")
FINAL_CSV = os.path.join(OUT_DIR, "table5.csv")
FINAL_TEX = os.path.join(OUT_DIR, "table5.tex")

# Grosse config (2-3h CPU)
GLOBAL_ARGS = {
    "epoch": 60,
    "batch_size": 32,
    "n_seeds": 1,        # always one seed per subprocess
}

# Table 5 drop setup
DROP_PERCENT_KEEP = 0.5


# -----------------------------
# 2) Parse final_test_acc
# -----------------------------

FINAL_ACC_RE = re.compile(r"final_test_acc:\s*([0-9]*\.?[0-9]+)")

def parse_final_acc(stdout: str) -> Optional[float]:
    m = FINAL_ACC_RE.findall(stdout)
    return float(m[-1]) if m else None


# -----------------------------
# 3) Build command / run
# -----------------------------

def build_cmd(dataset: str, method: MethodSpec, seed: int) -> List[str]:
    cmd = [
        PYTHON_BIN, MAIN_PY,
        "--config", DEFAULT_CONFIG,
        "--dataset", dataset,
        "--seed_base", str(seed),
    ]

    # global args
    for k, v in GLOBAL_ARGS.items():
        cmd += [f"--{k}", str(v)]

    # drop (store_true + value)
    cmd += ["--use_random_drop"]
    cmd += ["--random_percentage", str(DROP_PERCENT_KEEP)]

    # method args
    for k, v in method.args.items():
        cmd += [f"--{k}", str(v)]

    # method flags
    for fl in method.flags:
        cmd += [f"--{fl}"]

    return cmd


def run_one(dataset: str, method: MethodSpec, seed: int) -> Tuple[float, float, str]:
    """
    returns (acc, elapsed_seconds, log_path)
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{dataset}__{method.name}__seed{seed}.log")

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
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise RuntimeError(f"Crash: {log_path}\n--- last lines ---\n{tail}")

    acc = parse_final_acc(proc.stdout)
    if acc is None:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise RuntimeError(f"Accuracy not found: {log_path}\n--- last lines ---\n{tail}")

    return acc, dt, log_path


# -----------------------------
# 4) Resume / aggregation
# -----------------------------

def load_done_keys() -> set:
    """
    Returns set of (dataset, method, seed) already computed in RAW_CSV.
    """
    if not os.path.exists(RAW_CSV):
        return set()
    df = pd.read_csv(RAW_CSV)
    keys = set(zip(df["dataset"], df["method"], df["seed"]))
    return keys


def save_raw(rows: List[dict]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW_CSV, index=False)


def format_pm(mean: float, std: float, decimals: int = 2) -> str:
    mean *= 100.0
    std *= 100.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_final_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        raw_df.groupby(["dataset", "method"])["acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # std can be NaN if count==1
    agg["std"] = agg["std"].fillna(0.0)
    agg["pretty"] = agg.apply(lambda r: format_pm(r["mean"], r["std"], decimals=2), axis=1)

    pivot = agg.pivot(index="dataset", columns="method", values="pretty")
    pivot = pivot.reindex(columns=[m.name for m in METHODS])

    return pivot


def save_final(pivot: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    pivot.to_csv(FINAL_CSV)
    with open(FINAL_TEX, "w", encoding="utf-8") as f:
        f.write(pivot.to_latex(escape=False))

    print(f"\n✅ Saved raw   -> {RAW_CSV}")
    print(f"✅ Saved table -> {FINAL_CSV}")
    print(f"✅ Saved latex -> {FINAL_TEX}")
    print(f"✅ Logs        -> {LOG_DIR}\n")


# -----------------------------
# 5) Main
# -----------------------------

def main():
    if not os.path.exists(MAIN_PY):
        raise FileNotFoundError(f"Cannot find {MAIN_PY}")
    if not os.path.exists(DEFAULT_CONFIG):
        raise FileNotFoundError(f"Cannot find {DEFAULT_CONFIG}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load existing raw results if any (resume feature)
    rows: List[dict] = []
    if os.path.exists(RAW_CSV):
        rows = pd.read_csv(RAW_CSV).to_dict("records")

    done = load_done_keys()

    jobs = [(d, m, s) for d in DATASETS for m in METHODS for s in SEEDS]
    todo = [(d, m, s) for (d, m, s) in jobs if (d, m.name, s) not in done]

    pbar = tqdm(todo, desc="Table 5 FINAL (drop=50%)", unit="run")
    for dataset, method, seed in pbar:
        pbar.set_postfix_str(f"{dataset} | {method.name} | seed={seed}")

        try:
            acc, dt, log_path = run_one(dataset, method, seed)
        except Exception as e:
            print(f"\n[WARN]\n{e}\n")
            acc, dt, log_path = float("nan"), float("nan"), ""

        rows.append({
            "dataset": dataset,
            "method": method.name,
            "seed": seed,
            "acc": acc,
            "time_s": dt,
            "drop_keep": DROP_PERCENT_KEEP,
            "epoch": GLOBAL_ARGS["epoch"],
            "batch_size": GLOBAL_ARGS["batch_size"],
            "log_path": log_path,
        })

        # Always checkpoint progress
        save_raw(rows)

        # Optional: print interim table every few runs
        if len(rows) % 5 == 0:
            raw_df = pd.DataFrame(rows)
            pivot = build_final_table(raw_df.dropna(subset=["acc"]))
            print("\n[INTERIM TABLE]\n", pivot, "\n")

    # Final table
    raw_df = pd.DataFrame(rows)
    pivot = build_final_table(raw_df.dropna(subset=["acc"]))
    save_final(pivot)

    print("=== TABLE 5 (FINAL) ===")
    print(pivot.to_string())


if __name__ == "__main__":
    main()