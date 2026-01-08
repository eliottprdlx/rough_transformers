#!/usr/bin/env python3
"""
make_table5_fbm.py — ULTRA LIGHT

FBM is expensive, so this is a *small* complement to Table 5:
- Dataset: FractionalBrownianMotion
- Conditions: NoDrop vs RandomDrop(B = resampled each epoch; handled in main.py)
- Methods: Transformer vs RFormer-L
- Few epochs, few seeds
- Robust logging + resume

Outputs:
- results/table5_fbm_raw.csv
- results/table5_fbm.csv
- results/table5_fbm.tex
- results/logs_table5_fbm/*.log
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
# 1) Spec (ULTRA LIGHT)
# -----------------------------

DATASET = "FractionalBrownianMotion"

# Ultra-light: 1 seed by default. If manageable, set to [42, 43]
SEEDS = [42]

DEFAULT_CONFIG = os.path.join("src", "UEA", "configs", "base.yaml")
MAIN_PY = os.path.join("src", "UEA", "main.py")
PYTHON_BIN = sys.executable

OUT_DIR = "results"
LOG_DIR = os.path.join(OUT_DIR, "logs_table5_fbm")

RAW_CSV = os.path.join(OUT_DIR, "table5_fbm_raw.csv")
FINAL_CSV = os.path.join(OUT_DIR, "table5_fbm.csv")
FINAL_TEX = os.path.join(OUT_DIR, "table5_fbm.tex")

# Keep this small: FBM is slow
GLOBAL_ARGS = {
    "epoch": 200,        # ULTRA LIGHT
    "batch_size": 64,   # adjust if RAM allows
    "n_seeds": 1,       # always 1 seed per subprocess
}

DROP_KEEP = 0.3   # Table 5 missingness level


@dataclass(frozen=True)
class CondSpec:
    """NoDrop vs RandomDrop(B). For RandomDrop, main.py does resampling each epoch."""
    name: str
    args: Dict[str, Any]
    flags: List[str]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    args: Dict[str, Any]
    flags: List[str]


METHODS = [
    MethodSpec("LSTM", {"model": "lstm"}, []),
    MethodSpec("Transformer", {"model": "transformer"}, []),
    MethodSpec("NCDE", {"model": "ncde"}, []),

    MethodSpec(
        "RFormer-G",
        {
            "model": "transformer",
            "sig_level": 3,
            "num_windows": 25,   # 🔑 FBM speed
        },
        ["use_signatures", "global_backward", "add_time"],
    ),

    MethodSpec(
        "RFormer-L",
        {
            "model": "transformer",
            "sig_level": 3,
<<<<<<< HEAD
            "num_windows": 40,
=======
            "num_windows": 25,
>>>>>>> 9b44c29d0dfde17ac1a17ebe392dee4a3c11a414
        },
        ["use_signatures", "local_tight", "add_time"],
    ),

    MethodSpec(
        "RFormer-GL",
        {
            "model": "transformer",
            "sig_level": 3,
<<<<<<< HEAD
            "num_windows": 40,
=======
            "num_windows": 25,
>>>>>>> 9b44c29d0dfde17ac1a17ebe392dee4a3c11a414
        },
        ["use_signatures", "global_backward", "local_tight", "add_time"],
    ),
]

CONDS = [
    CondSpec("NoDrop", {}, []),
    CondSpec("RandomDropB", {"random_percentage": DROP_KEEP}, ["use_random_drop"]),
]


# -----------------------------
# 2) Parse final_test_acc
# -----------------------------

FINAL_ACC_RE = re.compile(r"final_test_acc:\s*([0-9]*\.?[0-9]+)")

def parse_final_acc(stdout: str) -> Optional[float]:
    m = FINAL_ACC_RE.findall(stdout)
    return float(m[-1]) if m else None


# -----------------------------
# 3) Build / run
# -----------------------------

def build_cmd(method: MethodSpec, cond: CondSpec, seed: int) -> List[str]:
    drop = cond.args.get("random_percentage", None)
    cmd = [
        PYTHON_BIN, MAIN_PY,
        "--config", DEFAULT_CONFIG,
        "--dataset", DATASET,
        "--seed_base", str(seed),
        "--random_percentage", str(drop) if drop is not None else "1.0",
    ]

    if drop is not None:
        cmd.append("--use_random_drop")

    # global args
    for k, v in GLOBAL_ARGS.items():
        cmd += [f"--{k}", str(v)]

    # method args + flags
    for k, v in method.args.items():
        cmd += [f"--{k}", str(v)]
    for fl in method.flags:
        cmd += [f"--{fl}"]

    # condition args + flags
    for k, v in cond.args.items():
        cmd += [f"--{k}", str(v)]
    for fl in cond.flags:
        cmd += [f"--{fl}"]

    return cmd


def run_one(method: MethodSpec, cond: CondSpec, seed: int) -> Tuple[float, float, str]:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{DATASET}__{method.name}__{cond.name}__seed{seed}.log")

    cmd = build_cmd(method, cond, seed)

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
# 4) Resume helpers
# -----------------------------

def load_existing_raw() -> pd.DataFrame:
    if os.path.exists(RAW_CSV):
        return pd.read_csv(RAW_CSV)
    return pd.DataFrame(columns=["method", "cond", "seed", "acc", "time_s", "log_path"])


def done_keys(df: pd.DataFrame) -> set:
    if df.empty:
        return set()
    return set(zip(df["method"], df["cond"], df["seed"]))


def save_raw(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)


def format_pm(mean: float, std: float, decimals: int = 2) -> str:
    mean *= 100.0
    std *= 100.0
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_final_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    # aggregate over seeds (even if 1 seed -> std=0)
    agg = (
        raw_df.groupby(["method", "cond"])["acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["pretty"] = agg.apply(lambda r: format_pm(r["mean"], r["std"]), axis=1)

    # pivot: rows=method, cols=cond
    pivot = agg.pivot(index="method", columns="cond", values="pretty")
    pivot = pivot.reindex(index=[m.name for m in METHODS], columns=[c.name for c in CONDS])
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

    raw_df = pd.DataFrame(columns=["method", "cond", "seed", "acc", "time_s", "log_path"])
    done = done_keys(raw_df)

    jobs = [(m, c, s) for m in METHODS for c in CONDS for s in SEEDS]
    todo = [(m, c, s) for (m, c, s) in jobs if (m.name, c.name, s) not in done]

    pbar = tqdm(todo, desc="T5-FBM ultra-light", unit="run")
    for method, cond, seed in pbar:
        pbar.set_postfix_str(f"{method.name} | {cond.name} | seed={seed}")

        try:
            acc, dt, log_path = run_one(method, cond, seed)
            row = {"method": method.name, "cond": cond.name, "seed": seed,
                   "acc": acc, "time_s": dt, "log_path": log_path}
        except Exception as e:
            print(f"\n[WARN]\n{e}\n")
            row = {"method": method.name, "cond": cond.name, "seed": seed,
                   "acc": float('nan'), "time_s": float('nan'), "log_path": ""}

        raw_df = pd.concat([raw_df, pd.DataFrame([row])], ignore_index=True)
        save_raw(raw_df)

    clean_df = raw_df.dropna(subset=["acc"])
    pivot = build_final_table(clean_df)
    save_final(pivot)

    print("=== TABLE 5 (FBM ultra-light) ===")
    print(pivot.to_string())


if __name__ == "__main__":
    main()