#!/usr/bin/env python3
"""
make_fig3.py — Reproduction Figure 3 (paper-like) [INSTRUMENTED + CACHED RFormer]

Ajouts vs ta version:
- prints flush + timestamps
- tqdm sur seeds + epochs + batch preprocessing
- RFormer: pré-calcul des signatures (train/test) 1 seule fois par seed
- mode --quick pour sanity check rapide
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm

from model_classification import DecoderTransformer
from neuralcde_classification import NCDE_classification
from sig_utils import ComputeSignatures


# =========================
# 0) Utils / Repro
# =========================

def now() -> str:
    import time
    return time.strftime("%H:%M:%S")

def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


# =========================
# 1) Dataset synthétique
# =========================

@dataclass
class SynthConfig:
    n_train: int = 800
    n_test: int = 200
    n_classes: int = 100
    w_min: float = 10.0
    w_max: float = 500.0
    T: int = 2000
    t0: float = 0.0
    t1: float = 1.0
    noise_std: float = 0.1
    use_envelope: bool = True

class FrequencyDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.X = X  # (N,T,C)
        self.y = y  # (N,)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_frequency_data(cfg: SynthConfig, seed: int) -> Tuple[FrequencyDataset, FrequencyDataset]:
    set_all_seeds(seed)

    N = cfg.n_train + cfg.n_test
    T = cfg.T
    device = torch.device("cpu")

    t = torch.linspace(cfg.t0, cfg.t1, T, device=device).view(T, 1)
    ws = torch.linspace(cfg.w_min, cfg.w_max, cfg.n_classes, device=device)

    y = torch.randint(low=0, high=cfg.n_classes, size=(N,), device=device)
    nu = 2.0 * math.pi * torch.rand(N, 1, 1, device=device)
    w = ws[y].view(N, 1, 1)

    if cfg.use_envelope:
        g = 0.5 + 0.5 * torch.sin(math.pi * t)  # (T,1)
    else:
        g = torch.ones_like(t)

    x = g.view(1, T, 1) * torch.sin(w * t.view(1, T, 1) + nu)
    x = x + cfg.noise_std * torch.randn_like(x)

    # (N,T,2) : [t, x]
    X = torch.cat([t.view(1, T, 1).repeat(N, 1, 1), x], dim=2)

    X_train, y_train = X[:cfg.n_train], y[:cfg.n_train]
    X_test, y_test = X[cfg.n_train:], y[cfg.n_train:]

    return FrequencyDataset(X_train, y_train), FrequencyDataset(X_test, y_test)

def make_frequency_data_long(cfg: SynthConfig, seed: int) -> Tuple[FrequencyDataset, FrequencyDataset]:
    """Long sinusoidal: fréquence change au milieu mais label = fréquence de la 1ère moitié (paper-like)."""
    set_all_seeds(seed)

    N = cfg.n_train + cfg.n_test
    T = cfg.T
    device = torch.device("cpu")

    t = torch.linspace(cfg.t0, cfg.t1, T, device=device).view(T, 1)
    ws = torch.linspace(cfg.w_min, cfg.w_max, cfg.n_classes, device=device)

    y = torch.randint(low=0, high=cfg.n_classes, size=(N,), device=device)
    nu = 2.0 * math.pi * torch.rand(N, 1, 1, device=device)
    w1 = ws[y].view(N, 1, 1)

    # fréquence 2 aléatoire
    y2 = torch.randint(low=0, high=cfg.n_classes, size=(N,), device=device)
    w2 = ws[y2].view(N, 1, 1)

    if cfg.use_envelope:
        g = 0.5 + 0.5 * torch.sin(math.pi * t)
    else:
        g = torch.ones_like(t)

    mid = T // 2
    t_all = t.view(1, T, 1)

    x = torch.zeros(N, T, 1, device=device)
    x[:, :mid, :] = g[:mid].view(1, mid, 1) * torch.sin(w1 * t_all[:, :mid, :] + nu)
    x[:, mid:, :] = g[mid:].view(1, T - mid, 1) * torch.sin(w2 * t_all[:, mid:, :] + nu)
    x = x + cfg.noise_std * torch.randn_like(x)

    X = torch.cat([t.view(1, T, 1).repeat(N, 1, 1), x], dim=2)

    X_train, y_train = X[:cfg.n_train], y[:cfg.n_train]
    X_test, y_test = X[cfg.n_train:], y[:cfg.n_train].new_tensor(y[cfg.n_train:])  # same labels
    return FrequencyDataset(X_train, y_train), FrequencyDataset(X_test, y_test)


# =========================
# 2) RFormer signatures (cached)
# =========================

@dataclass
class SigCfg:
    univariate: bool = False
    sig_level: int = 2
    num_windows: int = 100
    global_backward: bool = True
    global_forward: bool = False
    local_tight: bool = True
    local_wide: bool = False
    local_width: float = 50.0

def rformer_transform_tensor(X_points: torch.Tensor, sigcfg: SigCfg, device: torch.device) -> torch.Tensor:
    """
    X_points: (N,T,C)
    return: (N,W,Fsig)
    """
    N, T, C = X_points.shape
    x = np.linspace(0, T - 1, T)

    class _Cfg: pass
    cfg = _Cfg()
    cfg.global_backward = sigcfg.global_backward
    cfg.global_forward = sigcfg.global_forward
    cfg.local_tight = sigcfg.local_tight
    cfg.local_wide = sigcfg.local_wide
    cfg.num_windows = sigcfg.num_windows
    cfg.sig_level = sigcfg.sig_level
    cfg.univariate = sigcfg.univariate
    cfg.local_width = int(sigcfg.local_width)

    return ComputeSignatures(X_points.to(device), x, cfg, device)

@torch.no_grad()
def precompute_rformer_dataset(
    ds: FrequencyDataset,
    sigcfg: SigCfg,
    device: torch.device,
    batch_size: int,
    desc: str,
) -> TensorDataset:
    """
    Pré-calcule signatures pour tout le dataset (beaucoup plus rapide/robuste que par batch/epoch).
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    X_out = []
    y_out = []

    total_samples = len(ds)
    seen = 0

    pbar = tqdm(
        loader,
        desc=f"{desc} (precompute)",
        unit="batch",
        total=len(loader),
        leave=True,
    )

    for X, y in pbar:
        B = X.shape[0]
        Xsig = rformer_transform_tensor(X, sigcfg, device)  # (B,W,Fsig)
        X_out.append(Xsig.cpu())
        y_out.append(y.cpu())

        seen += B
        pbar.set_postfix_str(f"samples {seen}/{total_samples}")

    Xsig_all = torch.cat(X_out, dim=0)
    y_all = torch.cat(y_out, dim=0)
    return TensorDataset(Xsig_all, y_all)


# =========================
# 3) Models factory
# =========================

@dataclass
class TrainCfg:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_head: int = 4
    num_layers: int = 2
    embedded_dim: int = 64

def make_transformer_model(input_dim: int, seq_len: int, n_classes: int, traincfg: TrainCfg, device: torch.device) -> nn.Module:
    class _Cfg: pass
    cfg = _Cfg()
    cfg.n_head = traincfg.n_head
    cfg.num_layers = traincfg.num_layers
    cfg.embedded_dim = traincfg.embedded_dim
    cfg.embd_pdrop = 0.1
    cfg.attn_pdrop = 0.1
    cfg.resid_pdrop = 0.1
    cfg.q_len = 1
    cfg.v_partition = 0.1
    cfg.sub_len = 1
    cfg.scale_att = False
    cfg.sparse = False
    cfg.overlap = False

    return DecoderTransformer(
        cfg,
        input_dim=input_dim,
        n_head=traincfg.n_head,
        layer=traincfg.num_layers,
        seq_num=1,
        n_embd=traincfg.embedded_dim,
        win_len=seq_len,
        num_classes=n_classes,
    ).to(device)

def make_ncde_model(input_channels: int, n_classes: int, device: torch.device) -> nn.Module:
    # ton NCDE_classification accepte X_points (B,T,C) et calcule coeffs en interne (vu ton test)
    return NCDE_classification(
        input_channels=input_channels,
        hidden_channels=64,
        output_channels=n_classes,
        num_layers=2,
        mlp_hidden_dim=128,
        method="rk4",
    ).to(device)


# =========================
# 4) Train loop (tqdm epochs)
# =========================

def train_and_log_curve(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    title: str,
) -> np.ndarray:
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    curve = np.zeros((epochs,), dtype=np.float32)

    epbar = tqdm(range(epochs), desc=title, unit="epoch", leave=False)
    for ep in epbar:
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # test
        model.eval()
        accs = []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                accs.append(accuracy_from_logits(logits, y))

        curve[ep] = float(np.mean(accs))
        epbar.set_postfix_str(f"test={curve[ep]:.3f}")

    return curve


# =========================
# 5) Run dataset (seeds)
# =========================

def run_dataset(
    dataset_name: str,
    make_data_fn,
    data_cfg: SynthConfig,
    train_cfg: TrainCfg,
    seeds: List[int],
    device: torch.device,
    out_dir: str,
    do_ncde: bool = True,
) -> Dict[str, np.ndarray]:
    os.makedirs(out_dir, exist_ok=True)

    sigcfg = SigCfg(sig_level=2, num_windows=100, global_backward=True, local_tight=True)

    all_curves: Dict[str, List[np.ndarray]] = {"Transformer": [], "RFormer": []}
    if do_ncde:
        all_curves["NCDE"] = []

    seedbar = tqdm(seeds, desc=f"{dataset_name}: seeds", unit="seed")
    for seed in seedbar:
        log(f"{dataset_name} | seed={seed} | building data...")
        train_ds, test_ds = make_data_fn(data_cfg, seed=seed)

        # Base loaders (raw X)
        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False)

        # ---------- Transformer ----------
        set_all_seeds(seed)
        tr = make_transformer_model(
            input_dim=train_ds.X.shape[2],
            seq_len=train_ds.X.shape[1],
            n_classes=data_cfg.n_classes,
            traincfg=train_cfg,
            device=device,
        )
        curve_tr = train_and_log_curve(
            tr, train_loader, test_loader, device,
            epochs=train_cfg.epochs, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
            title=f"{dataset_name} | Transformer | seed={seed}"
        )
        all_curves["Transformer"].append(curve_tr)

        # ---------- RFormer (PRECOMPUTE signatures) ----------
        log(f"{dataset_name} | seed={seed} | precomputing signatures (train/test)...")
        sig_train = precompute_rformer_dataset(
            train_ds, sigcfg, device=device, batch_size=train_cfg.batch_size, desc=f"{dataset_name} sig train"
        )
        sig_test = precompute_rformer_dataset(
            test_ds, sigcfg, device=device, batch_size=train_cfg.batch_size, desc=f"{dataset_name} sig test"
        )

        # signature dims
        X0, _ = sig_train[0]
        sig_seq_len = X0.shape[0]
        sig_input_dim = X0.shape[1]

        sig_train_loader = DataLoader(sig_train, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)
        sig_test_loader = DataLoader(sig_test, batch_size=train_cfg.batch_size, shuffle=False)

        set_all_seeds(seed)
        rf = make_transformer_model(
            input_dim=sig_input_dim,
            seq_len=sig_seq_len,
            n_classes=data_cfg.n_classes,
            traincfg=train_cfg,
            device=device,
        )
        curve_rf = train_and_log_curve(
            rf, sig_train_loader, sig_test_loader, device,
            epochs=train_cfg.epochs, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
            title=f"{dataset_name} | RFormer | seed={seed}"
        )
        all_curves["RFormer"].append(curve_rf)

        # ---------- NCDE ----------
        if do_ncde:
            set_all_seeds(seed)
            ncde = make_ncde_model(input_channels=train_ds.X.shape[2], n_classes=data_cfg.n_classes, device=device)
            curve_ncde = train_and_log_curve(
                ncde, train_loader, test_loader, device,
                epochs=train_cfg.epochs, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
                title=f"{dataset_name} | NCDE | seed={seed}"
            )
            all_curves["NCDE"].append(curve_ncde)

    out = {k: np.stack(v, axis=0) for k, v in all_curves.items()}

    # save csv
    csv_path = os.path.join(out_dir, f"fig3_{dataset_name}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,model,mean,std\n")
        for model_name, curves in out.items():
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
            for ep in range(train_cfg.epochs):
                f.write(f"{ep+1},{model_name},{mean[ep]},{std[ep]}\n")

    log(f"[OK] Saved curves CSV -> {csv_path}")
    return out


def plot_fig3(curves_left: Dict[str, np.ndarray], curves_right: Dict[str, np.ndarray], epochs: int, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

    def _plot_panel(ax, curves: Dict[str, np.ndarray], title: str):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        x = np.arange(1, epochs + 1)

        order = ["Transformer", "RFormer", "NCDE"]
        for name in order:
            if name not in curves:
                continue
            y = curves[name].mean(axis=0)
            s = curves[name].std(axis=0, ddof=1) if curves[name].shape[0] > 1 else np.zeros_like(y)
            ax.plot(x, y, label=name)
            ax.fill_between(x, y - s, y + s, alpha=0.2)

        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    _plot_panel(axes[0], curves_left, "Sinusoidal dataset")
    _plot_panel(axes[1], curves_right, "Long Sinusoidal dataset")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log(f"[OK] Saved figure -> {out_path}")


# =========================
# 6) Main + quick mode
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Tiny run to verify everything works fast.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--no_ncde", action="store_true", help="Skip NCDE (useful if too slow).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    out_dir = os.path.join("results", "F3")
    os.makedirs(out_dir, exist_ok=True)

    seeds = [42, 43, 44]

    if args.quick:
        log("QUICK MODE enabled")
        seeds = [42]  # un seul seed
        args.epochs = min(args.epochs, 5)
        args.T = min(args.T, 200)
        args.n_train = min(args.n_train, 128)
        args.n_test = min(args.n_test, 64)

    cfg_short = SynthConfig(
        n_train=args.n_train, n_test=args.n_test, n_classes=100,
        T=args.T, noise_std=0.1, use_envelope=True
    )
    cfg_long = SynthConfig(
        n_train=args.n_train, n_test=args.n_test, n_classes=100,
        T=args.T, noise_std=0.1, use_envelope=True
    )

    train_cfg = TrainCfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        embedded_dim=64,
        num_layers=2,
        n_head=4,
    )

    do_ncde = not args.no_ncde

    log("Running Sinusoidal...")
    curves_short = run_dataset(
        dataset_name="sinusoidal",
        make_data_fn=make_frequency_data,
        data_cfg=cfg_short,
        train_cfg=train_cfg,
        seeds=seeds,
        device=device,
        out_dir=out_dir,
        do_ncde=do_ncde,
    )

    log("Running Long Sinusoidal...")
    curves_long = run_dataset(
        dataset_name="long_sinusoidal",
        make_data_fn=make_frequency_data_long,
        data_cfg=cfg_long,
        train_cfg=train_cfg,
        seeds=seeds,
        device=device,
        out_dir=out_dir,
        do_ncde=do_ncde,
    )

    fig_path = os.path.join(out_dir, "fig3.png")
    plot_fig3(curves_short, curves_long, epochs=train_cfg.epochs, out_path=fig_path)


if __name__ == "__main__":
    main()