"""
Question 3 – training engine for FaultCNN.

One call = one experiment:
    from src.train import run_experiment
    run_experiment("E003", cfg, what="...", group="improve", hypothesis="...")

For every epoch it records: train loss, mean gradient norm, learning rate,
validation loss, validation TPR@FPR5 / ROC-AUC / pAUC, epoch time.
It keeps the checkpoint with the best VALIDATION TPR@FPR5 (early stopping),
saves everything under experiments/runs/<id>/ and appends one row to
experiments/experiment_log.csv. The TEST set is never loaded here.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from src import config
from src.dataset import MimiiSpecDataset, worker_init_fn
from src.evaluation import get_device, metrics_by_type, predict_logits
from src.model import FaultCNN, count_params

RUNS_DIR = config.ROOT / "experiments" / "runs"
MODELS_DIR = config.ROOT / "models"
LOG_CSV = config.ROOT / "experiments" / "experiment_log.csv"
LOG_COLUMNS = ["exp_id", "date", "group", "model_family", "what_changed", "key_hyperparameters", "seed",
               "epochs_run", "best_epoch", "val_TPR@FPR5", "val_ROC_AUC", "val_pAUC_0.1",
               "val_TPR@FPR5_pump", "val_TPR@FPR5_fan", "train_time_min", "hypothesis", "outcome",
               "what_i_learned", "git_commit"]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def make_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg["optimizer"]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "adam":   # classic Adam: weight decay is added to the gradient (coupled L2)
        return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], nesterov=True,
                               weight_decay=cfg["weight_decay"])
    raise ValueError(name)


def lr_lambda_factory(cfg, steps_per_epoch: int):
    total = cfg["max_epochs"] * steps_per_epoch
    warm = steps_per_epoch                          # 1 warm-up epoch
    if cfg["scheduler"] == "none":
        return lambda step: 1.0

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return f


def freeze_bn_of_frozen_stages(model: FaultCNN) -> None:
    """Frozen modules must not update BatchNorm running statistics either."""
    for block in model.features:
        if not any(p.requires_grad for p in block.parameters()):
            block.eval()


def run_experiment(exp_id: str, cfg: dict, what: str = "", group: str = "", hypothesis: str = "",
                   max_epochs: int | None = None, num_workers: int = 0, verbose: bool = True) -> dict:
    cfg = json.loads(json.dumps(cfg))                      # deep copy
    if max_epochs:
        cfg["max_epochs"] = max_epochs
    set_seed(cfg["seed"])
    device = get_device()
    run_dir = RUNS_DIR / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    train_ds = MimiiSpecDataset("train", augment=cfg["augment"], seed=cfg["seed"], aug=cfg["aug"])
    val_ds = MimiiSpecDataset("val", augment=False)
    val_type = val_ds.index["machine_type"].to_numpy()
    g = torch.Generator(); g.manual_seed(cfg["seed"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=num_workers, worker_init_fn=worker_init_fn if num_workers else None,
                        generator=g, pin_memory=device.type == "cuda")

    model = FaultCNN(pretrained=cfg["pretrained"]).to(device)
    model.set_trainable_stages(cfg["trainable_stages"])
    pw = train_ds.pos_weight() if cfg["pos_weight"] == "auto" else float(cfg["pos_weight"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg, len(loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if verbose:
        print(f"\n=== {exp_id} [{group}] {what}\n    device={device} | trainable params="
              f"{count_params(model, trainable_only=True):,} | pos_weight={pw:.3f} | "
              f"{len(loader)} steps/epoch | max {cfg['max_epochs']} epochs")

    history, best, best_epoch, bad = [], -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        model.train(); freeze_bn_of_frozen_stages(model)
        losses, gnorms = [], []
        for x, m, y in loader:
            x, m, y = x.to(device, non_blocking=True), m.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logit = model(x, m)
            loss = loss_fn(logit.float(), y)
            if not torch.isfinite(loss):
                losses.append(float("nan")); break
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item()); gnorms.append(float(gn))
        train_loss = float(np.mean(losses)) if losses else float("nan")

        z_val = predict_logits(model, val_ds, device)              # one pass: scores + loss
        s_val = expit(z_val)
        mv = metrics_by_type(val_ds.labels, s_val, val_type)
        vloss = float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(z_val), torch.from_numpy(val_ds.labels), pos_weight=torch.tensor(pw)))
        row = {"epoch": epoch, "train_loss": train_loss, "grad_norm": float(np.mean(gnorms)) if gnorms else float("nan"),
               "lr": opt.param_groups[0]["lr"], "val_loss": vloss,
               "val_TPR@FPR5": mv["all"]["TPR@FPR5"], "val_ROC_AUC": mv["all"]["ROC_AUC"],
               "val_pAUC_0.1": mv["all"]["pAUC_0.1"],
               "val_TPR@FPR5_pump": mv.get("pump", {}).get("TPR@FPR5", np.nan),
               "val_TPR@FPR5_fan": mv.get("fan", {}).get("TPR@FPR5", np.nan),
               "epoch_time_s": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if verbose:
            print(f"  ep {epoch:2d} | train {train_loss:.4f} | val loss {vloss:.4f} | "
                  f"val TPR@5% {row['val_TPR@FPR5']:.3f} | AUC {row['val_ROC_AUC']:.3f} | "
                  f"lr {row['lr']:.1e} | {row['epoch_time_s']:.0f}s")

        if not math.isfinite(train_loss):
            print("  loss became NaN/inf -> run stopped (diverged)")
            break
        if row["val_TPR@FPR5"] > best:
            best, best_epoch, bad = row["val_TPR@FPR5"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "epoch": epoch, "exp_id": exp_id},
                       MODELS_DIR / f"{exp_id}.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"  early stop: no improvement for {cfg['patience']} epochs")
                break

    train_min = (time.time() - t_start) / 60
    hist = pd.DataFrame(history)
    b = hist.loc[hist.epoch == best_epoch].iloc[0] if best_epoch else hist.iloc[-1]
    summary = {"exp_id": exp_id, "group": group, "what": what, "hypothesis": hypothesis, "cfg": cfg,
               "epochs_run": int(hist.epoch.max()), "best_epoch": int(best_epoch),
               "val_TPR@FPR5": float(b["val_TPR@FPR5"]), "val_ROC_AUC": float(b["val_ROC_AUC"]),
               "val_pAUC_0.1": float(b["val_pAUC_0.1"]),
               "val_TPR@FPR5_pump": float(b["val_TPR@FPR5_pump"]), "val_TPR@FPR5_fan": float(b["val_TPR@FPR5_fan"]),
               "train_time_min": round(train_min, 2), "device": str(device),
               "trainable_params": count_params(model, trainable_only=True),
               "diverged": not math.isfinite(train_loss), "git_commit": git_commit()}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    append_log(summary)
    if verbose:
        print(f"  BEST epoch {best_epoch}: val TPR@FPR5 = {summary['val_TPR@FPR5']:.3f} "
              f"(pump {summary['val_TPR@FPR5_pump']:.3f}, fan {summary['val_TPR@FPR5_fan']:.3f}) "
              f"| {train_min:.1f} min")
    return summary


def append_log(s: dict) -> None:
    c = s["cfg"]
    hp = (f"opt={c['optimizer']} lr={c['lr']} wd={c['weight_decay']} bs={c['batch_size']} "
          f"stages={c['trainable_stages']} pretrained={c['pretrained']} sched={c['scheduler']} "
          f"aug={'off' if not c['augment'] else ','.join(k for k, v in c['aug'].items() if v)} "
          f"pos_weight={c['pos_weight']}")
    row = {"exp_id": s["exp_id"], "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "group": s["group"],
           "model_family": "ANN (FaultCNN)", "what_changed": s["what"], "key_hyperparameters": hp,
           "seed": c["seed"], "epochs_run": s["epochs_run"], "best_epoch": s["best_epoch"],
           "val_TPR@FPR5": round(s["val_TPR@FPR5"], 4), "val_ROC_AUC": round(s["val_ROC_AUC"], 4),
           "val_pAUC_0.1": round(s["val_pAUC_0.1"], 4),
           "val_TPR@FPR5_pump": round(s["val_TPR@FPR5_pump"], 4), "val_TPR@FPR5_fan": round(s["val_TPR@FPR5_fan"], 4),
           "train_time_min": s["train_time_min"], "hypothesis": s["hypothesis"],
           "outcome": "DIVERGED (NaN loss)" if s["diverged"] else "completed",
           "what_i_learned": "", "git_commit": s["git_commit"]}
    write_log_row(row)


def write_log_row(row: dict) -> None:
    LOG_CSV.parent.mkdir(exist_ok=True)
    if LOG_CSV.exists():
        log = pd.read_csv(LOG_CSV)
        if not set(LOG_COLUMNS).issubset(log.columns):      # old Q1 template -> keep rows, new columns
            log = log.reindex(columns=LOG_COLUMNS)
        log = log[log.exp_id != row["exp_id"]]               # re-run replaces the old row
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row], columns=LOG_COLUMNS)
    log.to_csv(LOG_CSV, index=False)
