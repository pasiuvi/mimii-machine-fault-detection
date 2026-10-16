"""
Question 3 – the complete, pre-planned list of neural-network experiments.

Every run is defined here BEFORE it is run, with its hypothesis. All runs use the
SAME split (train / validation), and are compared on VALIDATION only. The test set
is touched once, at the very end, by scripts/q3_final_evaluate.py.

Groups
  initial  : the first, simple neural model                         (Q3a "initial ANN")
  improve  : the steps that turned it into the final design         (Q3a "how it was improved")
  lr       : learning-rate study                                    (Q3b hyperparameter 1)
  bs       : batch-size study                                       (Q3b hyperparameter 2)
  depth    : fine-tuning depth = number of unfrozen modules         (Q3b hyperparameter 3)
  optim    : optimiser comparison AdamW / Adam(L2) / SGD-momentum   (Q3b optimisers)
  ablation : remove one augmentation at a time                      (supports Q2 design choices)
  failed   : designed experiments expected to fail, with a reason   (Q3b "did not work")
  final    : best development config, re-run with 2 extra seeds     (seed variance)
"""
from __future__ import annotations

# Shared defaults = the Q2 design ("reference" config, E003)
BASE = {
    "pretrained": True,
    "trainable_stages": 9,          # full fine-tuning
    "optimizer": "adamw",
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "momentum": 0.9,                # only used by SGD
    "batch_size": 32,
    "max_epochs": 25,
    "patience": 6,                  # early stopping on validation TPR@FPR5
    "scheduler": "cosine",          # 1 warm-up epoch then cosine decay
    "augment": True,
    "aug": {"crop": True, "gain": True, "specaug": True, "freq_shift": False},
    "pos_weight": "auto",           # negatives / positives in train (3.959)
    "seed": 42,
}


def _cfg(**over):
    c = {**BASE, **{k: v for k, v in over.items() if k != "aug"}}
    c["aug"] = {**BASE["aug"], **over.get("aug", {})}
    return c


EXPERIMENTS = [
    # ---------------- initial model ----------------
    dict(id="E001", group="initial",
         what="Initial ANN: frozen pretrained backbone (head only), Adam 1e-3, no augmentation, constant LR",
         hypothesis="A simple transfer-learning starting point; expected to under-fit because only 164k head weights learn",
         cfg=_cfg(trainable_stages=0, optimizer="adam", lr=1e-3, weight_decay=0.0, augment=False,
                  scheduler="none", max_epochs=15)),
    # ---------------- improvement steps ----------------
    dict(id="E002", group="improve",
         what="Unfreeze the whole backbone (full fine-tuning), AdamW 3e-4 + cosine, still no augmentation",
         hypothesis="More capacity fixes under-fitting but will over-fit the 4 training machines without augmentation",
         cfg=_cfg(augment=False)),
    dict(id="E003", group="improve",
         what="Reference design (Q2): full fine-tuning + crop + gain + SpecAugment, AdamW 3e-4, cosine",
         hypothesis="Augmentation reduces over-fitting and improves validation TPR@FPR5",
         cfg=_cfg()),
    # ---------------- learning rate ----------------
    dict(id="E004", group="lr", what="Learning rate 1e-4 (else = E003)",
         hypothesis="Too small: slower progress, may stop before converging", cfg=_cfg(lr=1e-4)),
    dict(id="E005", group="lr", what="Learning rate 1e-3 (else = E003)",
         hypothesis="Faster, but may damage pretrained features", cfg=_cfg(lr=1e-3)),
    dict(id="E006", group="lr", what="Learning rate 3e-3 (else = E003)",
         hypothesis="Expected to FAIL: too large for fine-tuning, unstable loss", cfg=_cfg(lr=3e-3)),
    # ---------------- batch size ----------------
    dict(id="E007", group="bs", what="Batch size 16 (else = E003)",
         hypothesis="Noisier gradients act as regularisation; more steps per epoch", cfg=_cfg(batch_size=16)),
    dict(id="E008", group="bs", what="Batch size 64 (else = E003)",
         hypothesis="Smoother gradients but fewer updates; BatchNorm statistics more stable",
         cfg=_cfg(batch_size=64)),
    # ---------------- fine-tuning depth ----------------
    dict(id="E009", group="depth", what="Fine-tune last 0 of 9 modules (head only) WITH augmentation",
         hypothesis="Frozen ImageNet features are not specific enough for machine sounds", cfg=_cfg(trainable_stages=0)),
    dict(id="E010", group="depth", what="Fine-tune last 3 of 9 modules",
         hypothesis="Adapting only high-level layers keeps generic low-level filters", cfg=_cfg(trainable_stages=3)),
    dict(id="E011", group="depth", what="Fine-tune last 6 of 9 modules",
         hypothesis="Middle ground between capacity and preserved pretraining", cfg=_cfg(trainable_stages=6)),
    # ---------------- optimisers ----------------
    dict(id="E012", group="optim", what="Adam with coupled L2 weight decay 1e-4, lr 3e-4",
         hypothesis="Similar to AdamW; L2 interacts with adaptive scaling so regularisation is weaker",
         cfg=_cfg(optimizer="adam", weight_decay=1e-4)),
    dict(id="E013", group="optim", what="SGD + momentum 0.9 (Nesterov), lr 1e-2, wd 1e-4",
         hypothesis="Needs a larger LR; slower start than Adam-type, can generalise well",
         cfg=_cfg(optimizer="sgd", lr=1e-2, weight_decay=1e-4)),
    dict(id="E014", group="optim", what="SGD + momentum 0.9 (Nesterov), lr 3e-3, wd 1e-4",
         hypothesis="Checks SGD's sensitivity to the learning rate",
         cfg=_cfg(optimizer="sgd", lr=3e-3, weight_decay=1e-4)),
    # ---------------- augmentation ablation ----------------
    dict(id="E015", group="ablation", what="Remove the +-6 dB gain augmentation",
         hypothesis="Validation may rise (loudness helps on the val machine) but the model relies on loudness",
         cfg=_cfg(aug={"gain": False})),
    dict(id="E016", group="ablation", what="Remove SpecAugment",
         hypothesis="More over-fitting: larger train/val gap", cfg=_cfg(aug={"specaug": False})),
    # ---------------- designed failures ----------------
    dict(id="E017", group="failed", what="Train from scratch (no ImageNet weights), else = E003",
         hypothesis="Expected to FAIL: 4 training machines are too few to learn good filters (Q2 2.b)",
         cfg=_cfg(pretrained=False, lr=1e-3)),
    dict(id="E018", group="failed", what="Add random frequency-shift augmentation (+-8 mel bands)",
         hypothesis="Expected to FAIL: breaks the Q2 assumption that frequency position carries meaning",
         cfg=_cfg(aug={"freq_shift": True})),
    dict(id="E019", group="failed", what="Unweighted loss (pos_weight = 1)",
         hypothesis="Rare faults get less attention; ranking at low FPR may suffer", cfg=_cfg(pos_weight=1.0)),
]

PLAN = {e["id"]: e for e in EXPERIMENTS}
GROUPS = ["initial", "improve", "lr", "bs", "depth", "optim", "ablation", "failed"]
