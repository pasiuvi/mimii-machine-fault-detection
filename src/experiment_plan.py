"""
Question 3 – the complete, pre-planned list of neural-network experiments (ROUND 1).

Why a Round 1?  Round 0 (archived in experiments/archive/round0_*) showed three problems
on the validation machine (id_04), BEFORE any test use:
  1. Over-fitting to the 2 training machines: train loss -> 0.002 while val loss rose to 4-9.
  2. Pooled TPR@FPR5 mixes pumps and fans whose scores sit on different scales
     (E001: pump 0.17, fan 0.39, but pooled 0.07). The engineer always knows the type,
     so the PRIMARY metric is now MACRO TPR@FPR5 (one threshold per machine type).
  3. The validation metric jumped between epochs (0.00 -> 0.37), so "best epoch" was luck.
Round 1 therefore adds ANN techniques aimed at generalising to an UNSEEN machine
(lower LR + stronger weight decay, weight EMA, within-type mixup) and a more stable
selection rule. Every run is defined here BEFORE it runs, with its hypothesis.
All runs use the same train/val split and are compared on VALIDATION only.

Groups
  initial  : the first, simple neural model                         (Q3a "initial ANN")
  improve  : step-by-step improvements, one change at a time        (Q3a "how it was improved")
  lr       : learning-rate study, around the reference E005         (Q3b hyperparameter 1)
  bs       : batch-size study                                       (Q3b hyperparameter 2)
  depth    : fine-tuning depth = number of unfrozen modules         (Q3b hyperparameter 3)
  optim    : AdamW / Adam(L2) / SGD-momentum                        (Q3b optimisers)
  ablation : remove one ingredient of the reference at a time       (which parts matter)
  domain   : one idea aimed directly at the machine-to-machine shift
  failed   : designed experiments expected to fail, with a reason   (Q3b "did not work")
  combine  : ROUND 2 confirmation - combine the one-factor winners  (added after Round 1,
             decision record reports/Q3_round2_combination.md, written BEFORE these runs)
  final    : selected config, re-run with 2 extra seeds             (seed variance)
"""
from __future__ import annotations

# ---- Q2 design, as used in Round 0 (E003 below) ----
Q2 = {
    "pretrained": True,
    "trainable_stages": 9,          # full fine-tuning (9 of 9 feature modules)
    "optimizer": "adamw",
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "momentum": 0.9,                # only used by SGD
    "batch_size": 32,
    "max_epochs": 25,
    "patience": 7,                  # early stopping on the selection metric (val macro TPR@FPR5)
    "scheduler": "cosine",          # 1 warm-up epoch then cosine decay
    "augment": True,
    "aug": {"crop": True, "gain": True, "specaug": True, "freq_shift": False},
    "mixup_alpha": 0.0,             # 0 = off; >0 = Beta(alpha, alpha) mixing WITHIN a machine type
    "ema_decay": 0.0,               # 0 = off; e.g. 0.998 = exponential moving average of the weights
    "clip_norm": False,             # per-clip band-mean removal (domain experiment)
    "pos_weight": "auto",           # negatives / positives in train (~3.96)
    "seed": 42,
}


def _cfg(base=Q2, **over):
    c = {**base, **{k: v for k, v in over.items() if k != "aug"}}
    c["aug"] = {**base["aug"], **over.get("aug", {})}
    return c


# ---- REFERENCE design for all studies (= E005): the end of the improvement path ----
REF_ID = "E005"
REF = _cfg(lr=1e-4, weight_decay=5e-2, ema_decay=0.998, mixup_alpha=0.4)


def _ref(**over):
    return _cfg(REF, **over)


EXPERIMENTS = [
    # ---------------- initial model ----------------
    dict(id="E001", group="initial",
         what="Initial ANN: frozen pretrained backbone (head only), Adam 1e-3, no augmentation, constant LR",
         hypothesis="Simple transfer-learning start; expected to UNDER-fit (only ~164k head weights learn)",
         cfg=_cfg(trainable_stages=0, optimizer="adam", lr=1e-3, weight_decay=0.0, augment=False,
                  scheduler="none", max_epochs=15)),
    # ---------------- improvement path (one change per step) ----------------
    dict(id="E002", group="improve",
         what="Step 1: unfreeze the whole backbone (full fine-tuning), AdamW 3e-4 + cosine, no augmentation",
         hypothesis="Fixes under-fitting but OVER-fits the training machines (Round 0: val loss rose to 4-9)",
         cfg=_cfg(augment=False)),
    dict(id="E003", group="improve",
         what="Step 2: + Q2 augmentation (random 4-s crop, +-6 dB gain, SpecAugment)",
         hypothesis="Augmentation narrows the train/val gap but is not enough on its own (Round 0)",
         cfg=_cfg()),
    dict(id="E004", group="improve",
         what="Step 3: + gentler fine-tuning: LR 1e-4, weight decay 0.05, EMA of weights (decay 0.998)",
         hypothesis="Smaller, smoother updates keep more of the general pretrained features and make "
                    "the validation curve less jumpy (EMA averages ~500 steps)",
         cfg=_cfg(lr=1e-4, weight_decay=5e-2, ema_decay=0.998)),
    dict(id="E005", group="improve",
         what="Step 4 = REFERENCE: + mixup within machine type (alpha 0.4)",
         hypothesis="Mixing clips forbids memorising individual recordings; best generalisation to the new machine",
         cfg=REF),
    # ---------------- learning rate (reference = 1e-4) ----------------
    dict(id="E006", group="lr", what="Learning rate 3e-5 (else = E005)",
         hypothesis="Too small: under-trains within 25 epochs", cfg=_ref(lr=3e-5)),
    dict(id="E007", group="lr", what="Learning rate 3e-4 (else = E005)",
         hypothesis="Faster, but more over-fitting to the training machines", cfg=_ref(lr=3e-4)),
    dict(id="E008", group="lr", what="Learning rate 1e-3 (else = E005)",
         hypothesis="Too large for fine-tuning: destroys pretrained features, unstable", cfg=_ref(lr=1e-3)),
    # ---------------- batch size (reference = 32) ----------------
    dict(id="E009", group="bs", what="Batch size 16 (else = E005)",
         hypothesis="Noisier gradients regularise; twice the steps per epoch", cfg=_ref(batch_size=16)),
    dict(id="E010", group="bs", what="Batch size 64 (else = E005)",
         hypothesis="Smoother gradients, half the updates; may need a larger LR", cfg=_ref(batch_size=64)),
    # ---------------- fine-tuning depth (reference = 9 of 9) ----------------
    dict(id="E011", group="depth", what="Fine-tune last 0 of 9 modules (head only), else = E005",
         hypothesis="Frozen ImageNet features generalise but are not specific enough",
         cfg=_ref(trainable_stages=0, lr=1e-3)),
    dict(id="E012", group="depth", what="Fine-tune last 3 of 9 modules, else = E005",
         hypothesis="Adapting only high-level layers keeps generic filters -> may transfer best",
         cfg=_ref(trainable_stages=3)),
    dict(id="E013", group="depth", what="Fine-tune last 6 of 9 modules, else = E005",
         hypothesis="Middle ground between capacity and preserved pretraining", cfg=_ref(trainable_stages=6)),
    # ---------------- optimisers (reference = AdamW) ----------------
    dict(id="E014", group="optim", what="Adam with coupled L2 (weight_decay 1e-4), else = E005",
         hypothesis="L2 inside Adam is scaled by the adaptive step, so it regularises less than AdamW",
         cfg=_ref(optimizer="adam", weight_decay=1e-4)),
    dict(id="E015", group="optim", what="SGD + Nesterov momentum 0.9, lr 1e-2, wd 1e-4, else = E005",
         hypothesis="Needs a ~100x larger LR; slower start, often generalises well",
         cfg=_ref(optimizer="sgd", lr=1e-2, weight_decay=1e-4)),
    dict(id="E016", group="optim", what="SGD + Nesterov momentum 0.9, lr 3e-3, wd 1e-4, else = E005",
         hypothesis="Shows how sensitive SGD is to its learning rate",
         cfg=_ref(optimizer="sgd", lr=3e-3, weight_decay=1e-4)),
    # ---------------- ablations (remove one ingredient of E005) ----------------
    dict(id="E017", group="ablation", what="Remove EMA of weights (else = E005)",
         hypothesis="Validation metric becomes noisier between epochs", cfg=_ref(ema_decay=0.0)),
    dict(id="E018", group="ablation", what="Remove SpecAugment (else = E005)",
         hypothesis="More over-fitting: larger train/val gap", cfg=_ref(aug={"specaug": False})),
    dict(id="E019", group="ablation", what="Remove the +-6 dB gain augmentation (else = E005)",
         hypothesis="Model leans on loudness, which Q1 showed is machine-specific", cfg=_ref(aug={"gain": False})),
    # ---------------- domain shift ----------------
    dict(id="E020", group="domain", what="Per-clip band-mean removal (clip_norm), else = E005",
         hypothesis="Removes each machine's own spectral envelope (domain shift) - but may also remove "
                    "the steady fault tone (Q1 Finding 2); uncertain",
         cfg=_ref(clip_norm=True)),
    # ---------------- designed failures ----------------
    dict(id="E021", group="failed", what="Train from scratch (no ImageNet weights), LR 1e-3, else = E005",
         hypothesis="Expected to FAIL: 2 machines per type are too few to learn good filters (Q2 2.b)",
         cfg=_ref(pretrained=False, lr=1e-3)),
    dict(id="E022", group="failed", what="Add random frequency-shift augmentation (+-8 mel bands)",
         hypothesis="Expected to FAIL: breaks the Q2 assumption that frequency position carries meaning",
         cfg=_ref(aug={"freq_shift": True})),
    dict(id="E023", group="failed", what="Unweighted loss (pos_weight = 1), else = E005",
         hypothesis="Expected to hurt: rare faults get less weight, low-FPR ranking suffers",
         cfg=_ref(pos_weight=1.0)),
    # ---------------- ROUND 2: combine the winners of the one-factor studies ----------------
    # Round 1 winners (stable val macro TPR@FPR5): SGD-Nesterov 1e-2 (E015, 0.435), AdamW LR 1e-3 (E008, 0.362),
    # last 3 blocks (E012, 0.358), batch 16 (E009, 0.339), no EMA (E017, 0.288 vs E005 0.234).
    dict(id="E024", group="combine", what="SGD-Nesterov 1e-2 + fine-tune last 3 blocks (E015 + E012)",
         hypothesis="Best optimiser + best depth: keeping generic low-level filters frozen should help SGD transfer too",
         cfg=_ref(optimizer="sgd", lr=1e-2, weight_decay=1e-4, trainable_stages=3)),
    dict(id="E025", group="combine", what="SGD-Nesterov 1e-2 + batch size 16 (E015 + E009)",
         hypothesis="More, noisier updates helped AdamW; may help SGD - or overshoot (LR/batch noise doubles)",
         cfg=_ref(optimizer="sgd", lr=1e-2, weight_decay=1e-4, batch_size=16)),
    dict(id="E026", group="combine", what="SGD-Nesterov 1e-2 without EMA (E015 + E017)",
         hypothesis="EMA lowered the level at LR 1e-4 (E017); checks whether it also holds SGD back",
         cfg=_ref(optimizer="sgd", lr=1e-2, weight_decay=1e-4, ema_decay=0.0)),
    dict(id="E027", group="combine", what="AdamW LR 1e-3 + fine-tune last 3 blocks (E008 + E012)",
         hypothesis="Best Adam-type setting + best depth; tests whether the depth gain also holds at a larger LR",
         cfg=_ref(lr=1e-3, trainable_stages=3)),
]

PLAN = {e["id"]: e for e in EXPERIMENTS}
GROUPS = ["initial", "improve", "lr", "bs", "depth", "optim", "ablation", "domain", "failed", "combine"]
