"""
Question 2 – produce every artefact needed for the Solution Design section.

Outputs (all in reports/):
  q2_model_summary.txt            full layer-by-layer summary (torchinfo)
  q2_layer_table.csv / .md        stage table: block, output shape (train crop & full clip),
                                  activation, parameters
  q2_param_counts.csv             parameters per part + trainable count per fine-tuning depth
  q2_design_checks.json           all sanity-check numbers (pos_weight, initial loss, overfit test,
                                  CPU latency, pretrained weights loaded or not)
  figures/q2_architecture.png     architecture diagram (built from the real shapes)
  figures/q2_input_pipeline.png   what the network sees: raw -> normalised -> augmented
  figures/q2_overfit_check.png    tiny-batch over-fit test (proves loss/output are wired correctly)

Run (needs:  python -m src.features  first):
    python -m scripts.q2_architecture_report
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.dataset import MimiiSpecDataset  # noqa: E402
from src.model import FaultCNN, count_params, make_loss  # noqa: E402

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
R = config.REPORTS_DIR
FIG = config.FIGURES_DIR
STAGE_INFO = [  # (name, what it is, activation)
    ("Stem", "Conv 3x3, stride 2 (1->32 ch) + BN", "SiLU"),
    ("Stage 1", "1 x MBConv1, k3x3, SE", "SiLU"),
    ("Stage 2", "2 x MBConv6, k3x3, stride 2, SE", "SiLU"),
    ("Stage 3", "2 x MBConv6, k5x5, stride 2, SE", "SiLU"),
    ("Stage 4", "3 x MBConv6, k3x3, stride 2, SE", "SiLU"),
    ("Stage 5", "3 x MBConv6, k5x5, SE", "SiLU"),
    ("Stage 6", "4 x MBConv6, k5x5, stride 2, SE", "SiLU"),
    ("Stage 7", "1 x MBConv6, k3x3, SE", "SiLU"),
    ("Top conv", "Conv 1x1 (320->1280 ch) + BN", "SiLU"),
]


def shapes_for(model: FaultCNN, T: int) -> list[tuple]:
    out = []
    hooks = [b.register_forward_hook(lambda m, i, o, k=k: out.append((k, tuple(o.shape[1:]))))
             for k, b in enumerate(model.features)]
    with torch.no_grad():
        model(torch.zeros(1, 1, config.MODEL_N_MELS, T), torch.tensor([0]))
    for h in hooks:
        h.remove()
    return [s for _, s in sorted(out)]


def fmt(s):  # (C, F, T) -> "C x F x T"
    return " x ".join(str(v) for v in s)


def layer_table(model: FaultCNN) -> pd.DataFrame:
    s_crop = shapes_for(model, config.CROP_FRAMES)
    s_full = shapes_for(model, config.CLIP_FRAMES)
    rows = [{"Layer": "Input", "Type": "normalised log-mel (channel 0)",
             "Output (train crop)": f"1 x {config.MODEL_N_MELS} x {config.CROP_FRAMES}",
             "Output (full clip)": f"1 x {config.MODEL_N_MELS} x {config.CLIP_FRAMES}",
             "Activation": "-", "Parameters": 0}]
    for (name, typ, act), blk, sc, sf in zip(STAGE_INFO, model.features, s_crop, s_full):
        rows.append({"Layer": name, "Type": typ, "Output (train crop)": fmt(sc),
                     "Output (full clip)": fmt(sf), "Activation": act, "Parameters": count_params(blk)})
    h = model.head
    rows += [
        {"Layer": "Global avg pool", "Type": "mean over frequency and time", "Output (train crop)": "1280",
         "Output (full clip)": "1280", "Activation": "-", "Parameters": 0},
        {"Layer": "Concat machine type", "Type": "one-hot pump/fan", "Output (train crop)": "1282",
         "Output (full clip)": "1282", "Activation": "-", "Parameters": 0},
        {"Layer": "Dense 1", "Type": "Linear 1282->128 + Dropout 0.3", "Output (train crop)": "128",
         "Output (full clip)": "128", "Activation": "ReLU", "Parameters": count_params(h[0])},
        {"Layer": "Output", "Type": "Linear 128->1 (logit)", "Output (train crop)": "1",
         "Output (full clip)": "1", "Activation": "Sigmoid (score)", "Parameters": count_params(h[3])},
    ]
    df = pd.DataFrame(rows)
    df.loc[len(df)] = ["TOTAL", "", "", "", "", count_params(model)]
    return df


def param_counts(model: FaultCNN) -> pd.DataFrame:
    rows = [{"Part": "Backbone (EfficientNet-B0 features)", "Parameters": count_params(model.features)},
            {"Part": "Head (dense layers)", "Parameters": count_params(model.head)},
            {"Part": "Total", "Parameters": count_params(model)}]
    for n in [0, 3, 6, 9]:
        model.set_trainable_stages(n)
        rows.append({"Part": f"Trainable if last {n} of 9 feature modules are unfrozen"
                     + (" (head only)" if n == 0 else " (full fine-tuning)" if n == 9 else ""),
                     "Parameters": count_params(model, trainable_only=True)})
    model.set_trainable_stages(9)
    return pd.DataFrame(rows)


def draw_architecture(tbl: pd.DataFrame, total: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=color, ec="#333333", lw=1.1))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9.2, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=7.6)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))

    # --- row 1: input pipeline
    y1, h1 = 4.55, 1.15
    box(0.1, y1, 2.1, h1, "Audio clip", "10 s, 16 kHz, mic channel 0", "#e8eef7")
    box(2.6, y1, 2.3, h1, "Log-mel spectrogram", "128 mel x 313 frames (dB)", "#e8eef7")
    box(5.3, y1, 2.5, h1, "Normalise per mel band", "mean/std from TRAIN only", "#e8eef7")
    box(8.2, y1, 3.3, h1, "Training-only augmentation",
        "random 4-s crop (126 frames)\n±6 dB gain, SpecAugment 2F+2T masks", "#fff4d6")
    box(11.9, y1, 3.0, h1, "Network input", "1 x 128 x 126 (train)\n1 x 128 x 313 (inference)", "#e8eef7")
    for a, b in [(2.2, 2.6), (4.9, 5.3), (7.8, 8.2), (11.5, 11.9)]:
        arrow(a, y1 + h1 / 2, b, y1 + h1 / 2)

    # --- row 2: backbone stages
    y2, h2 = 2.35, 1.45
    names = tbl["Layer"].tolist()[1:10]
    shapes = tbl["Output (train crop)"].tolist()[1:10]
    params = tbl["Parameters"].tolist()[1:10]
    w2, gap, x0 = 1.42, 0.2, 0.1
    for k, (n, s, p) in enumerate(zip(names, shapes, params)):
        x = x0 + k * (w2 + gap)
        box(x, y2, w2, h2, n, f"{s}\n{p/1e3:.1f}k params", "#dff0e1" if k else "#e8f5e9")
        if k:
            arrow(x - gap, y2 + h2 / 2, x, y2 + h2 / 2)
    ax.text(7.45, y2 + h2 + 0.12, "EfficientNet-B0 backbone, ImageNet-pretrained, adapted to 1 channel  "
            "(output shape C x F x T for a 4-s training crop)", ha="center", fontsize=9, style="italic")
    ymid = y1 - 0.2
    ax.plot([13.4, 13.4, 0.81], [y1, ymid, ymid], color="#333", lw=1.3)
    arrow(0.81, ymid, 0.81, y2 + h2)

    # --- row 3: head
    y3, h3 = 0.25, 1.35
    box(0.1, y3, 2.4, h3, "Global average pool", "over freq + time -> 1280", "#f3e5f5")
    box(2.9, y3, 2.4, h3, "Concat machine type", "one-hot pump/fan -> 1282", "#f3e5f5")
    box(5.7, y3, 2.7, h3, "Dense 128, ReLU", "Dropout 0.3", "#f3e5f5")
    box(8.8, y3, 2.4, h3, "Dense 1 (logit)", "Sigmoid -> fault score 0-1", "#f3e5f5")
    box(11.6, y3, 3.3, h3, "Decision", "alarm if score >= threshold\n(threshold set on VAL at 5 % FPR)",
        "#fde2e2")
    last_x = x0 + 8 * (w2 + gap) + w2 / 2
    ymid2 = (y2 + y3 + h3) / 2
    ax.plot([last_x, last_x, 1.3], [y2, ymid2, ymid2], color="#333", lw=1.3)
    arrow(1.3, ymid2, 1.3, y3 + h3)
    for a, b in [(2.5, 2.9), (5.3, 5.7), (8.4, 8.8), (11.2, 11.6)]:
        arrow(a, y3 + h3 / 2, b, y3 + h3 / 2)
    ax.text(7.5, 6.05, f"FaultCNN – total parameters: {total:,}   |   "
            "loss: BCEWithLogits with pos_weight = negatives / positives (train)",
            ha="center", fontsize=10.5, weight="bold")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def input_pipeline_figure(path: Path) -> None:
    raw = np.load(config.FEATURE_DIR / "train.npy", mmap_mode="r")
    idx = pd.read_csv(config.FEATURE_DIR / "train_index.csv")
    i = int(idx.index[idx.label == 1][0])
    ds_eval = MimiiSpecDataset("train", augment=False)
    ds_aug = MimiiSpecDataset("train", augment=True)
    panels = [(np.asarray(raw[i], dtype=np.float32), "1. Log-mel (dB), full 10-s clip"),
              (ds_eval[i][0][0].numpy(), "2. Normalised with train statistics"),
              (ds_aug[i][0][0].numpy(), "3. Training sample: 4-s crop + gain + SpecAugment")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), gridspec_kw={"width_ratios": [313, 313, 126 * 1.6]})
    for ax, (img, title) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title, fontsize=10); ax.set_xlabel("frame"); ax.set_ylabel("mel band")
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def overfit_check(pos_weight: float) -> dict:
    """Train on 32 fixed clips (16 normal, 16 abnormal) with no augmentation.
    A correctly wired model + loss must drive the loss close to zero."""
    ds = MimiiSpecDataset("train", augment=False)
    lab = ds.labels
    rng = np.random.default_rng(config.SEED)
    sel = np.concatenate([rng.choice(np.where(lab == 0)[0], 16, replace=False),
                          rng.choice(np.where(lab == 1)[0], 16, replace=False)])
    xs, ms, ys = zip(*(ds[int(j)] for j in sel))
    x = torch.stack(xs)[..., :config.CROP_FRAMES]
    m, y = torch.stack(ms), torch.stack(ys)
    model = FaultCNN(pretrained=True)
    loss_fn = make_loss(pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for step in range(60):
        opt.zero_grad()
        logits = model(x, m)
        loss = loss_fn(logits, y)
        loss.backward(); opt.step()
        losses.append(loss.item())
    # accuracy of the last training step (train-mode BatchNorm, as during fitting)
    acc = float(((torch.sigmoid(logits.detach()) > 0.5).float() == y).float().mean())
    plt.figure(figsize=(6, 3.4))
    plt.plot(losses, marker=".", lw=1)
    plt.xlabel("step"); plt.ylabel("weighted BCE loss")
    plt.title("Sanity check: over-fitting 32 training clips")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(FIG / "q2_overfit_check.png", dpi=200); plt.close()
    return {"overfit_loss_first": round(losses[0], 4), "overfit_loss_last": round(losses[-1], 4),
            "overfit_train_accuracy_after": round(acc, 3), "overfit_steps": len(losses)}


def latency(model: FaultCNN) -> float:
    model.eval()
    x = torch.randn(1, 1, config.MODEL_N_MELS, config.CLIP_FRAMES)
    m = torch.tensor([0])
    with torch.no_grad():
        for _ in range(3):
            model(x, m)
        t = time.perf_counter()
        for _ in range(20):
            model(x, m)
    return (time.perf_counter() - t) / 20 * 1000


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if not (config.FEATURE_DIR / "train.npy").exists():
        raise SystemExit("Feature cache not found. Run first:  python -m src.features")

    print("Building FaultCNN (downloads ~20 MB ImageNet weights the first time) ...")
    model = FaultCNN(pretrained=True)
    model.eval()

    tbl = layer_table(model)
    tbl.to_csv(R / "q2_layer_table.csv", index=False)
    (R / "q2_layer_table.md").write_text(tbl.to_markdown(index=False), encoding="utf-8")
    print(tbl.to_string(index=False))

    pc = param_counts(model)
    pc.to_csv(R / "q2_param_counts.csv", index=False)
    print("\n" + pc.to_string(index=False))

    try:
        from torchinfo import summary
        s = summary(model, input_data=[torch.zeros(1, 1, config.MODEL_N_MELS, config.CROP_FRAMES),
                                       torch.tensor([0])],
                    depth=3, col_names=["output_size", "num_params"], verbose=0)
        (R / "q2_model_summary.txt").write_text(str(s), encoding="utf-8")
    except ImportError:
        (R / "q2_model_summary.txt").write_text(str(model), encoding="utf-8")

    draw_architecture(tbl, count_params(model), FIG / "q2_architecture.png")
    input_pipeline_figure(FIG / "q2_input_pipeline.png")

    train = MimiiSpecDataset("train")
    pw = train.pos_weight()
    loss_fn = make_loss(pw)
    with torch.no_grad():
        x, m, y = zip(*(train[i] for i in range(64)))
        init_loss = float(loss_fn(model(torch.stack(x), torch.stack(m)), torch.stack(y)))

    print("\nRunning the tiny-batch over-fit check (about 1-3 minutes on CPU) ...")
    checks = {
        "pretrained_imagenet_weights_loaded": model.pretrained,
        "train_clips": len(train), "train_abnormal": int(train.labels.sum()),
        "pos_weight": round(pw, 3),
        "initial_loss_64_clips": round(init_loss, 4),
        "cpu_latency_ms_per_10s_clip": round(latency(model), 1),
        "total_parameters": count_params(model),
        **overfit_check(pw),
    }
    (R / "q2_design_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    print("\nSaved: reports/q2_*.csv|md|txt|json and reports/figures/q2_*.png")


if __name__ == "__main__":
    main()
