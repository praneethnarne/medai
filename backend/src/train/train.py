"""Two-stage fine-tuning for the MEdAI baseline.

    python backend/src/train/train.py --collection ham10000 --target lesion_type

Stage 1 trains only the classifier head with the backbone frozen. Stage 2
unfreezes the backbone and continues at a much lower learning rate. The second
stage is the point: the codebase this project replaced froze its backbone and
never unfroze it, which is why it plateaued around 0.69 accuracy. Feature
extraction alone leaves most of the available performance on the table.

Every run writes its config, per-epoch history and best checkpoint under
results/runs/<run_id>/, so any reported number can be traced back to the exact
settings that produced it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bootstrap  # noqa: F401,E402  (sets SSL_CERT_FILE before torch.hub runs)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import models  # noqa: E402

from data.dataset import ISICDataset  # noqa: E402
from train import metrics  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]

ARCHS = {
    "efficientnet_b0": (models.efficientnet_b0,
                        models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (models.mobilenet_v3_small,
                           models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(arch: str, n_classes: int):
    builder, weights = ARCHS[arch]
    model = builder(weights=weights)
    if arch.startswith("efficientnet"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_f, n_classes)
        head = model.classifier
    elif arch.startswith("mobilenet"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_f, n_classes)
        head = model.classifier
    else:
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, n_classes)
        head = model.fc
    return model, head


def set_backbone_trainable(model, head, trainable: bool) -> None:
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return metrics.report(y_true, y_pred, class_names)


def run_epochs(model, head, loaders, device, class_names, *, epochs, lr,
               weight_decay, loss_fn, stage, history, best, run_dir, patience):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    train_loader, val_loader = loaders
    since_improved = 0

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        total_loss, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()

        val = evaluate(model, val_loader, device, class_names)
        record = {
            "stage": stage,
            "epoch": epoch,
            "lr": sched.get_last_lr()[0],
            "train_loss": total_loss / max(seen, 1),
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_macro_recall": val["macro_recall"],
            "seconds": round(time.time() - started, 1),
        }
        history.append(record)
        print(f"  [{stage}] epoch {epoch:2d}/{epochs}  "
              f"loss {record['train_loss']:.4f}  "
              f"val_acc {val['accuracy']:.4f}  "
              f"val_macroF1 {val['macro_f1']:.4f}  "
              f"({record['seconds']:.0f}s)", flush=True)

        # Selection is on macro F1, not accuracy: with these class balances a
        # model that ignores the rare classes can still look accurate.
        if val["macro_f1"] > best["macro_f1"]:
            best.update({"macro_f1": val["macro_f1"], "stage": stage,
                         "epoch": epoch, "report": val})
            torch.save(model.state_dict(), run_dir / "best.pt")
            since_improved = 0
        else:
            since_improved += 1
            if patience and since_improved >= patience:
                print(f"  [{stage}] no improvement for {patience} epochs, stopping")
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", default="ham10000")
    ap.add_argument("--target", default="lesion_type", choices=["triage", "lesion_type"])
    ap.add_argument("--arch", default="efficientnet_b0", choices=sorted(ARCHS))
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-full", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-full", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--weighting", default="balanced", choices=["balanced", "sqrt", "none"])
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to start from, for the fine-tune phase")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()

    train_ds = ISICDataset(args.collection, "train", target=args.target)
    val_ds = ISICDataset(args.collection, "val", target=args.target)
    class_names = list(train_ds.classes)

    run_id = args.run_name or (f"{args.collection}_{args.target}_{args.arch}_"
                               f"{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir = ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders = (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.workers, persistent_workers=args.workers > 0,
                   drop_last=True),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.workers, persistent_workers=args.workers > 0),
    )

    model, head = build_model(args.arch, len(class_names))
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"initialised from {args.init_from}")
    model = model.to(device)

    if args.weighting == "none":
        weights = None
    else:
        weights = train_ds.class_weights(args.weighting).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights,
                                        label_smoothing=args.label_smoothing)

    config = {
        **vars(args),
        "run_id": run_id,
        "device": str(device),
        "class_names": class_names,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "class_counts": {c: int(n) for c, n in
                         zip(class_names, train_ds.class_counts().tolist())},
        "class_weights": (None if weights is None
                          else [round(w, 4) for w in weights.cpu().tolist()]),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"run {run_id} on {device}")
    print(f"  train {len(train_ds)}  val {len(val_ds)}  classes {class_names}")

    history: list[dict] = []
    best = {"macro_f1": -1.0}

    print("stage 1: frozen backbone, head only")
    set_backbone_trainable(model, head, False)
    run_epochs(model, head, loaders, device, class_names, epochs=args.epochs_head,
               lr=args.lr_head, weight_decay=args.weight_decay, loss_fn=loss_fn,
               stage="head", history=history, best=best, run_dir=run_dir,
               patience=0)

    print("stage 2: full fine-tune")
    set_backbone_trainable(model, head, True)
    run_epochs(model, head, loaders, device, class_names, epochs=args.epochs_full,
               lr=args.lr_full, weight_decay=args.weight_decay, loss_fn=loss_fn,
               stage="full", history=history, best=best, run_dir=run_dir,
               patience=args.patience)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (run_dir / "best_val_report.json").write_text(json.dumps(best, indent=2) + "\n")

    print(f"\nbest: stage={best['stage']} epoch={best['epoch']} "
          f"val macro F1={best['macro_f1']:.4f}")
    print(metrics.format_report(best["report"]))
    print(f"\nartifacts in results/runs/{run_id}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
