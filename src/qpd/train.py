"""Training entry point.

Stages (run in order):
    python -m qpd.train stage=finetune     # finetune ResNet-18 on the GPU, save artifacts/backbone.pt
    python -m qpd.train stage=embed        # cache 512-d embeddings for train/val/test
    python -m qpd.train stage=head head=quantum quantum.n_qubits=6 data_frac=0.05 seed=1
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from qpd.data import load_split, make_loader, stratified_fraction
from qpd.metrics import binary_metrics
from qpd.models import LogisticHead, QuantumHead, build_backbone, build_head
from qpd.models.backbone import EMBED_DIM
from qpd.utils import count_params, get_device, set_seed


def _pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = labels.mean()
    return torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(device))).cpu())
        ys.append(y.view(-1))
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def finetune(cfg: DictConfig, device) -> None:
    art = Path(cfg.paths.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = load_split("train", cfg.data.root, cfg.data.size), load_split("val", cfg.data.root, cfg.data.size)
    train_dl = make_loader(train_ds, cfg.data.batch_size, True, cfg.data.num_workers)
    val_dl = make_loader(val_ds, cfg.data.batch_size, False, cfg.data.num_workers)

    model = nn.Sequential(build_backbone(pretrained=True), LogisticHead(EMBED_DIM)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.finetune.lr, weight_decay=cfg.finetune.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_ds.labels.ravel()).to(device))

    best = -1.0
    for epoch in range(cfg.finetune.epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.view(-1).float().to(device))
            loss.backward()
            opt.step()
        m = binary_metrics(*_predict(model, val_dl, device))
        mlflow.log_metrics({f"val_{k}": v for k, v in m.items()}, step=epoch)
        print(f"epoch {epoch}: val auroc={m['auroc']:.4f}")
        if m["auroc"] > best:
            best = m["auroc"]
            torch.save(model[0].state_dict(), art / "backbone.pt")
    mlflow.log_metric("best_val_auroc", best)


@torch.no_grad()
def embed(cfg: DictConfig, device) -> None:
    art = Path(cfg.paths.artifacts)
    backbone = build_backbone(pretrained=False)
    backbone.load_state_dict(torch.load(art / "backbone.pt", map_location="cpu"))
    backbone.to(device).eval()
    out = {}
    for split in ("train", "val", "test"):
        ds = load_split(split, cfg.data.root, cfg.data.size, train_aug=False)
        feats, labels = [], []
        for x, y in make_loader(ds, cfg.data.batch_size, False, cfg.data.num_workers):
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y.view(-1))
        out[f"X_{split}"] = torch.cat(feats).numpy()
        out[f"y_{split}"] = torch.cat(labels).numpy()
        print(f"{split}: {out[f'X_{split}'].shape}")
    np.savez_compressed(art / "embeddings.npz", **out)


def train_head(cfg: DictConfig, device) -> dict:
    d = np.load(Path(cfg.paths.artifacts) / "embeddings.npz")
    idx = stratified_fraction(d["y_train"], cfg.data_frac, cfg.seed)
    Xtr, ytr = torch.tensor(d["X_train"][idx]), torch.tensor(d["y_train"][idx]).float()
    Xva, yva = torch.tensor(d["X_val"]), d["y_val"]
    Xte, yte = torch.tensor(d["X_test"]), d["y_test"]

    q = cfg.quantum
    kw = {"backend": q.backend, "shots": q.shots} if cfg.head == "quantum" else {}
    head = build_head(cfg.head, Xtr.shape[1], q.n_qubits, q.n_layers, **kw)
    if cfg.head != "quantum" or q.backend == "local":
        head = head.to(device)
    dev = next(head.parameters()).device
    mlflow.log_param("trainable_params", count_params(head))
    mlflow.log_param("n_train", len(idx))

    opt = torch.optim.Adam(head.parameters(), lr=cfg.head_train.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(ytr.numpy()).to(dev))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr),
                                         batch_size=cfg.head_train.batch_size, shuffle=True)

    def score(X):
        head.eval()
        with torch.no_grad():
            return torch.sigmoid(head(X.to(dev))).cpu().numpy()

    best, best_state, t0 = -1.0, None, time.time()
    for epoch in range(cfg.head_train.epochs):
        head.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(head(xb.to(dev)), yb.to(dev))
            loss.backward()
            opt.step()
        val = binary_metrics(yva, score(Xva))
        mlflow.log_metrics({"train_loss": loss.item(), "val_auroc": val["auroc"]}, step=epoch)
        if val["auroc"] > best:
            best, best_state = val["auroc"], {k: v.detach().clone() for k, v in head.state_dict().items()}
    train_time = time.time() - t0

    head.load_state_dict(best_state)
    if cfg.head == "quantum" and q.eval_backend != q.backend:
        head.cpu()
        head.set_backend(q.eval_backend, q.shots)
        dev = torch.device("cpu")
    test = binary_metrics(yte, score(Xte))
    results = {"head": cfg.head, "data_frac": cfg.data_frac, "seed": cfg.seed,
               "n_qubits": q.n_qubits, "n_layers": q.n_layers, "trainable_params": count_params(head),
               "train_time_s": train_time, "best_val_auroc": best, **{f"test_{k}": v for k, v in test.items()}}
    mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, float)})
    print(json.dumps(results, indent=2))
    Path("results").mkdir(exist_ok=True)
    tag = f"{cfg.head}_q{q.n_qubits}_l{q.n_layers}_f{cfg.data_frac}_s{cfg.seed}"
    (Path("results") / f"{tag}.json").write_text(json.dumps(results, indent=2))
    if cfg.head == "quantum":
        torch.save(head.state_dict(), Path(cfg.paths.artifacts) / f"head_{tag}.pt")
    return results


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name=f"{cfg.stage}-{cfg.head}"):
        flat = OmegaConf.to_container(cfg, resolve=True)
        flat.pop("hydra", None)
        mlflow.log_params({"stage": cfg.stage, "head": cfg.head, "seed": cfg.seed, "data_frac": cfg.data_frac,
                           "n_qubits": cfg.quantum.n_qubits, "n_layers": cfg.quantum.n_layers,
                           "quantum_backend": cfg.quantum.backend})
        {"finetune": finetune, "embed": embed, "head": train_head}[cfg.stage](cfg, device)


if __name__ == "__main__":
    main()
