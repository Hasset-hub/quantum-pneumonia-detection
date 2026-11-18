"""Inference demo: upload a chest X-ray, get a pneumonia probability.

    MODEL_HEAD=artifacts/head_quantum_q4_l2_f1.0_s0.pt uvicorn app.main:app
"""
import io
import os
import re

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from qpd.data import build_transforms
from qpd.models import QuantumHead, build_backbone
from qpd.models.backbone import EMBED_DIM

ART = os.environ.get("ARTIFACTS", "artifacts")
HEAD_PATH = os.environ.get("MODEL_HEAD", f"{ART}/head_quantum_q4_l2_f1.0_s0.pt")
n_qubits, n_layers = map(int, re.search(r"_q(\d+)_l(\d+)_", HEAD_PATH).groups())

backbone = build_backbone(pretrained=False)
backbone.load_state_dict(torch.load(f"{ART}/backbone.pt", map_location="cpu"))
backbone.eval()
head = QuantumHead(EMBED_DIM, n_qubits, n_layers, backend="local")
head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu"))
head.eval()
tf = build_transforms(train=False)

app = FastAPI(title="Quantum Pneumonia Detection", description="Research demo. Not a medical device.")


@app.get("/health")
def health():
    return {"status": "ok", "n_qubits": n_qubits, "n_layers": n_layers}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("L").resize((128, 128))
    with torch.no_grad():
        p = torch.sigmoid(head(backbone(tf(img).unsqueeze(0)))).item()
    return {"pneumonia_probability": round(p, 4), "label": "pneumonia" if p >= 0.5 else "normal"}
