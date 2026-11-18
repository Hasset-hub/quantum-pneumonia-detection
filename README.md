# Hybrid Quantum-Classical Pneumonia Detection

COSC 461/561 Project 2 · Hasset Admasu, Nathnael Abraham, Andres Iniguez

A ResNet-18 finetuned on chest X-rays compresses each image into a few features. A variational
quantum circuit, simulated with the [QuantumRings SDK](https://www.quantumrings.com/), then
classifies the image as **normal** or **pneumonia**.

## Research question

Can a variational quantum circuit, used as the classification head on top of a finetuned CNN,
match a classical head with the same number of trainable parameters for pneumonia detection,
and does it hold up better when training data is scarce?

The full specification is in [`docs/Project2_Specifications.tex`](docs/Project2_Specifications.tex).

## Pipeline

```
X-ray (128x128) -> ResNet-18 (finetuned, GPU) -> 512-d embedding (cached)
    -> Linear(512 -> n) -> pi * tanh -> RY angle encoding on n qubits
    -> L x [RY, RZ on each qubit + CNOT ring] -> <Z_i> per qubit -> Linear(n -> 1) -> P(pneumonia)
```

| Head | What it is |
|---|---|
| `logreg` | Baseline 1: logistic regression on the embedding |
| `mlp` | Baseline 2: same projection plus a small MLP, sized to match the quantum head's parameter count |
| `quantum` | Main approach: the variational circuit above |

The quantum head has two interchangeable backends:

- `local`: exact statevector simulation written in PyTorch. Differentiable and fast, needs no account. Use it for big sweeps and CI.
- `quantumrings`: the QuantumRings simulator with finite shots and parameter-shift gradients. Use it on small data fractions and to score the final test set (`quantum.eval_backend=quantumrings`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add your free QuantumRings key from portal.quantumrings.com
python scripts/check_quantumrings.py   # local and QuantumRings results should agree
```

## Running experiments

```bash
make data          # download PneumoniaMNIST and print class balance
make finetune      # finetune ResNet-18 (run this on the course GPU)
make embed         # cache embeddings for train/val/test
make heads         # main comparison: logreg vs mlp vs quantum, 3 seeds
make sweep-data    # Experiment 1: 1%, 5%, 25%, 100% of training data
make sweep-circuit # Experiment 2: qubits {4,6,8} x layers {1,2,4}
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse every run at http://localhost:5000
```

Single run with overrides:

```bash
python -m qpd.train stage=head head=quantum quantum.n_qubits=6 quantum.n_layers=2 data_frac=0.05 seed=1
```

Each head run writes a JSON file to `results/` and logs everything to MLflow.

## Repository layout

```
configs/config.yaml         Hydra config (every experiment knob)
src/qpd/data.py             PneumoniaMNIST loading, augmentation, stratified subsampling
src/qpd/models/             backbone, classical heads, quantum head
src/qpd/train.py            finetune / embed / head training stages
src/qpd/metrics.py          AUROC, F1, sensitivity, specificity, accuracy
scripts/check_quantumrings.py  cross-checks the two quantum backends
app/main.py                 FastAPI inference demo
tests/                      pytest suite run by GitHub Actions
notebooks/                  EDA notebook goes here
docs/                       project specification (LaTeX)
```

## Demo

```bash
make serve    # or: docker build -t qpd . && docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts qpd
curl -F file=@xray.png localhost:8000/predict
```

## Model card

- **Intended use:** coursework and research on hybrid quantum-classical models.
- **Not for clinical use.** The model is trained on a small pediatric dataset and has not been validated for diagnosis.
- **Data:** PneumoniaMNIST (MedMNIST v2), derived from the Kermany et al. pediatric chest X-ray dataset. About 74% of images are pneumonia, so AUROC is the primary metric.
