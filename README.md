# CardioAgent-Ops
An end-to-end containerized medical decision platform powered by microservices. Features audio signal processing (MLOps), a fine-tuned LLM, a RAG system, and an autonomous AI agent network, fully automated with CI/CD/CT pipelines.

```
.wav ─> api-gateway ─> audio-engine ──────────────> prediction (normal / abnormal)
          (5)            (3)  └─ model "production" from the MLflow registry
                                                         │
                         agent-orchestrator (4) <────────┘
                           1. LLM writes a search query
                           2. rag-engine ─> vector-db (2): ESC guideline passages
                           3. LLM writes the clinical note, citing the passages
                           └─ every step traced in MLflow
```

| Service | URL |
|---|---|
| api-gateway | http://localhost:8000/docs |
| audio-engine | http://localhost:8002/docs |
| rag-engine | http://localhost:8003/docs |
| agent-orchestrator | http://localhost:8004/docs |
| vector-db (Qdrant) | http://localhost:6333/dashboard |
| MLflow | http://localhost:5000 |

## Run
```bash
docker compose up --build -d
curl -X POST http://localhost:8003/ingest        # index the ESC guidelines once
```
The first start downloads the LLM (`llama3.2:3b`, ~2 GB): `docker compose logs -f llm-pull`.

## MLflow
The tracking server is built from [mlflow-server/](mlflow-server/) (SQLite +
artifacts in `mlflow-server/mlflow-data/`). It runs with the platform, or alone
with `cd mlflow-server && docker compose up -d`; both use the same data folder.

MLflow is used for two things:

**1. Comparing the 5 heart-sound models** (experiment `heart-sound-classification`)

| Family | Model | Input |
|---|---|---|
| ML | Random Forest, SVM, XGBoost | 26 features: mean + std of the 13 MFCCs |
| DL | 1D-CNN | raw signal (5 s at 2 kHz) |
| DL | 2D-CNN | MFCC matrix (13 × 20) |

The dataset (`ourdataset/ourdataset/train/{normal,abnormal}`, not in git) is
evaluated with **3-fold cross-validation**: each fold trains on 66% of the
patients and tests on the other 34%. The recordings of one patient
(`112_..._A.wav`, `112_..._B.wav`...) always stay in the same fold, so a
model is never tested on a patient it was trained on.

```bash
docker compose up -d mlflow
docker compose run --rm audio-engine python train.py data --promote
# quick test: --models svm cnn_1d --epochs 2
```
Without Docker: `cd audio-engine && pip install -r requirements.txt`, then
`MLFLOW_TRACKING_URI=http://localhost:5000 python train.py ../ourdataset/ourdataset/train --promote`.

In MLflow, the `comparison` run contains one child run per model, with the
metrics of every fold and their mean/std (`macc`, `sensitivity`,
`specificity`, `accuracy`, `f1`, `roc_auc`) and `comparison.md`.
Select the 5 child runs → **Compare** to get the charts.

**Benchmark report:** `python audio-engine/benchmark/generate_report.py` reads
the last comparison from MLflow and writes the table, confusion matrices, ROC
curves and training curves to
[audio-engine/benchmark/results/report.md](audio-engine/benchmark/results/report.md).

The best model by **MAcc** = (sensitivity + specificity) / 2 (the PhysioNet
2016 score, robust to the 77% / 23% class imbalance) is retrained on the whole
dataset and registered as `heart-sound-classifier`. `--promote` sets its
`production` alias, which the audio-engine serves. After promoting another
version (CLI or MLflow UI): `curl -X POST http://localhost:8002/model/reload`.

**2. Tracing the LLM agent** (experiment `agent-orchestrator`, tab *Traces*)

Every `/report` call is one trace: search query written by the LLM →
passages returned by the RAG → final note, with the prompts, outputs,
latency and token counts. Traces are tagged with `prompt_version`,
`llm_model` and the audio model version that made the prediction.

## Analyze a recording
```bash
curl -X POST http://localhost:8000/analyze_pcg \
  -F "file=@audio-engine/test.wav" \
  -F "patient=65 year old man, shortness of breath on exertion"
```
