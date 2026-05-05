# ============================================================
# app.py — RXT-J+ FastAPI Deployment
# Run with: python -m uvicorn app:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import torch
import joblib
import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn

DEVICE      = torch.device('cpu')
SEQ_LEN     = 8
CARDINALITY = 4

# ── Model architecture ───────────────────────────────────────────────────────

class ResNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cardinality=CARDINALITY):
        super().__init__()
        group_dim = out_dim // cardinality
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, group_dim), nn.BatchNorm1d(group_dim), nn.ReLU(),
                nn.Linear(group_dim, group_dim), nn.BatchNorm1d(group_dim)
            ) for _ in range(cardinality)
        ])
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(torch.cat([p(x) for p in self.paths], dim=-1) + self.shortcut(x))

class ResNeXtExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            ResNeXtBlock(input_dim, 128), ResNeXtBlock(128, 128),
            ResNeXtBlock(128, 64),        ResNeXtBlock(64,  64)
        )
    def forward(self, x): return self.net(x)

class SelfAttentionGRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len  = seq_len
        self.step_dim = input_dim // seq_len
        self.gru      = nn.GRU(self.step_dim, hidden_dim, num_layers=2,
                               batch_first=True, dropout=0.3)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(),
                                       nn.Linear(32, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(32, 1), nn.Identity())
    def forward(self, x):
        x      = x.view(x.size(0), self.seq_len, self.step_dim)
        out, _ = self.gru(x)
        attn   = torch.softmax(self.attention(out), dim=1)
        ctx    = (attn * out).sum(dim=1)
        return self.classifier(ctx).squeeze(1), attn.squeeze(-1)

class AttentionRXTJ(nn.Module):
    def __init__(self, input_dim, seq_len=SEQ_LEN):
        super().__init__()
        self.resnext  = ResNeXtExtractor(input_dim)
        self.attn_gru = SelfAttentionGRU(input_dim=64, seq_len=seq_len)
    def forward(self, x):
        return self.attn_gru(self.resnext(x))


# ── Load artefacts ────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))

def load_models():
    config   = json.load(open(os.path.join(BASE, 'results/deployment_config.json')))
    nystroem = joblib.load(os.path.join(BASE, 'models/nystroem.pkl'))
    ipca     = joblib.load(os.path.join(BASE, 'models/incremental_pca.pkl'))
    ifm      = joblib.load(os.path.join(BASE, 'models/isolation_forest.pkl'))

    input_dim = ipca.n_components_
    model     = AttentionRXTJ(input_dim).to(DEVICE)
    state     = torch.load(os.path.join(BASE, 'models/attention_rxtj.pt'),
                           map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, nystroem, ipca, ifm, config

print("Loading RXT-J+ models...")
model, nystroem, ipca, ifm, config = load_models()

W_MODEL   = config['W_MODEL']
W_IFM     = config['W_IFM']
THRESHOLD = config['THRESHOLD']
print(f"Models loaded. W_MODEL={W_MODEL:.4f}, W_IFM={W_IFM:.4f}, Threshold={THRESHOLD:.2f}")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RXT-J+ Fraud Risk Scoring API",
    description="Real-time payment fraud detection — Attention-RXT-J+ with Jaya optimisation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],   # null allows file:// access from demo HTML
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_001")
    features: List[float] = Field(..., description="Preprocessed feature vector")

class DirectScoreRequest(BaseModel):
    transaction_id: str
    features: List[float] = Field(..., description="50-dim EARN+ feature vector (post-IPCA)")

class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]

class RiskResponse(BaseModel):
    transaction_id: str
    risk_score:  float
    decision:    str
    confidence:  float
    latency_ms:  float


# ── Core scorer ───────────────────────────────────────────────────────────────

def score_earn_features(Z_earn: np.ndarray):
    """Score 50-dim EARN+ features directly — no Nystroem/IPCA needed."""
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(Z_earn).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()

    ifm_raw  = ifm.score_samples(Z_earn)
    ifm_min, ifm_max = ifm_raw.min(), ifm_raw.max()
    ifm_norm = 1 - (ifm_raw - ifm_min) / (ifm_max - ifm_min + 1e-9)
    ifm_norm = np.clip(ifm_norm, 0, 1)

    total = W_MODEL + W_IFM + 1e-9
    risk  = np.clip((W_MODEL / total) * probs + (W_IFM / total) * ifm_norm, 0, 1)
    preds = (risk > THRESHOLD).astype(int)
    conf  = np.where(preds == 1, risk, 1 - risk)
    return risk, preds, conf

def score_raw_features(features_np: np.ndarray):
    """Score raw features — runs full Nystroem + IPCA pipeline first."""
    Z_nys  = nystroem.transform(features_np)
    Z_earn = ipca.transform(Z_nys)
    return score_earn_features(Z_earn)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "RXT-J+ Fraud Risk Scoring API",
        "version": "1.0.0",
        "status":  "live",
        "model_auc": config['final_auc'],
        "model_mcc": config['final_mcc'],
        "latency_ms": config['latency_ms']
    }

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.get("/model/info")
def model_info():
    return {
        "weights":     {"W_MODEL": W_MODEL, "W_IFM": W_IFM},
        "threshold":   THRESHOLD,
        "performance": {
            "auc": config['final_auc'],
            "mcc": config['final_mcc'],
            "false_positives": config['false_positives'],
            "false_negatives": config['false_negatives'],
        },
        "speed": {
            "latency_ms":     config['latency_ms'],
            "throughput_tps": config['throughput_tps']
        }
    }

@app.get("/demo-samples")
def demo_samples():
    """Serve real test samples for the web demo (from Z_test_earn.npy)."""
    path = os.path.join(BASE, 'data', 'demo_samples.json')
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="demo_samples.json not found — run save_demo_samples.py first"
        )
    with open(path, 'r') as f:
        return JSONResponse(content=json.load(f))

@app.post("/score", response_model=RiskResponse)
def score_single(req: TransactionRequest):
    """Score a single transaction (full pipeline — raw features → Nystroem → IPCA → model)."""
    t0 = time.time()
    try:
        features_np = np.array(req.features, dtype=np.float32).reshape(1, -1)
        risk, preds, conf = score_raw_features(features_np)
        latency = (time.time() - t0) * 1000
        return RiskResponse(
            transaction_id = req.transaction_id,
            risk_score     = round(float(risk[0]), 4),
            decision       = "FRAUD" if preds[0] == 1 else "LEGIT",
            confidence     = round(float(conf[0]), 4),
            latency_ms     = round(latency, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/direct")
def score_direct(req: DirectScoreRequest):
    """Score using 50-dim EARN+ features directly (post-IPCA). Used by web demo."""
    t0 = time.time()
    try:
        Z_earn = np.array(req.features, dtype=np.float32).reshape(1, -1)

        if Z_earn.shape[1] != ipca.n_components_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {ipca.n_components_} features, got {Z_earn.shape[1]}"
            )

        risk, preds, conf = score_earn_features(Z_earn)
        latency = (time.time() - t0) * 1000

        return {
            "transaction_id": req.transaction_id,
            "risk_score":     round(float(risk[0]), 4),
            "decision":       "FRAUD" if preds[0] == 1 else "LEGIT",
            "confidence":     round(float(conf[0]), 4),
            "latency_ms":     round(latency, 4),
            "model_version":  "rxtj_plus_v1"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/batch")
def score_batch(req: BatchRequest):
    """Score up to 1000 transactions at once."""
    if len(req.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transactions per batch")
    t0 = time.time()
    try:
        ids         = [t.transaction_id for t in req.transactions]
        features_np = np.array([t.features for t in req.transactions], dtype=np.float32)
        risk, preds, conf = score_raw_features(features_np)
        total_latency = (time.time() - t0) * 1000

        results = [
            {
                "transaction_id": ids[i],
                "risk_score":     round(float(risk[i]), 4),
                "decision":       "FRAUD" if preds[i] == 1 else "LEGIT",
                "confidence":     round(float(conf[i]), 4),
                "latency_ms":     round(total_latency / len(ids), 4)
            }
            for i in range(len(ids))
        ]
        return {
            "results":          results,
            "total_latency_ms": round(total_latency, 2),
            "fraud_count":      int(preds.sum()),
            "legit_count":      int((preds == 0).sum())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
