import os
import sqlite3
import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. ARCHITECTURE ---
class ResNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cardinality=4):
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

class AttentionRXTJ(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.resnext = nn.ModuleDict({
            "net": nn.Sequential(
                ResNeXtBlock(input_dim, 128), ResNeXtBlock(128, 128),
                ResNeXtBlock(128, 64), ResNeXtBlock(64, 64)
            )
        })
        self.attn_gru = nn.ModuleDict({
            "gru": nn.GRU(64, 32, num_layers=2, batch_first=True),
            "attention": nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)),
            "classifier": nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        })
    def forward(self, x):
        x = self.resnext["net"](x)
        gru_out, _ = self.attn_gru["gru"](x.unsqueeze(1))
        return self.attn_gru["classifier"](gru_out[:, -1, :]), None

# --- 2. INITIALIZATION ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def init_db():
    conn = sqlite3.connect("fraud_storage.db")
    conn.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, tid TEXT, amt REAL, score REAL, status TEXT)")
    conn.commit()
    conn.close()

init_db()

print("🚀 Loading RXT-J+ Deployment Engine...")
try:
    # Loading ALL artifacts from your screenshot
    imputer = joblib.load('models/imputer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    nystroem = joblib.load('models/nystroem.pkl')
    ipca = joblib.load('models/incremental_pca.pkl')
    
    model = AttentionRXTJ(input_dim=ipca.n_components_)
    model.load_state_dict(torch.load('models/attention_rxtj.pt', map_location='cpu'))
    model.eval()
    print("✅ All systems online.")
except Exception as e:
    print(f"❌ Initialization Failed: {e}")

# --- 3. THE PIPELINE ---
class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    is_foreign: bool
    customer_age: int

@app.post("/score/form")
async def score_transaction(req: TransactionRequest):
    try:
        # 1. Feature Construction
        raw_vals = [req.amount, 1.0 if req.is_foreign else 0.0, float(req.customer_age)]
        
        # --- CRITICAL: MATCH THIS NUMBER TO YOUR SCALER ---
        # If your terminal says "expected 30", change 128 to 30
        feature_count = 128 
        padded = raw_vals + [0.0] * (feature_count - len(raw_vals))
        X = np.array(padded).reshape(1, -1)

        # 2. Transformation Chain
        # This is usually where the 500 error happens
        X = imputer.transform(X)
        X = scaler.transform(X)
        X = nystroem.transform(X)
        X = ipca.transform(X)

        # 3. Model Inference
        with torch.no_grad():
            tensor_in = torch.FloatTensor(X)
            logits, _ = model(tensor_in)
            risk_score = torch.sigmoid(logits).item()

        decision = "FRAUD" if risk_score > 0.5 else "LEGIT"

        # 4. Database Logging
        conn = sqlite3.connect("fraud_storage.db")
        conn.execute("INSERT INTO history (tid, amt, score, status) VALUES (?, ?, ?, ?)",
                     (req.transaction_id, req.amount, round(risk_score, 4), decision))
        conn.commit()
        conn.close()

        return {"transaction_id": req.transaction_id, "risk_score": round(risk_score, 4), "decision": decision}

    except Exception as e:
        # This prints the REAL error to your terminal
        print(f"ERROR DURING INFERENCE: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/history")
async def get_history():
    conn = sqlite3.connect("fraud_storage.db")
    cursor = conn.cursor()
    cursor.execute("SELECT tid, amt, score, status FROM history ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return [{"tid": r[0], "amt": r[1], "score": r[2], "status": r[3]} for r in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)