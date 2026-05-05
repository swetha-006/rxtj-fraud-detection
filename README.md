# 🛡️ ML Transaction Risk Scoring for Payment Fraud Detection

**RXT-J+ Model** — ResNeXt-Embedded GRU with Jaya Optimization for Real-Time Fraud Detection

📌 Overview

This project presents an **end-to-end Machine Learning transaction risk scoring system** designed to detect payment fraud in real time. Built as a Capstone Project by Team 13, Department of CSE (CYS), SKCT, the system leverages a novel deep learning architecture called **RXT-J+** — combining **ResNeXt feature extraction**, a **Self-Attention GRU classifier**, and **Jaya-optimized ensemble weighting** — to achieve sub-millisecond fraud detection with high accuracy and minimal false positives.
The system is deployed as a **FastAPI REST backend** with an interactive **HTML dashboard frontend**, supporting live transaction scoring and behavioral anomaly detection.

---

## ✨ Key Results

| Metric | Value |
|---|---|
| **AUC-ROC** | 96.24% |
| **MCC (Matthews Correlation)** | 0.8426 |
| **Throughput** | ~51,351 transactions/second |
| **Latency** | 0.019 ms per transaction |
| **Optimal Threshold** | 0.50 |
| **False Positive Rate** | ~6.66% |
| **Dataset** | IEEE-CIS Fraud Detection (660K transactions) |

---

## 🏗️ Architecture

```
Transaction Input
       │
       ▼
┌─────────────────────┐
│  EARN Ensembler     │  ← High + Low dimensional feature extraction
│  (KernelPCA +       │
│   Nystroem +        │
│   IncrementalPCA)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ResNeXt Extractor  │  ← 4-cardinality grouped convolutions
│  (4 blocks: 128→64) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Self-Attention GRU │  ← Sequential pattern + attention weights
│  (2-layer, hidden=64)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Jaya Optimizer     │  ← Ensemble weight tuning (no hyperparameters)
│  W_model=0.534      │
│  W_IFM=0.466        │
└────────┬────────────┘
         │
         ▼
   Risk Score (0–1)
   + Fraud / Legit Label
```

The model also incorporates:
- **Isolation Forest Module (IFM)** — unsupervised anomaly scoring
- **Autoencoder** — reconstruction loss for behavioral drift detection
- **SMOTE** — synthetic minority oversampling for class balance

---

## 📁 Project Structure

```
rxtj_project/
├── app.py                        # FastAPI main deployment (full RXT-J+ pipeline)
├── backend.py                    # Lightweight FastAPI backend with SQLite history
├── index.html                    # Main frontend dashboard
├── rxtj_app.html                 # RXT-J transaction scoring UI
├── fraudshield_app.html          # FraudShield merchant risk UI
├── requirements.txt              # Python dependencies
│
├── models/                       # Trained model artifacts
│   ├── attention_rxtj.pt         # RXT-J+ attention GRU weights
│   ├── resnet_extractor.pt       # ResNeXt feature extractor
│   ├── autoencoder.pt            # Autoencoder for anomaly detection
│   ├── isolation_forest.pkl      # Isolation Forest model
│   ├── kernel_pca.pkl            # KernelPCA transformer
│   ├── incremental_pca.pkl       # IncrementalPCA transformer
│   ├── nystroem.pkl              # Nystroem approximation
│   ├── scaler.pkl                # Feature scaler
│   ├── imputer.pkl               # Missing value imputer
│   └── jaya_optimal_weights.pkl  # Jaya-optimized ensemble weights
│
├── notebooks/                    # Training notebooks (step-by-step pipeline)
│   ├── 01_preprocessing.ipynb
│   ├── 02_balancing.ipynb
│   ├── 03_earn_features.ipynb
│   ├── 04_attention_rxtj.ipynb
│   ├── 05_jaya_optimize.ipynb
│   └── save_demo_samples.py
│
├── data/
│   ├── demo_samples.json         # Pre-loaded demo transactions
│   ├── IEEE CIS/                 # Raw dataset (not tracked in Git — see below)
│   ├── Cleaned/                  # Preprocessed balanced arrays
│   └── *.png                     # Training plots & visualizations
│
└── results/
    ├── deployment_config.json    # Final model config & performance metrics
    ├── confusion_matrix_final.png
    ├── roc_comparison.png
    ├── jaya_convergence.png
    └── threshold_optimisation.png
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/rxtj-fraud-detection.git
cd rxtj-fraud-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Backend API

```bash
# Full RXT-J+ pipeline (recommended)
python -m uvicorn app:app --reload --port 8000

# OR lightweight backend
python -m uvicorn backend:app --reload --port 8000
```

The API will be live at: `http://localhost:8000`

### 5. Open the Frontend

Open `index.html` in your browser (or use Live Server in VS Code). The dashboard connects to the FastAPI backend on port 8000.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status and model info |
| `POST` | `/predict` | Score a single transaction |
| `POST` | `/predict_batch` | Score a batch of transactions |
| `GET` | `/demo` | Returns pre-loaded demo samples |
| `GET` | `/stats` | System performance stats |
| `GET` | `/history` | Transaction prediction history (SQLite) |

### Sample Request — `/predict`

```json
POST /predict
Content-Type: application/json

{
  "TransactionAmt": 150.0,
  "card1": 12345,
  "card2": 321.0,
  "addr1": 315.0,
  "addr2": 87.0,
  "dist1": 19.0,
  "P_emaildomain": 0,
  "R_emaildomain": 0
}
```

### Sample Response

```json
{
  "transaction_id": "TXN-20260305-001",
  "risk_score": 0.823,
  "label": "FRAUD",
  "model_prob": 0.791,
  "ifm_score": 0.612,
  "latency_ms": 0.019,
  "attention_weights": [0.12, 0.18, ...]
}
```

---

## 📊 Training Pipeline

The training pipeline is split into 5 Jupyter notebooks:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_preprocessing.ipynb` | Merge identity + transaction CSVs, handle nulls, encode categoricals |
| 2 | `02_balancing.ipynb` | SMOTE + undersampling to balance 3.5% fraud ratio |
| 3 | `03_earn_features.ipynb` | EARN Ensembler: KernelPCA + Nystroem + IncrementalPCA feature fusion |
| 4 | `04_attention_rxtj.ipynb` | Train ResNeXt extractor + Self-Attention GRU classifier |
| 5 | `05_jaya_optimize.ipynb` | Jaya optimization to find optimal RXT-J + IFM ensemble weights |

---

## 📦 Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset from Kaggle.

> ⚠️ The raw dataset files (`data/IEEE CIS/`) are **not included** in this repository due to size (>1.3 GB). Download them from:
>
> 🔗 [Kaggle: IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)

Place the downloaded files in: `data/IEEE CIS/`
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

The preprocessed balanced arrays (`data/Cleaned/X_balanced.npy`, `y_balanced.npy`) are also excluded from Git due to size (~700 MB).

---

## 🧪 Technologies Used

| Category | Tools |
|---|---|
| **ML Framework** | PyTorch 2.2.2 |
| **API Backend** | FastAPI 0.110.0, Uvicorn |
| **Data Processing** | NumPy, scikit-learn, SMOTE |
| **Streaming** | kafka-python |
| **Storage** | SQLite (transaction history) |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Validation** | Pydantic 2.6.4 |

---


## 📄 Base Paper

> *"Online Payment Fraud Detection Model Using Machine Learning Techniques"*
>
> RXT-J Model — ResNeXt-Embedded GRU with Jaya Optimization.
> Evaluated on IEEE-CIS, PaySim, and European transaction datasets.
> Achieved **98% accuracy**, outperforming existing models by 10–18%.

---

## 📜 License

This project is developed for academic purposes under the B.E. Capstone Project curriculum (NEP 2020), Sri Krishna College of Technology.

---

> *This project aligns with NEP 2020 goals of innovation, sustainability, and IKS integration.*
