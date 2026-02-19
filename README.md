# Coconut Auction Price Predictor

ML-based price prediction system with Streamlit web interface and Docker deployment.

---

## 📁 Project Structure

```
├── model/                      # ML training
│   ├── train_model.py         # Train XGBoost model
│   ├── requirements.txt       # Training dependencies
│   └── outputs/               # Model artifacts (generated)
│
├── app/                        # Web application
│   ├── app.py                 # Streamlit interface
│   ├── Dockerfile             # Container configuration
│   ├── deployment_requirements.txt
│   ├── run_app.bat            # Windows launcher
│   └── run_app.sh             # Linux/Mac launcher
│
├── data_preprocessor/          # Data cleaning
└── srilanka_coconut_auction_data_cleaned.csv
```

---

## 🚀 Quick Setup

### 1. Train Model

```bash
cd model
pip install -r requirements.txt
python train_model.py
```

### 2. Run App

**Windows:**

```bash
cd app
run_app.bat
```

**Linux/Mac:**

```bash
cd app
chmod +x run_app.sh
./run_app.sh
```

**Access:** http://localhost:8501

---

## 🐳 Docker Deployment

### Build & Run

```bash
# Build from project root (not from app folder)
docker build -f app/Dockerfile -t coconut-predictor .
docker run -p 8501:8501 coconut-predictor
```

**Note:** Train model first before building Docker image.

---

## 📊 Features

- XGBoost regression model with lag features
- Interactive Streamlit web interface
- Feature importance visualization
- Auto-calculated derived features
- Docker containerization

---

## 🛠️ Tech Stack

- Python 3.9+
- XGBoost
- Streamlit
- pandas, numpy, scikit-learn
- Docker
