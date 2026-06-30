# 📊 RetailPulse – AI-Powered Customer Analytics & Demand Forecasting

> An end-to-end retail analytics platform built during the **Zidio Development Data Science Internship** (June–September 2026).

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Prophet](https://img.shields.io/badge/Prophet-1.1-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 What Is RetailPulse?

RetailPulse transforms raw e-commerce transaction data into actionable business intelligence through five integrated ML modules — all accessible via a single interactive Streamlit dashboard.

**Business problems it solves:**

| Problem | RetailPulse Module |
|---------|-------------------|
| "Which customers are most valuable?" | RFM + K-Means Segmentation |
| "Who is about to churn?" | XGBoost Churn Prediction |
| "How much will we sell next month?" | Prophet Demand Forecasting |
| "Which products need reordering?" | EOQ Inventory Optimisation |
| "How is revenue trending?" | Sales Dashboard |

---

## ✨ Features

- **📊 Sales Dashboard** — Revenue trends, top products, country breakdown, day-of-week heatmap
- **👥 Customer Segmentation** — RFM scoring + K-Means clustering into 6 business segments (Champions, Loyal, At-Risk, etc.) with 3D scatter visualisation
- **⚠️ Churn Prediction** — XGBoost classifier with risk scores, threshold slider, and feature importance chart
- **📈 Demand Forecasting** — Facebook Prophet with 30-day horizon, confidence intervals, and What-If demand adjustment
- **📦 Inventory Optimisation** — Economic Order Quantity (EOQ) + safety stock calculator with live What-If controls

---

## 🗂 Project Structure

```
RetailPulse/
│
├── app.py                    # 🖥  Streamlit dashboard (7 pages)
├── run_pipeline.py           # ⚙️  End-to-end ML pipeline runner
├── config.py                 # 🔧  Central configuration (paths, hyperparameters)
├── requirements.txt          # 📦  Python dependencies
├── Dockerfile                # 🐳  Container deployment
├── .gitignore
│
├── src/                      # 🐍  Core Python modules
│   ├── data_preprocessing.py   # Load, clean, RFM features, daily sales
│   ├── feature_engineering.py  # Churn feature construction
│   ├── segmentation.py         # K-Means customer segmentation
│   ├── churn_prediction.py     # XGBoost churn model
│   ├── demand_forecasting.py   # Prophet demand forecast
│   ├── inventory.py            # EOQ inventory optimisation
│   └── utils.py                # Shared helper functions
│
├── notebooks/
│   └── RetailPulse_EDA.ipynb   # 📓  Exploratory Data Analysis
│
├── tests/
│   └── test_pipeline.py        # ✅  Pytest unit tests (7 test classes)
│
├── data/
│   ├── raw/                    # 📥  Input data (not tracked by git)
│   └── processed/              # 📤  Model outputs (not tracked by git)
│
├── models/                   # 🤖  Saved ML models (.pkl)
├── reports/                  # 📸  Generated charts (.png)
└── assets/                   # 🖼  Static assets
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/joneswesley/retailpulse.git
cd retailpulse
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `prophet` requires additional system dependencies. If installation fails, remove it from `requirements.txt` — the pipeline falls back to a linear forecast model automatically.

---

## 🏃 Usage

### Quick Start (with synthetic demo data)

```bash
# Step 1 — Generate data + train all models
python run_pipeline.py

# Step 2 — Launch the dashboard
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

You can also click the **"▶ Load Demo Data"** button directly in the dashboard without running the pipeline.

### Using the Real Dataset

1. Download the [Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) from the UCI ML Repository
2. Save it as `data/raw/online_retail.csv`
3. Run the pipeline with the `--real-data` flag:

```bash
python run_pipeline.py --real-data
```

### Pipeline Flags

```bash
python run_pipeline.py --real-data      # Use real dataset instead of synthetic
python run_pipeline.py --skip-churn     # Skip XGBoost (if not installed)
python run_pipeline.py --skip-forecast  # Skip Prophet (uses linear fallback)
```

### Running Individual Modules

```bash
python src/data_preprocessing.py   # Generate cleaned data + RFM
python src/segmentation.py         # K-Means customer segmentation
python src/feature_engineering.py  # Build churn features
python src/churn_prediction.py     # Train XGBoost churn model
python src/demand_forecasting.py   # Train Prophet forecast
python src/inventory.py            # Compute inventory recommendations
```

---

## 🧪 Testing

```bash
pytest tests/test_pipeline.py -v
```

**Expected output: 20+ tests, all passing**

Test coverage:
- Data preprocessing (cleaning, RFM, daily sales)
- Customer segmentation (K-Means, silhouette, cluster labels)
- Feature engineering (churn features, binary labels)
- Inventory optimisation (EOQ, safety stock, stock status)
- Utility functions (MAPE, currency formatting, safe CSV loading)

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, quick KPIs, getting started guide |
| 📊 Sales Dashboard | Revenue trends with date filter, top products, country map, day-of-week heatmap |
| 👥 Customer Segmentation | Segment distribution pie, RFM histograms, 3D scatter, segment summary table |
| 📈 Demand Forecast | Prophet forecast chart, weekly breakdown, What-If demand adjustment slider |
| ⚠️ Churn Prediction | Risk distribution, probability histogram, threshold slider, high-risk customer table |
| 📦 Inventory | Stock status pie, scatter plot, reorder recommendations table, download CSV |
| ℹ️ About | Tech stack, model targets, author info, deployment instructions |

---

## 📈 ML Models & Performance Targets

| Model | Algorithm | Metric | Target |
|-------|-----------|--------|--------|
| Customer Segmentation | K-Means | Silhouette Score | ≥ 0.35 |
| Demand Forecasting | Prophet (+ linear fallback) | MAPE | ≤ 12% |
| Churn Prediction | XGBoost | AUC-ROC | ≥ 0.88 |
| Churn Prediction | XGBoost | Precision@20% | ≥ 0.75 |
| Inventory | EOQ Formula | — | Service Level ≥ 95% |

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML | Scikit-learn, XGBoost |
| Forecasting | Prophet, statsmodels |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, NumPy |
| Persistence | Joblib |
| Notebook | Jupyter |
| Testing | Pytest |
| Deployment | Docker, Streamlit Cloud |

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t retailpulse .

# Run the container
docker run -p 8501:8501 retailpulse

# Open the dashboard
# http://localhost:8501
```

---

## ☁️ Streamlit Cloud Deployment

1. Push this repository to GitHub (public repo)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → select your repo → set **Main file** to `app.py`
4. Click **Deploy**

The app generates synthetic demo data on first load — no external data file required.

---

## 📸 Screenshots

| Sales Dashboard | Customer Segmentation |
|---|---|
| *(screenshot placeholder)* | *(screenshot placeholder)* |

| Demand Forecast | Inventory Dashboard |
|---|---|
| *(screenshot placeholder)* | *(screenshot placeholder)* |

---

## 🔮 Future Improvements

- [ ] Add SHAP explainability charts for the churn model
- [ ] Integrate real-time data via a REST API or database connection
- [ ] Add email alert system for low-stock products
- [ ] Implement A/B testing framework for promotional campaigns
- [ ] Add LSTM ensemble model alongside Prophet
- [ ] User authentication for multi-tenant deployments
- [ ] Export reports as PDF

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Bhavani Nalajala**  
Data Science & Analytics Intern — Zidio Development (2026)

- 🌐 [LinkedIn](https://linkedin.com/in/your-linkedin-username)
- 🐙 [GitHub](https://github.com/your-github-username)

---

*Built with ❤️ as part of the Zidio Development Data Science Internship.*
