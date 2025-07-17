# 🔐 Aave Credit Score Prediction

A machine learning pipeline that estimates the creditworthiness of DeFi users on the Aave protocol using on-chain behavioral data. This system generates synthetic credit scores from wallet transaction histories to reduce overcollateralization and promote trust in decentralized lending.

## 🚀 Features

- 🧾 Aggregates and engineers wallet features from Aave transaction logs
- 📊 Trains predictive models (Random Forest, XGBoost) on synthetic credit scores
- 🧠 Explains predictions with SHAP interpretability plots
- 📈 Analyzes wallet behavior based on credit score brackets
- 🧮 Saves pre-trained models and wallet score predictions

---

## 📂 Project Structure

```
aave-credit-score/
├── data/
│ ├── raw/                          # Raw JSON user transaction data
│ └── processed/                    # Cleaned feature data (CSV)
│
├── outputs/
│ ├── models/                       # Trained Random Forest & XGBoost models
│ ├── scores/                       # Predicted scores + visualizations
│ └── shap/                         # SHAP explainability plots
│
├── src/
│ ├── aggregator.py                 # Feature aggregation logic
│ ├── scorer.py                     # Predict credit scores using models
│ ├── train_model.py                # Train and save models
│ ├── json_loader.py                # Efficient JSON loader
│ ├── explain_model_shap.py         # SHAP feature importance visualizer
│ ├── analyze_scored_wallets.py     # Score distribution & behavior analysis
│ └── generate_scores.py            # 🔁 Main pipeline (run this!)
│
├── notebooks/                      # Jupyter-based exploration
│ ├── 01_data_exploration.ipynb
│ ├── 02_data_aggregation.ipynb
│ └── 03_model_evaluation.ipynb
│
├── requirements.txt
├── README.md
└── analysis.md
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **ML Frameworks**: Scikit-learn, XGBoost
- **Data Handling**: Pandas, NumPy, ijson
- **Visualization**: Matplotlib, Seaborn, SHAP
- **Model Serialization**: `joblib`
- **Data Source**: Aave user wallet transaction logs (via local JSON)

---

## 📈 Workflow Overview

1. **Feature Extraction**  
   `aggregator.py`: Processes wallet-level transaction patterns (borrow, deposit, repay) from JSON logs and saves a feature matrix.

2. **Model Training**  
   `train_model.py`: Trains Random Forest and XGBoost regressors on synthetically computed credit scores and evaluates performance.

3. **Prediction & Scoring**  
   `generate_scores.py`: Predicts credit scores, explains predictions with SHAP, and visualizes score distribution across wallets.

4. **Behavior Analysis**  
   `analyze_scored_wallets.py`: Identifies behavioral patterns among high vs. low score users.

---

## 🧪 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your Aave transaction JSON file at:  
`data/raw/user_transactions.json`

Then extract features:

```bash
python src/aggregator.py
```

### 3. Train Models

```bash
python src/train_model.py
```

This creates `random_forest_model.pkl` and `xgboost_model.pkl` in `outputs/models/`.

### 4. Generate Scores & Analyze

```bash
python src/generate_scores.py
```

This:
- Prompts for model choice
- Predicts and saves wallet scores
- Generates SHAP plots
- Displays distribution and stats

### 5. Analyze Separately (Optional)

```bash
python src/analyze_scored_wallets.py
```

---

## 🧠 Key Features Used

- `repayment_efficiency = total_rapay / total_borrow`
- `total_repay`
- `num_repay`
- `deposit_frequency = num_deposit / avg_tx_gap_sec`
- `borrow_to_deposit_ratio = tota_borrow / total_deposit`
- `total_deposit`
- `net_borrow = total_borrow - total_repay`
- `activity_score = total_transaction / avg_tx_gap_secc`

---
## feature Importance chart (XGBoost model)
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/d1cf2221-c051-4d61-9f52-48dbad81deab" />

## feature Importance chart (Random Forest model)
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f6d9a083-0fca-4b5a-846d-be8da5af7a86" />

---
## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Ritik Chauhan**  
🌐 [GitHub](https://github.com/ritikchauhan2004)  
📬 Feel free to fork or raise issues to contribute!

---