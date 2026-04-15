# 🔍 Fraud Detection Pipeline

**Author:** Vaibhav Gupta | IIT Kanpur & IIT Delhi Certified  
**Stack:** Python · XGBoost · Random Forest · SMOTE · SHAP · SQL · Pandas

---

## Overview
End-to-end fraud detection system achieving:
- **95% Precision** on fraud class
- Reduced false positives via optimised threshold tuning
- Full SHAP explainability for each prediction
- SQL analytics layer for business reporting

## Architecture
```
Raw Transactions
      ↓
Feature Engineering (velocity, time, distance, combos)
      ↓
SMOTE Oversampling (handles class imbalance)
      ↓
Ensemble Model [XGBoost (3x) + Random Forest (2x) + Logistic Regression (1x)]
      ↓
Threshold Optimisation (PR-AUC maximised)
      ↓
SHAP Explanation + SQL Reporting
```

## Project Structure
```
2_fraud_detection/
├── fraud_pipeline.py   # Full ML pipeline
├── fraud_queries.sql   # 8 production SQL analytics queries
├── requirements.txt
└── models/             # Saved model (auto-created after training)
```

## Quick Start
```bash
pip install -r requirements.txt
python fraud_pipeline.py
```

## Key Results
| Metric    | Value  |
|-----------|--------|
| Precision | 95%    |
| ROC-AUC   | ~0.98  |
| PR-AUC    | ~0.87  |

## Inference (single transaction)
```python
from fraud_pipeline import predict_transaction

result = predict_transaction({
    "amount": 5000, "hour_of_day": 2, "is_international": 1,
    "num_transactions_1h": 12, "card_age_days": 5,
    "distance_from_home": 800, "merchant_category": 15,
})
# → {"fraud_probability": 0.92, "is_fraud": True, "risk_level": "HIGH"}
```