# Predictive Maintenance & Health Index Framework
## High-Speed Precision Machining — Automotive Powertrain Components

End-to-end pipeline: physics-informed synthetic sensor generation -> mechanical
feature engineering -> XGBoost (RUL regression + tool wear state classification)
-> SHAP interpretability -> OEE/ROI business impact translation.

## Run order
```
pip install -r requirements.txt
python src/01_generate_synthetic_data.py   # sensor telemetry, 45 machines, run-to-failure
python src/02_feature_engineering.py       # physics-based feature engineering
python src/03_train_model.py               # XGBoost + hyperparameter tuning + SHAP
python src/04_business_impact.py           # OEE impact, cost-benefit, health index
```

## Outputs
- `data/` — raw synthetic sensor stream + engineered feature set (CSV)
- `outputs/figures/` — RUL prediction plot, confusion matrix, SHAP summary plots,
  OEE impact bridge chart, health index trajectory
- `outputs/reports/model_results_summary.json` — model metrics + SHAP rankings
- `outputs/reports/oee_business_impact.json` — OEE/cost-benefit numbers
- `outputs/reports/Predictive_Maintenance_Executive_Report.docx` — full executive report

## Key results (this run)
- RUL Regressor MAE: 13.82 cycles (R2=0.5861)
- Tool State Classifier Accuracy: 99.9%
- OEE improvement: +7.52 pp (82.71% -> 90.22%)
- Projected annual savings: Rs 14,308,800
