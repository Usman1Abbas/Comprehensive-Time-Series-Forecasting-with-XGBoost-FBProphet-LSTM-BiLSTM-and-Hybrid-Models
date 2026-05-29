# 📈 Comprehensive Time-Series Forecasting

A comparative study benchmarking five approaches to time-series forecasting on the same problem — from gradient-boosted trees to deep sequence models to hybrids — so the trade-offs between them are visible side by side.

## Models compared
| Notebook | Model | Family |
|---|---|---|
| `XGBoost.ipynb` | XGBoost (with engineered lag/calendar features) | Gradient boosting |
| `FBProphet.ipynb` | Facebook Prophet | Additive / decomposition |
| `LSTM.ipynb` | LSTM | Deep recurrent |
| `BiLSTM.ipynb` | Bidirectional LSTM | Deep recurrent |
| *(hybrid)* | Hybrid (statistical + neural) | Ensemble |

## What this demonstrates
- **Feature engineering** for tabular forecasting (lags, rolling stats, calendar features) to make a tree model competitive on sequential data.
- **Classical vs. deep** trade-offs: Prophet's interpretability and seasonality handling vs. LSTM/BiLSTM's capacity to learn nonlinear temporal dependencies.
- **Hybrid modeling** — combining strengths to beat any single model.
- Consistent **evaluation** (e.g. MAE / RMSE / MAPE) across models for a fair comparison.

## Stack
Python · pandas · NumPy · scikit-learn · XGBoost · Prophet · TensorFlow/Keras · Matplotlib · Jupyter

## How to use
Open any notebook in Jupyter and run top-to-bottom. Each is self-contained: data prep → model → forecast → evaluation → plots.
