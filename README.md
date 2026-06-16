<p align="center">
  <img src="assets/banner.svg" alt="Time-Series Forecasting — benchmarking XGBoost, Prophet, LSTM, BiLSTM and hybrids" width="100%">
</p>

<h1 align="center">Comprehensive Time-Series Forecasting</h1>

<p align="center"><em>Five forecasting approaches, one dataset, side by side — so the trade-offs are visible.</em></p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-complete-1b9c85">
  <img alt="python" src="https://img.shields.io/badge/Python-3.10-0d1b2a">
  <img alt="xgboost" src="https://img.shields.io/badge/XGBoost-gradient%20boosting-125a4d">
  <img alt="prophet" src="https://img.shields.io/badge/Prophet-additive-125a4d">
  <img alt="keras" src="https://img.shields.io/badge/TensorFlow%2FKeras-LSTM%20%C2%B7%20BiLSTM-e9c46a">
</p>

**A comparative study that benchmarks five families of time-series models on the same problem — daily Air Quality Index (AQI) forecasting from a 2019–2024 series.** It pits **XGBoost** (gradient-boosted trees with engineered lag/calendar features) against **Facebook Prophet** (additive decomposition) and against **LSTM** and **BiLSTM** deep recurrent networks built in **TensorFlow/Keras**, then frames how their strengths combine into a **hybrid** view. Every model is scored with the same **MAE / RMSE / MAPE** metrics so the comparison is fair.

> The point isn't to crown one model — it's to make the classical-vs-deep, interpretable-vs-flexible trade-offs measurable on identical data.

---

## ✨ Models compared

| Notebook | Model | Family | What it brings |
|---|---|---|---|
| [`XGBoost.ipynb`](XGBoost.ipynb) | XGBoost | Gradient boosting | Engineered lag (7/14/30d), rolling means & calendar features make a tree model competitive on sequential data; tuned via `GridSearchCV` + `TimeSeriesSplit` |
| [`FBProphet.ipynb`](FBProphet.ipynb) | Facebook Prophet | Additive / decomposition | Built-in trend + seasonality + US-holiday effects, with hyperparameter search and cross-validation |
| [`LSTM.ipynb`](LSTM.ipynb) | LSTM | Deep recurrent | Learns nonlinear temporal dependencies from scaled sequences, with dropout + early stopping |
| [`BiLSTM.ipynb`](BiLSTM.ipynb) | Bidirectional LSTM | Deep recurrent | Reads sequences both directions for richer temporal context |
| *(synthesis)* | Hybrid (statistical + neural) | Ensemble | The conceptual takeaway — combining the families' strengths rather than relying on any one |

## 🏗️ Pipeline

Each notebook is self-contained and follows the same flow:

```
AQI_19-24(test).csv
      │
      ▼
 Clean + outlier treatment (IQR clipping)
      │
      ▼
 Scale  ──►  Feature build              (lags: 7/14/30d, rolling means, calendar)
      │            │
      ▼            ▼
 Train / validation / test split  (TimeSeriesSplit — no leakage)
      │
      ▼
 Model fit + hyperparameter search
      │
      ▼
 Forecast  ──►  Evaluate (MAE · RMSE · MAPE)  ──►  Plots
```

## 🚀 Run it

These are Jupyter notebooks — open any one and run top-to-bottom.

```bash
# 1. install the stack
pip install pandas numpy scikit-learn xgboost prophet tensorflow matplotlib seaborn jupyter

# 2. launch
jupyter notebook

# 3. open XGBoost.ipynb / FBProphet.ipynb / LSTM.ipynb / BiLSTM.ipynb
```

> Notebooks expect a CSV named `AQI_19-24(test).csv` (daily date + `AQI` column) in the working directory. Update the `pd.read_csv(...)` path at the top of each notebook to point at your data.

## 📊 Results

Reported on the held-out test set within each notebook (AQI units; lower is better):

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **XGBoost** | **5.82** | **7.70** | **4.23%** |
| Prophet | 5.67 *(best validation)* | 7.83 *(best validation)* | — |
| LSTM | 6.31 | 8.66 | 4.58% |
| BiLSTM | 6.33 | 8.77 | 4.57% |

Takeaways from these runs:
- **XGBoost with engineered features is the strongest test performer** here (lowest test RMSE / MAPE), showing a well-fed tree model can match or beat deep sequence models on this series.
- **Prophet** posts the lowest *validation* MAE and handles trend/seasonality/holidays with strong interpretability.
- **LSTM and BiLSTM land close together** (~4.6% error); bidirectionality didn't move the needle much on this dataset.

> Numbers above are taken directly from the notebook outputs in this repo and depend on the specific data split — re-running on different data will shift them.

## 🔧 Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Prophet · TensorFlow/Keras · Matplotlib · seaborn · Jupyter
