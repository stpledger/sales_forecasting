# 🏪 Store Sales - Time Series Forecasting

This notebook presents a comprehensive solution to the [Kaggle Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition. The goal is to forecast 16 days of sales for various product families across multiple Ecuadorian stores. The dataset includes historical sales, promotions, holidays, oil prices, and transactional metadata.

## 📌 Objective

To build a robust time series forecasting model capable of predicting store-item sales for a defined future period, with strong generalization performance on unseen data.

## 🧠 Approach Overview

This notebook adopts a deep learning-based approach, comparing the performance of a **Temporal Fusion Transformer (TFT)** with a custom **Multi-Layer Perceptron (MLP)** baseline. The strategy incorporates domain-specific feature engineering, time series-aware preprocessing, and tuning of learning rate schedules to optimize performance.

## 🛠️ Key Components

### 📅 Feature Engineering

A sample of the preprocessed data is available: [`feature_sample.csv`](feature_sample.csv). All sales values, including lags and rolling averages, are `log1p` transformed to minimize large residual differences for product families of varying magnitudes.

- **Static Categorical Variables**:  
  - `store_nbr`, `family`, `store_type`, `cluster`

- **Known Reals (exogenous vars)**:  
  - Calendar features like `holiday`, `special_day`, `store_closed`, Fourier-transformed `sin`/`cos` seasonal components, and day-of-week indicators
  - Exponentially decreasing feature `payday` (15th and last day of the month)

- **Unknown Reals (targets & covariates)**:  
  - `sales`, lagged sales features (`sales_lag_1`), rolling averages (`sales_window_7`), and `pct_change`
  - oil price (`dcoilwtico`), # of promotions (`onpromotion`)

### 🧮 Modeling

- **Baseline Model**: A custom MLP that processes engineered lag features
- **Advanced Model**: PyTorch Lightning’s implementation of **Temporal Fusion Transformer**, leveraging:
  - Embeddings for categorical variables
  - Encoder-decoder attention mechanism
  - Multi-horizon forecasting capabilities

### ⚙️ Training Configuration

- Optimizer: `AdamW`  
- Learning Rate Scheduler: `CosineAnnealingLR`  
- Loss Function: `RMSE` (with `log1p` scaled targets)
- Evaluation Metric: RMSLE (Root Mean Squared Logarithmic Error) on 16-day forecast horizon

RMSLE: $\sqrt{ \frac{1}{n} \sum_{i=1}^n \left(\log (1 + \hat{y}_i) - \log (1 + y_i)\right)^2}$

### ✅ Validation Strategy

For the TFT model, a time-based train-validation split is used to mimic real-world forecasting scenarios and avoid leakage. For the MLP model, a standard randomized train-test split is used since predictions are one day at a time.

Performance is tracked over time using RMSE loss, RMSLE score on 16 days of withheld data, and a few other typical regression metrics.

## 📈 Results & Insights

- The MLP baseline performs surprisingly well given its simplicity, especially when equipped with strong lag-based and calendar features.
- The TFT model shows promise but requires careful tuning and regularization to avoid overfitting due to its complexity.
- Extensive experimentation is conducted on model hyperparameters and input configurations to refine predictive performance.
- Both models achieve ~0.40 RMSLE scores, but the custom MLP model showed stronger performance overall.
- The best score achieved was 0.40391, placing my model in the top 50 on the Kaggle leaderboard.

## 🧾 File Structure

- [`project.ipynb`](project.ipynb): Rendered Jupyter notebook (use a browser to view)
- All modeling and feature engineering logic is included within the notebook.