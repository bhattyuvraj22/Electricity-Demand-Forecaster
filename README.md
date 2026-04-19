# Predictive Paradox — Hourly Electricity Demand Forecasting

IITG.ai Recruitment Task

---

## What This Project Does

This project predicts Bangladesh's electricity demand for the next hour using historical grid data, weather, and economic indicators. Two models are trained — Random Forest and XGBoost — and the better one is automatically picked and retrained on all available data.

---

## File Structure

```
DOTAI/
├── dataset/
│   ├── cleaned/
│   │   └── cleaned.csv                  # Final merged and cleaned dataset
│   └── raw/
│       ├── economic_full_1.csv          # Annual economic indicators (World Bank)
│       ├── PGCB_date_power_demand.xlsx  # Hourly grid demand & generation data
│       └── weather_data.xlsx            # Hourly weather observations
├── .gitignore
├── main.ipynb                           # Full pipeline notebook
├── README.md
└── requirements.txt
```

---

## Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries used: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

---

## Data Sources

### 1. PGCB Demand Data (`PGCB_date_power_demand.xlsx`)

Hourly power grid records from the Power Grid Company of Bangladesh.

| Column | Description |
|---|---|
| `datetime` | Hourly timestamp |
| `demand_mw` | Total electricity demand in megawatts |
| `generation_mw` | Total power being generated |
| `load_shedding` | Amount of load shed |
| `gas`, `liquid_fuel`, `coal`, `hydro`, `solar`, `wind` | Generation by fuel type |

### 2. Weather Data (`weather_data.xlsx`)

Hourly weather readings. The file has 4 rows of header junk that are skipped on load, and column names are assigned manually.

| Column | Description |
|---|---|
| `temperature_c` | Air temperature in Celsius |
| `feels_like_temp_c` | Feels-like temperature |
| `dew_point_c` | Dew point temperature |
| `soil_temp_c` | Soil temperature |
| `humidity_pct` | Relative humidity (%) |
| `rainfall_mm` | Rainfall in millimetres |
| `cloud_cover_pct` | Cloud cover (%) |
| `wind_direction_deg` | Wind direction in degrees |
| `sunshine_hours` | Hours of sunshine |

### 3. Economic Data (`economic_full_1.csv`)

Annual World Bank indicators for Bangladesh. The raw file stores years as columns (wide format), which is reshaped into a year-per-row table. Only data from 2015 onward is kept.

| Column | Description |
|---|---|
| `gdp_growth_pct` | Annual GDP growth rate |
| `gdp_per_person_usd` | GDP per capita in USD |
| `electricity_access_pct` | % of population with electricity access |
| `urban_growth_pct` | Annual urban population growth rate |
| `total_population` | Total population |
| `electricity_per_person_kwh` | Per capita electricity consumption |

---

## Data Cleaning

### Step 1 — Fix the demand data

- All numeric columns are force-converted from text to numbers; invalid entries become NaN.
- Duplicate timestamps are removed.
- The timeline is expanded to a complete hourly grid so any missing hours become explicit NaN rows rather than silent gaps.

### Step 2 — Remove outliers

Outliers are removed using a per-month IQR method:

- For each of the 12 calendar months, the acceptable range is calculated from **training data only** using `Q1 − 3×IQR` and `Q3 + 3×IQR`.
- Any demand value outside this range is set to NaN.
- The same bounds are then applied to the test data so the test set never influences the cleaning process.

### Step 3 — Fill missing values

- `demand_mw` — filled by linear interpolation for gaps up to 24 hours, then backward fill, then replaced with the median for anything still missing.
- Grid columns (`generation_mw`, fuel types) — forward filled, then backward filled, then set to zero if still missing.

### Step 4 — Merge weather data

The demand table is joined with the weather table on the `datetime` column. Any remaining weather gaps are filled with forward and backward fill.

### Step 5 — Merge economic data

A `year` column is extracted from `datetime`. The annual economic table is joined on year. Economic values are forward and backward filled so every hourly row has an economic context.

---

## Exploratory Data Analysis

### Demand Patterns

- **By hour of day** — line chart of average demand for each hour (0–23). Shows the daily demand cycle and the peak evening hour.
- **By month** — bar chart of average demand per month. Shows seasonal patterns driven by summer cooling load.

### Temperature vs Demand

A scatter plot of temperature against demand confirms that hotter days drive higher electricity use. The Pearson correlation value is shown directly on the chart.

### Correlation Heatmap

A heatmap of correlations between demand and all weather variables. Helps identify which weather features matter most and which overlap with each other (e.g. temperature and dew point are highly correlated).

---

## Feature Engineering

All features are built using only past information — the model never accidentally sees the future value it is trying to predict.

### Calendar Features

| Feature | Description |
|---|---|
| `hour_of_day` | Hour (0–23) |
| `day_of_week` | Day of week (0 = Monday, 6 = Sunday) |
| `month_of_year` | Month (1–12) |
| `quarter` | Quarter (1–4) |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |
| `is_summer` | 1 for June, July, August (peak cooling months in Bangladesh) |
| `is_monsoon_peak` | 1 for July, August (high humidity despite temperature drop) |

### Cyclical Encoding

Hours, months, and weekdays are also encoded as sine and cosine pairs. This stops the model from treating hour 23 and hour 0 as far apart when they are actually adjacent.

- `hour_sin`, `hour_cos`
- `month_sin`, `month_cos`
- `weekday_sin`, `weekday_cos`

### Lag Features

Past demand values used directly as features, telling the model what demand looked like at various points in the past.

| Lags included | Purpose |
|---|---|
| 1h, 2h, 3h, 4h, 6h, 12h | Short-term recent trend |
| 24h, 48h, 72h, 96h, 120h, 144h, 168h | Daily and weekly patterns |
| 167h, 169h | Neighbours of the 1-week-ago same hour |
| 336h, 504h, 672h | 2, 3, and 4 weeks ago |
| 8736h, 8760h, 8784h | Same hour roughly one year ago (±1 week) |

Additional derived lag features:

| Feature | How it is calculated |
|---|---|
| `demand_change_1h` | Demand 1h ago minus demand 2h ago |
| `demand_change_24h` | Demand 1h ago minus demand 25h ago |
| `demand_yoy_anchor` | Average of the three year-ago lags |
| `demand_yoy_growth` | How much demand has grown vs this time last year (capped at ±50%) |
| `demand_trend_7d` | Average hourly rate of change over the past 7 days |

### Rolling Window Features

Computed on demand shifted by 1 hour. Windows used: 3h, 6h, 12h, 24h, 48h, 168h.

For each window:
- `avg_demand_{w}h` — average demand over the window
- `demand_variability_{w}h` — standard deviation of demand over the window

Additional:
- `peak_demand_24h` — highest demand in the past 24 hours
- `lowest_demand_24h` — lowest demand in the past 24 hours
- `avg_same_hour_last_7d` — average of the same hour across the past 7 days
- `variability_same_hour_last_7d` — standard deviation of the same

### Weather Interaction Features

- `avg_humidity_24h` — average humidity over the past 24 hours
- `avg_humidity_168h` — average humidity over the past 7 days (captures prolonged monsoon conditions)
- `temp_humidity_interaction` — temperature × humidity / 100 (simple heat-index proxy)

### Grid State Lag Features

For each of `generation_mw`, `gas`, `liquid_fuel`, `coal`:
- `{col}_1h_ago` — value 1 hour ago
- `{col}_24h_ago` — value 24 hours ago

These 8 features give the model context about the current state of the power grid.

### Target Variable

`next_hour_demand_mw` — the demand value one hour into the future, created by shifting `demand_mw` back by 1 position. Rows where this target or the longest lag features are missing are dropped. The final feature count is 70+.

---

## Train / Test Split

The data is split at `2024-01-01`:

- **Training set** — all data before 2024
- **Test set** — all data from 2024 onward

Data is never shuffled. This gives an honest picture of how the model performs on genuinely unseen future data.

---

## Model Training

Both models are wrapped in a pipeline: `StandardScaler → Model`.

### Random Forest

Tuned using 5-fold cross-validation over 135 parameter combinations. Best parameters:

```
n_estimators     = 300
max_depth        = 20
min_samples_leaf = 5
```

Set `RUN_rf_TUNING = True` in the notebook to re-run the grid search.

### XGBoost

Tuned using 5-fold cross-validation over 360 parameter combinations. Best parameters:

```
n_estimators     = 800
learning_rate    = 0.02
max_depth        = 7
subsample        = 0.8
colsample_bytree = 0.7
min_child_weight = 3
```

Set `RUN_xgb_TUNING = True` in the notebook to re-run the grid search.

### Model Selection

Both models are evaluated on the test set. The one with the lower MAPE is automatically selected for all further steps.

---

## Results

Models are evaluated on the test set using four metrics:

| Metric | What it measures |
|---|---|
| MAPE | Average percentage error — primary selection metric |
| MAE | Average absolute error in MW |
| RMSE | Root mean squared error in MW |
| R² | How much demand variance the model explains (1.0 = perfect) |

### Visualisations

- **First week of test set** — line chart of actual vs predicted demand for the first 168 hours. Shows how well the model tracks the daily cycle on unseen data.
- **Full test set scatter** — actual on x-axis, predicted on y-axis, with a perfect-prediction line. Tight clustering along the diagonal means low bias.
- **Monthly MAPE bar chart** — error broken down by calendar month. Months above the overall MAPE are shown in red; months below in blue. Highlights any seasonal weaknesses.

---

## Feature Importance

Feature importances are extracted from the winning model and each feature is grouped into a category:

| Category | What it covers |
|---|---|
| Lag & Rolling | All past demand values and rolling statistics |
| Grid State | Lagged generation and fuel mix readings |
| Calendar | Hour, day, month, flags, cyclical encodings |
| Weather | Raw weather observation columns |
| Economic | Annual macro-economic indicators |

Two charts are shown: a bar chart of the top 25 features colour-coded by category, and a pie chart of total importance share per category. Lag and Rolling features hold the most importance, followed by Calendar and Grid State.

---

## Final Model

After evaluation, the best model is retrained on the full dataset (train + test) to use all available data before deployment.

**Save the model:**
```python
import joblib
joblib.dump(final_pipeline, 'final_model.pkl')
```

**Predict on new data:**
```python
new_data = pd.read_csv('new_live_data.csv')
new_data['Predicted_MW'] = final_pipeline.predict(new_data[FEATURE_COLUMNS])
new_data.to_csv('predictions_output.csv', index=False)
```

New data must contain all the same engineered feature columns used during training.