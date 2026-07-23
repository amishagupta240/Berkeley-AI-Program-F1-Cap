# Formula 1 Race Scheduling: EDA & Baseline Model

**Capstone Module 20.1 — Initial Report and Exploratory Data Analysis**
**Author:** Amisha Gupta

📓 [Jupyter Notebook](./f1_eda_capstone.ipynb)

---

## Research Question

> How can historical Formula 1 race data be used to identify trends in race scheduling and predict future season calendars or popular circuit selections?

This question matters to F1 organizations and event planners who must balance logistics, fan engagement, broadcaster demands, and global geographic representation when constructing each season's calendar.

---

## Dataset

- **Source:** [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — Rohan Rao, Kaggle
- **Files used:**

| File | Rows | Description |
|---|---|---|
| `races.csv` | 1,125 | Every F1 race with date, round, circuit, and session dates |
| `circuits.csv` | 77 | Circuit names, countries, and GPS coordinates |
| `seasons.csv` | 75 | One row per season, 1950–2024 |

- **Key fields:** `raceId`, `year`, `round`, `circuitId`, `name`, `date`, `time`, `fp1_date`, `fp2_date`, `fp3_date`, `quali_date`, `sprint_date`, `sprint_time`

---

## Data Cleaning

- Replaced `\N` sentinel strings (Ergast DB convention) with `NaN`
- Parsed `date` column as datetime; extracted `month`, `month_name`, `day_of_week`
- Engineered binary flags: `has_sprint`, `has_quali`, `has_fp_data`
- Removed 0 duplicate rows
- Merged `races` with `circuits` to attach country and geographic coordinates
- Note: Session date fields (FP1–3, qualifying, sprint) are only populated for 2015+ seasons — treated as structural missingness, not imputed

---

## Key EDA Findings

1. **Races per season have grown steadily** — from 7 in 1950 to 24 in 2024, with the sharpest growth occurring post-2010. The COVID-19 season (2020, 17 races) is a clear outlier from this trend.

2. **Classic European circuits dominate longevity** — Monza (74 races), Monaco (70), and Silverstone (59) are the three most frequently used venues. However, the number of new circuit debuts per decade has remained consistent, reflecting ongoing global expansion.

3. **Italy, Germany, the UK, and USA have hosted the most races historically**, but recent calendar additions skew toward the Middle East and Asia (Bahrain, Abu Dhabi, Saudi Arabia, Japan, Singapore).

4. **July is the busiest race month**; December and February are almost unused. The season reliably runs March through November, with a summer break in August.

5. **Sprint format is scaling up** — introduced in 2021 with 3 events, it doubled to 6 per season by 2023–2024, suggesting continued expansion in coming years.

6. **Outlier seasons**: IQR analysis flags the earliest seasons (1950–1959: 7–11 races) and the 2020 COVID season (17 races) as statistical outliers in race count.

---

## Baseline Model

**Task:** Predict the number of races in a future season (regression).

**Features:**
- `year` — captures the long-term upward trend
- `year_lag1` — prior season's race count (autoregressive signal)

**Model:** Linear Regression (baseline)

**Evaluation Metric:** MAE (Mean Absolute Error) — chosen because it is directly interpretable in the scheduling domain: an MAE of 1.1 means predictions are off by roughly 1 race per season.

| Metric | Score |
|---|---|
| MAE | ~1.1 races |
| RMSE | ~1.8 races |
| R² (test) | ~0.09 |
| CV R² (5-fold) | ~0.09 ± 0.18 |

The low R² reflects the model's difficulty handling abrupt changes (COVID, rapid post-2020 expansion). It serves as a useful lower-bound benchmark for more sophisticated models in Module 24.

---

## Next Steps (Module 24)

- Add features: continent diversity index, new circuit debut flag, decade dummies, rolling averages
- Try ARIMA or Facebook Prophet for proper time-series forecasting
- Clustering: group circuits by appearance frequency and geographic region
- Classification: predict whether a given circuit will appear on the next season's calendar
- Evaluate Random Forest and Gradient Boosting regressors as improvements over the linear baseline

---

## Repository Structure

```
f1_eda_capstone.ipynb   ← Main analysis notebook
races.csv               ← Race scheduling data (1950–2024)
circuits.csv            ← Circuit metadata
seasons.csv             ← Season index
README.md               ← This file
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```
