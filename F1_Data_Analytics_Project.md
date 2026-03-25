pip# 🏎️ Course 602 – Data Analytics using Python
## Semester Project

---

# Project Title: Exploratory Data Analysis and Predictive Modelling of Formula 1 World Championship (1950–2020)

> **Course:** 602 – Data Analytics using Python  
> **Format By:** Dr. Snehal K Joshi  
> **Student Group Size:** Maximum 2 (Individual preferred)  
> **Mode:** In-house lab / Personal laptop

---

## Dataset Details

| Field | Details |
|---|---|
| **Dataset Name** | Formula 1 World Championship (1950–2020) |
| **Source** | Kaggle |
| **Dataset Link** | https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020 |
| **Data Type** | Multiple structured CSV files (relational) |
| **Domain** | Sports Analytics / Motorsport Intelligence |

---

## Dataset Description

The Kaggle F1 dataset is a **multi-table relational dataset** comprising several CSV files that together capture the entire history of Formula 1 World Championship racing from 1950 to 2020.

### Key CSV Files and Their Contents

| File | Key Columns | Description |
|---|---|---|
| `races.csv` | raceId, year, round, circuitId, name, date | All races ever held |
| `results.csv` | resultId, raceId, driverId, constructorId, grid, position, points, laps, milliseconds, fastestLapTime, fastestLapSpeed | Final race results |
| `lap_times.csv` | raceId, driverId, lap, position, time, milliseconds | Lap-by-lap timing data |
| `pit_stops.csv` | raceId, driverId, stop, lap, time, duration, milliseconds | Pit stop details |
| `drivers.csv` | driverId, driverRef, forename, surname, dob, nationality | Driver master data |
| `constructors.csv` | constructorId, constructorRef, name, nationality | Team/constructor master data |
| `circuits.csv` | circuitId, name, location, country, lat, lng, alt | Circuit info and geography |
| `driver_standings.csv` | raceId, driverId, points, position, wins | Championship standings |
| `constructor_standings.csv` | raceId, constructorId, points, position, wins | Constructor standings |
| `qualifying.csv` | qualifyId, raceId, driverId, constructorId, q1, q2, q3 | Qualifying session times |
| `status.csv` | statusId, status | Race finish status codes (e.g., Finished, Accident, +1 Lap) |

### Why This Dataset Is Ideal For This Project

- Multi-table joins → real-world data engineering practice
- Rich temporal data (70 years) → time-series and trend analysis
- Mix of numerical, categorical, ordinal data types
- Outliers naturally present (crashes, weather, safety cars)
- Regression targets: lap time, final position
- Classification targets: podium finish (Yes/No), retirement (Yes/No)
- Missing data present (DNFs, DNS, no fastest lap data in early years)

---

## Project Objectives

After completing this project, students will be able to:

1. Load, merge, and understand a real-world multi-table dataset using Pandas
2. Perform thorough Exploratory Data Analysis (EDA) on sports data
3. Identify and handle missing values and outliers specific to motorsport context
4. Compute and interpret statistical measures on racing metrics
5. Automate EDA using reusable Python functions
6. Perform regression analysis to predict **lap times**
7. Build and evaluate a model to predict **race winner / finishing position**
8. Analyse **pit stop strategies** and their effect on race outcomes
9. Apply supervised learning (regression + classification) using Scikit-learn
10. Evaluate models using appropriate error metrics
11. Visualise findings through rich domain-specific plots

---

## Python Tools & Libraries

| Library | Purpose |
|---|---|
| `Python 3.x` | Core language |
| `pandas` | Data loading, merging, manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Base visualisations |
| `seaborn` | Statistical plots (heatmaps, pair plots, box plots) |
| `scikit-learn` | Regression, classification, model evaluation |
| `datetime` | Parsing race dates and lap time strings |

> **Optional (Bonus):** `plotly` for interactive charts, `folium` for circuit map visualisation

---

## Project Tasks

---

### Task 1: Data Loading and Understanding

**Objective:** Load all relevant CSV files and understand the structure of the dataset.

**Steps:**
- Load the following files using `pd.read_csv()`:
  - `races.csv`, `results.csv`, `lap_times.csv`, `pit_stops.csv`, `drivers.csv`, `constructors.csv`, `circuits.csv`, `qualifying.csv`, `status.csv`
- For each file, display:
  - First 5 rows (`.head()`)
  - Last 5 rows (`.tail()`)
  - Shape (`.shape`)
  - Column names and data types (`.dtypes`, `.info()`)
  - Basic statistics (`.describe()`)
- Merge key tables to form two **master DataFrames**:
  - **Race Results Master:** `results` + `races` + `drivers` + `constructors` + `status`
  - **Lap Times Master:** `lap_times` + `races` + `drivers`

**Data Type Identification:**

| Data Type | F1 Examples |
|---|---|
| Quantitative – Continuous | lap time (ms), fastest lap speed, points |
| Quantitative – Discrete | grid position, final position, laps completed, pit stop count |
| Qualitative – Nominal | driver nationality, constructor name, circuit country |
| Qualitative – Ordinal | race round number, championship standing position |

---

### Task 2: Exploratory Data Analysis (EDA)

#### 2A — Univariate Analysis

Analyse the following variables individually:

| Variable | Plot Types |
|---|---|
| Lap time (ms) | Histogram, Boxplot |
| Pit stop duration (seconds) | Histogram, Boxplot |
| Final finishing position | Bar chart, Histogram |
| Grid (starting) position | Histogram |
| Race points scored | Histogram, Boxplot |

**Key Questions to Answer:**
- What is the distribution of lap times? Is it normally distributed or skewed?
- Are there extreme outliers in pit stop durations? (Safety car pit stops, tyre failures)
- What is the most common finishing position?

#### 2B — Bivariate Analysis

Analyse relationships between pairs of variables:

| X Variable | Y Variable | Plot Type |
|---|---|---|
| Grid Position | Final Finishing Position | Scatter plot |
| Pit Stop Count | Final Position | Scatter + trend line |
| Fastest Lap Speed | Race Points | Scatter plot |
| Qualifying Time (Q3) | Grid Position | Scatter plot |
| Year | Average Lap Time | Line plot (trend over decades) |

Also compute:
- **Correlation matrix** for all numerical columns in the Race Results Master
- **Pearson correlation coefficient** between grid position and final position

#### 2C — Multivariate Analysis

- Compare **lap times across different constructors** using grouped box plots
- Analyse **points, wins, and podiums across nationalities** using a heatmap
- Plot **grid position vs final position vs points** using a pair plot (coloured by constructor)
- Visualise **average pit stop duration by constructor and year** using a grouped bar chart

---

### Task 3: Handling Missing Data and Outliers

#### Missing Data

Common missing value patterns in this dataset:
- `fastest_lap_time`: Not recorded in early F1 seasons (pre-1996)
- `q2`, `q3` qualifying times: Drivers eliminated in Q1 have no Q2/Q3 times
- `position`: NULL when driver did not finish (DNF, DSQ)
- `milliseconds` in results: Missing for some historical races

**Steps:**
1. Identify missing values: `df.isnull().sum()` and visualise using a heatmap
2. For each column, decide appropriate handling:
   - **Numerical columns** (lap time, fastest lap speed): Impute with **median** (not mean, due to skewness from outliers)
   - **Categorical columns** (status, constructor): Impute with **mode** or flag as `"Unknown"`
   - **Position (DNF)**: Do **not** impute — create a binary column `finished (1/0)` instead
3. Document your decisions with reasoning

#### Outlier Detection

1. Plot **boxplots** for: lap time, pit stop duration, points per race
2. Apply **IQR method** to detect outliers:
   - Lower bound = Q1 - 1.5 × IQR
   - Upper bound = Q3 + 1.5 × IQR
3. Contextualise outliers:
   - Extremely long pit stops → likely mechanical failure, not data error → **keep but flag**
   - Safety Car laps → inherently slower → **keep, note context**
   - Lap times of 0 or negative → data corruption → **remove**
4. Compare model performance before and after outlier treatment

---

### Task 4: Spread and Distribution of Data

For the following key variables — **lap time**, **pit stop duration**, **grid position**, **final position**:

**Calculate:**
- Mean, Median, Mode
- Variance and Standard Deviation
- Skewness (positive/negative) using `scipy.stats.skew()`
- Kurtosis using `scipy.stats.kurtosis()`
- Percentiles: 25th, 50th, 75th, 90th, 99th

**Visualise:**
- Histogram with KDE (Kernel Density Estimate) overlay
- Q-Q plot to test normality
- Compare distributions across eras: **1950–1980 vs 1981–2000 vs 2001–2020**

**Interpret:**
- Are lap times normally distributed? (Expected: right-skewed due to safety cars and slow laps)
- How has spread of lap times changed as cars evolved?
- Which constructors have the most consistent (low variance) lap times?

---

### Task 5: Automating EDA Using Python

Write a **reusable EDA module** with the following functions:

```python
def load_and_merge_f1_data(data_path):
    """Load all CSVs and return merged master DataFrames"""

def summarise_dataframe(df, name="DataFrame"):
    """Print shape, dtypes, missing %, and descriptive stats"""

def plot_univariate(df, column, bins=50):
    """Plot histogram + boxplot side by side for a numeric column"""

def plot_bivariate(df, x_col, y_col, hue=None):
    """Scatter plot with optional colour grouping"""

def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    """Seaborn heatmap of correlation matrix"""

def detect_outliers_iqr(df, column):
    """Return DataFrame of outlier rows for a given column using IQR"""

def analyse_missing(df):
    """Visualise missing values as a heatmap"""
```

Each function must:
- Accept a DataFrame as input
- Return results and/or display plots
- Include docstrings

---

### Task 6: Regression Analysis – Lap Time Prediction

**Problem Definition:** Predict the lap time (in milliseconds) for a given lap using race and driver features.

**Target Variable (Y):** `milliseconds` (from `lap_times.csv`)

**Feature Variables (X):**

| Feature | Source | Type |
|---|---|---|
| `lap` number | lap_times | Numerical |
| `grid` position | results | Numerical |
| `circuitId` | races | Categorical (encoded) |
| `constructorId` | results | Categorical (encoded) |
| `year` | races | Numerical |
| `driverId` | lap_times | Categorical (encoded) |

**Steps:**
1. Merge `lap_times` with `races`, `results`, `drivers`, `constructors`
2. Filter to a subset: e.g., races from **2010–2020** for computational feasibility
3. Encode categorical variables using `pd.get_dummies()` or `LabelEncoder`
4. Compute **covariance** and **Pearson correlation** between features and target
5. Split data: **80% Train / 20% Test** using `train_test_split()`
6. Build:
   - (i) **Simple Linear Regression** — lap number vs lap time
   - (ii) **Multiple Linear Regression** — all features vs lap time
7. Plot regression lines and residual plots

---

### Task 7: Supervised Learning – Regression Models

#### Lap Time Prediction (Regression)

Build and compare the following models using `sklearn`:

| Model | sklearn Class |
|---|---|
| Simple Linear Regression | `LinearRegression` |
| Multiple Linear Regression | `LinearRegression` (multi-feature) |
| Polynomial Regression | `PolynomialFeatures` + `LinearRegression` |

**Train/Validation/Test Split:** 70% / 10% / 20%

**Workflow for each model:**
1. Fit on training data
2. Evaluate on validation data (tune if needed)
3. Final evaluation on test data

#### Race Position Prediction (Regression)

**Target:** Final finishing position (`positionOrder`)  
**Features:** Grid position, qualifying time, constructor, year, circuit, pit stop count

Apply **Multiple Linear Regression** and report performance.

#### Overfitting and Underfitting Analysis

- Plot **learning curves**: Training error vs Validation error as training size increases
- Demonstrate overfitting: Use a high-degree polynomial (degree 10+) on lap time data
- Demonstrate underfitting: Use a simple mean predictor baseline
- Use **cross-validation** (`cross_val_score`) to get robust performance estimates

---

### Task 8: Pit Stop Strategy Analysis

**Objective:** Understand how pit stop strategy affects race outcome.

**Analysis Steps:**
1. Calculate **total pit stop time per driver per race** = sum of all pit stop durations
2. Calculate **number of pit stops per driver per race**
3. Analyse: does fewer pit stops correlate with better finishing position?
4. Find: which constructor has the **fastest average pit stop** time?
5. Visualise:
   - Scatter: Number of pit stops vs Final Position
   - Line chart: Average pit stop duration trend by year (2011–2020, as pit stop data starts 2011)
   - Bar chart: Top 10 constructors by average pit stop speed
6. Compute correlation between pit stop count and final position
7. Identify races where pit stop strategy visibly decided the winner (outlier analysis)

---

### Task 9: Overfitting and Underfitting Analysis

- Define overfitting and underfitting with F1 analogies (e.g., a model that perfectly memorises 2019 race data but fails on 2020)
- Plot training vs testing MSE across model complexities (polynomial degree 1 to 15)
- Implement **Regularisation**:
  - Ridge Regression (`Ridge`)
  - Lasso Regression (`Lasso`)
- Compare: Linear vs Polynomial vs Ridge vs Lasso on lap time prediction task

---

### Task 10: Classification Task – Podium Prediction

**Problem Definition:** Predict whether a driver will finish on the **podium (Top 3)** in a race.

**Target Variable:** `podium` → 1 if `positionOrder <= 3`, else 0

**Feature Variables:**

| Feature | Description |
|---|---|
| `grid` | Starting grid position |
| `qualifying_time_ms` | Q3 qualifying lap in ms |
| `constructor_points_before_race` | Constructor championship points entering the race |
| `driver_points_before_race` | Driver championship points entering the race |
| `circuitId` | Circuit (encoded) |
| `year` | Season year |

**Steps:**
1. Create the binary target column
2. Handle class imbalance (podium finishes are ~21% of all results): use `class_weight='balanced'`
3. Build: **Logistic Regression** classifier
4. Evaluate using:
   - **Accuracy**
   - **Confusion Matrix** (visualised as heatmap)
   - **Precision, Recall, F1-Score** (classification report)
   - **ROC-AUC Score**

---

### Task 11 & 12: Model Evaluation

Calculate and interpret the following for **all models built**:

#### Regression Models (Lap Time, Race Position)

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | Mean Absolute Error | Average prediction error in milliseconds |
| MSE | Mean Squared Error | Penalises large errors more |
| RMSE | √MSE | Same unit as lap time (ms) |
| R² Score | Coefficient of Determination | % variance explained (closer to 1 = better) |

#### Classification Model (Podium Prediction)

| Metric | Interpretation |
|---|---|
| Accuracy | % of correct predictions |
| Precision | Of predicted podiums, how many were actual? |
| Recall | Of actual podiums, how many were caught? |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the ROC curve |

**Create a model comparison table** summarising all metrics for all models.

---

### Task 13: Data Visualisation Summary

Produce the following final visualisation outputs (save all as `.png`):

| Plot | Description |
|---|---|
| Lap time distribution histogram + KDE | Univariate: normality check |
| Grid vs Final Position scatter | Bivariate: starting advantage |
| Correlation heatmap (race results) | Multivariate: feature relationships |
| Pit stops vs final position scatter | Strategy analysis |
| Lap time trend by year (line chart) | Temporal trend |
| Top 10 drivers by total wins (bar chart) | Domain insight |
| Constructor championship points over years (line) | Temporal multivariate |
| Confusion matrix heatmap (podium classifier) | Model evaluation |
| Learning curve (training vs test error) | Overfitting diagnosis |
| Regression line: Grid position vs Final position | Model visualisation |
| Circuit map: average lap time by country | Geospatial (bonus) |

---

## Expected Project Outcomes

By the end of this project, students will have:

- Performed a complete EDA on a real-world, multi-table sports dataset
- Built and evaluated regression models to **predict F1 lap times**
- Analysed **pit stop strategy** and its impact on race outcomes
- Built a classification model to **predict podium finishes**
- Understood overfitting and underfitting in a practical context
- Produced professional-grade data visualisations
- Connected statistical theory with motorsport domain knowledge

---

## Project Submission Guidelines

Students must submit the following:

1. **Python Notebook** (`.ipynb`) — fully executed with all outputs visible
2. **Project Report** (Word / PDF) containing:
   - Title page with student details
   - Introduction and motivation
   - Dataset description and preprocessing steps
   - EDA findings with plots and interpretation
   - Model building methodology
   - Results, metrics, and model comparison table
   - Observations, insights, and conclusion
   - References
3. **Plots folder** — all visualisations saved as `.png` files

---

## Suggested Evaluation Criteria

| Component | Marks |
|---|---|
| Data Loading, Merging & Cleaning | 20 |
| EDA & Visualisation | 25 |
| Regression & ML Models (Lap Time + Race Position) | 20 |
| Pit Stop Strategy Analysis | 10 |
| Classification (Podium Prediction) & Model Evaluation | 15 |
| Report & Presentation | 10 |
| **Total** | **100** |

---

## Suggested Project Timeline

| Week | Tasks |
|---|---|
| Week 1 | Data download, loading, merging, Task 1 |
| Week 2 | EDA – Tasks 2, 3, 4 |
| Week 3 | Automation + Regression – Tasks 5, 6 |
| Week 4 | ML Models + Pit Stop Analysis – Tasks 7, 8, 9 |
| Week 5 | Classification + Evaluation – Tasks 10, 11, 12 |
| Week 6 | Visualisation polish + Report writing – Task 13 |

---

*Project Format by: Dr. Snehal K Joshi | Course 602 – Data Analytics using Python*
