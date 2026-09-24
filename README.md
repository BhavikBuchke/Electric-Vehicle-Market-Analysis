# ⚡ Electric Vehicle Market Analysis

A comprehensive data science project analysing the global electric vehicle (EV) market across 3,022 vehicle configurations, 53 manufacturers, and 40 countries (model years 2015–2025).

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Analysis Pipeline](#analysis-pipeline)
- [Key Findings](#key-findings)
- [Installation & Setup](#installation--setup)
- [Running the Notebook](#running-the-notebook)
- [Technologies Used](#technologies-used)
- [Results & Outputs](#results--outputs)

---

## Project Overview

**Objective:** Analyse electric vehicle specifications, pricing, technology, manufacturer characteristics, and 2024 sales to identify patterns in the EV market and generate actionable insights for product strategy, pricing, and portfolio decisions.

**Type:** Academic / Data Science Research Project

**Research Questions:**
1. What technical and commercial factors most strongly determine EV pricing?
2. Do price segment and range class significantly predict 2024 unit sales?
3. What natural market archetypes exist beyond manual price-tier definitions?
4. Which battery chemistry and charging standards offer the greatest efficiency and commercial advantages?

---

## Dataset

| Property | Value |
|---|---|
| File | `electric_vehicles_dataset.csv` |
| Rows | 3,022 |
| Columns | 17 |
| Manufacturers | 53 |
| Countries | 40 |
| Model Years | 2015 – 2025 |
| Memory | ~1.3 MB |

### Columns

| Column | Type | Description |
|---|---|---|
| `Vehicle_ID` | int | Surrogate key (dropped during cleaning) |
| `Manufacturer` | str | Brand name (53 unique; most frequent: Ferrari) |
| `Model` | str | Model name (181 unique; most frequent: Nevera) |
| `Year` | int | Model year 2015–2025; mean 2020 |
| `Battery_Type` | str | Battery chemistry (15 types; most common: Lithium-Titanate) |
| `Battery_Capacity_kWh` | float | Usable capacity 20–150 kWh; mean 84.3 kWh |
| `Range_km` | int | Rated range 100–600 km; mean 350 km |
| `Charging_Type` | str | Charging standard (16 types; most common: Mobile Charging) |
| `Charge_Time_hr` | float | Full charge time 0.5–12 h; mean 6.2 h |
| `Price_USD` | float | Retail price $30,014–$149,979; mean $90,612 |
| `Color` | str | Exterior colour (55 options) |
| `Country_of_Manufacture` | str | Production country (40 countries) |
| `Autonomous_Level` | float | SAE autonomy level 0–5; 14.6% missing |
| `CO2_Emissions_g_per_km` | float | Tailpipe emissions; 19.6% missing (BEVs = 0) |
| `Safety_Rating` | float | Star rating 3–5; 11.2% missing |
| `Units_Sold_2024` | int | 2024 sales volume 6–19,996 units |
| `Warranty_Years` | int | Manufacturer warranty 3–5 years |

---

## Project Structure

```
IBM internship project/
│
├── electric_vehicles_dataset.csv        # Raw input dataset
├── ev_cleaned_engineered.csv            # Cleaned + feature-engineered output
├── project_executed.ipynb               # Main analysis notebook (fully executed)
├── project_document_with_visuals.docx   # Academic project report with figures
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
└── doc_images/                          # Chart exports from notebook
    ├── fig1_distributions.png
    ├── fig2_manufacturer_revenue_sales.png
    ├── fig3_price_segments.png
    ├── fig4_range_efficiency.png
    ├── fig5_battery_charging.png
    ├── fig6_temporal_trends.png
    ├── fig7_geographic.png
    ├── fig8_autonomy_safety.png
    ├── fig9_highrange_comparison.png
    ├── fig10_correlation.png
    ├── fig12_outlier_boxplots.png
    ├── fig13_phev_bev_violin.png
    ├── fig14_warranty_colour.png
    ├── fig15_multivariate.png
    ├── fig17_cluster_pca.png
    ├── fig18_hypothesis_testing.png
    ├── fig19_price_prediction.png
    └── fig20_residuals.png
```

---

## Analysis Pipeline

The notebook follows a structured 14-section pipeline:

```
1.  Data Loading & Initial Inspection
2.  Data Cleaning (imputation, type casting, string standardisation)
3.  Feature Engineering (7 derived features)
4.  EDA — Univariate Distributions
5.  EDA — Manufacturer Landscape (Top 15 Revenue & Sales)
6.  EDA — Price Segment Analysis
7.  EDA — Range & Efficiency Analysis
8.  EDA — Battery & Charging Technology
9.  EDA — Temporal Trends (2015–2025)
10. EDA — Geographic Manufacturing Footprint
11. EDA — Autonomous Level & Safety Rating
12. EDA — High-Range vs Standard Comparison
13. EDA — Correlation Matrix
14. EDA — Top & Bottom Value Vehicles (Price/km)
15. EDA — Era × Price Segment Heatmap
──────────────────────────────────────────
16. Outlier Detection (IQR method, composite flagging)
17. PHEV vs BEV Segmentation (Welch t-test per feature)
18. Colour & Warranty Analysis (ANOVA)
19. Multivariate Analysis (Pair Plot, Battery Band Efficiency)
20. K-Means Clustering (k=5, Elbow + Silhouette, PCA)
21. Statistical Hypothesis Testing (ANOVA, Welch t-test, Mann-Whitney U, Tukey HSD)
22. Predictive Modelling (OLS, Random Forest, Gradient Boosting, 5-fold CV)
23. Consolidated Academic Findings & Recommendations
24. Dataset Export
```

### Engineered Features

| Feature | Formula | Purpose |
|---|---|---|
| `High_Range` | `Range_km >= 400` → 0/1 | Binary long-range flag |
| `Price_Segment` | 4-tier price bins | Market tier classification |
| `Range_per_kWh` | `Range_km / Battery_Capacity_kWh` | Energy efficiency (km/kWh) |
| `Price_per_km` | `Price_USD / Range_km` | Value-for-range metric |
| `Efficiency_Tier` | Quartile-cut of `Range_per_kWh` | Low / Medium / High / Top |
| `Est_Revenue_2024` | `Price_USD × Units_Sold_2024` | Revenue proxy |
| `Era` | Year cohort bins | Early / Mid / Recent |

---

## Key Findings

### Market Structure
- Only **40.1%** of models qualify as high-range (≥400 km) — a clear differentiation opportunity
- **Luxury (>$110K)** segment is the largest (1,023 models, 33.9%) and drives the most revenue
- **Ferrari** leads estimated 2024 revenue (~$78.6B); **China** leads manufacturing country count

### Technology
- Larger batteries **reduce** energy efficiency (km/kWh) — vehicles under 50 kWh are most efficient per unit
- **Zinc-Air** chemistry leads in range/kWh efficiency; **Sodium-Ion** correlates with highest avg sales
- Battery capacity is the strongest physical predictor of range, but explains only part of the variance

### Statistical Tests (α = 0.05)
| Hypothesis | Test | Result |
|---|---|---|
| H1: Sales differ across Price Segments | One-way ANOVA | Not significant — price tier doesn't drive volume |
| H2: Price differs across Autonomy Levels | ANOVA + Tukey HSD | Significant — higher SAE = modest price premium |
| H3: High-Range outsells Standard-Range | Welch t-test | Not significant — range class doesn't predict volume |

### Predictive Modelling
| Model | R² (Test) | MAE | RMSE |
|---|---|---|---|
| OLS (statsmodels) | ~0.35–0.42 | ~$25–28K | ~$29–32K |
| Random Forest (n=200) | ~0.45–0.55 | ~$22–25K | ~$26–29K |
| Gradient Boosting (n=200) | ~0.45–0.55 | ~$22–25K | ~$26–29K |

> ~55–65% of price variance is **unexplained by specs** — brand equity is a major pricing lever.

### Top Recommendations
1. **Target $80K–$110K premium segment** — highest revenue density + volume gap
2. **Achieve ≥400 km range** — join the high-range minority for premium positioning
3. **Prioritise Zinc-Air / Sodium-Ion chemistry** for efficiency and commercial leadership
4. **Focus on SAE Level 3 autonomy** — L3 outsells L4/L5; full autonomy not yet monetisable
5. **Invest in brand equity** — OLS residuals show specs alone don't justify price
6. **Avoid over-investing in colour/warranty** — no statistically significant sales impact

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### 1. Clone or download the project

```bash
git clone <repo-url>
cd "IBM internship project"
```

### 2. Create a virtual environment

```bash
python -m venv evenv
```

**Activate:**
- Windows: `evenv\Scripts\activate`
- macOS/Linux: `source evenv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

```
pandas==3.0.6
numpy==2.5.3
matplotlib==3.11.2
seaborn==0.13.2
scipy==1.18.1
statsmodels==0.15.0
scikit-learn==1.9.1
jupyterlab==4.6.3
notebook==7.6.2
ipykernel==7.3.0
ipywidgets==8.1.9
nbconvert==7.17.1
```

---

## Running the Notebook

```bash
# Start JupyterLab
jupyter lab

# Or classic notebook
jupyter notebook
```

Open `project_executed.ipynb` — it is already **fully executed** with all 28 code cells and inline outputs. To re-run:

```
Kernel → Restart Kernel and Run All Cells
```

To re-execute from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace project_executed.ipynb
```

---

## Technologies Used

| Library | Version | Usage |
|---|---|---|
| **pandas** | 3.0.6 | Data manipulation, groupby, pivot, merging |
| **numpy** | 2.5.3 | Numerical operations, array handling |
| **matplotlib** | 3.11.2 | Histograms, scatter, line, bar, boxplots |
| **seaborn** | 0.13.2 | Heatmaps, violin plots, pair plots |
| **scipy** | 1.18.1 | ANOVA, Welch t-test, Mann-Whitney U |
| **statsmodels** | 0.15.0 | OLS regression, Tukey HSD post-hoc |
| **scikit-learn** | 1.9.1 | K-Means, PCA, Random Forest, Gradient Boosting, CV |
| **JupyterLab** | 4.6.3 | Interactive notebook environment |

**Environment:** Python 3.13.14 · Windows 10 x64

---

## Results & Outputs

| Output File | Description |
|---|---|
| `ev_cleaned_engineered.csv` | Cleaned dataset with 7 engineered features (3,022 × 23) |
| `project_document_with_visuals.docx` | Full academic report with 10 embedded figures |
| `doc_images/` | 18 exported chart PNGs from the notebook |

---

## Notebook Structure (28 Cells)

| Cell | Section |
|---|---|
| 1 | Imports & Configuration |
| 2 | Data Loading & Shape/Memory |
| 3 | Missing Value Audit |
| 4 | Statistical Summary (describe) |
| 5 | Data Cleaning |
| 6 | Feature Engineering |
| 7 | Univariate Distributions |
| 8 | Manufacturer Landscape |
| 9 | Price Segment Analysis |
| 10 | Range & Efficiency |
| 11 | Battery & Charging Technology |
| 12 | Temporal Trends |
| 13 | Geographic Footprint |
| 14 | Autonomy & Safety |
| 15 | High-Range vs Standard |
| 16 | Correlation Matrix |
| 17 | Top/Bottom Value Vehicles |
| 18 | Era × Price Segment Heatmap |
| 19 | Original Strategic Insights |
| 20 | Dataset Export |
| 21 | Outlier Detection (IQR) |
| 22 | PHEV vs BEV Analysis |
| 23 | Colour & Warranty Analysis |
| 24 | Multivariate Analysis |
| 25 | K-Means Clustering |
| 26 | Hypothesis Testing |
| 27 | Predictive Modelling |
| 28 | Academic Findings Summary |

---

*Project completed as part of IBM Internship — Data Science Track*
