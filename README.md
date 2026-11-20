# Customer Segmentation and Behavioural Analysis: Standard, RFM and Constrained K-Means

COS 802 Research Project, University of Pretoria
Author: Nqobile Ngidi (u15189181)

This repository contains the code, results and report for a study comparing three ways of segmenting e-commerce customers:

1. **Standard K-means** on seven behavioural features
2. **RFM K-means** on Recency, Frequency and Monetary value
3. **Soft constrained K-means**, which penalises broken must-link / cannot-link pairs built from customers' country

Customers are clustered on an earlier **training window** and judged on how well the segments separate their behaviour in a later **90-day holdout**. The holdout data is never used to build the clusters.

---

## Repository contents

```
.
├── README.md
├── requirements.txt
├── run_analysis.py                     # main experiment script (reproduces every number in the report)
├── notebooks/
│   └── Masters_Constrained_Customer_Segmentation_Experiment.ipynb   # original Colab notebook
├── data/
│   └── Online Retail.xlsx              # NOT included – download it (see "Data" below)
├── results/                            # contents of analysis_results.zip
└── report/
    ├── main.tex                        # IEEE-format report
    └── figures/                        # pre-rendered figures (PDF) and their pgfplots sources
```

### What is in `analysis_results.zip`

These are the outputs of `run_analysis.py`. Unzip them into `results/`.

| File | Contents | Report section |
|---|---|---|
| `customer_features_and_future_outcomes.csv` | One row per customer (3,370): the seven training features, the five holdout outcomes, and the cluster label from each method | Data, Methodology |
| `descriptives.csv` | Median, mean, max and skew of every feature and outcome | Data characteristics (Table 4) |
| `statistical_validation.csv` | Silhouette, Davies–Bouldin, Calinski–Harabasz and min/max cluster size for each method at *k* = 4 | RQ1 |
| `cross_space_validation.csv` | Every method's clusters scored in both the extended and the RFM feature space | RQ1 (cross-space table) |
| `k_selection.csv` | Inertia, the three quality scores and future-outcome η² for *k* = 2…8 in both spaces | RQ1 (choice of *k*) |
| `future_business_separation.csv` | η² and Kruskal–Wallis H and *p* for each method × future outcome | RQ2 |
| `eta_bootstrap.csv` | 95% bootstrap intervals for each η² and for the differences between methods (1,000 resamples) | RQ2 (uncertainty) |
| `robust_effect_sizes.csv` | η² on log(1 + y) and rank-based ε² for each method × outcome | RQ2 (robust effect sizes) |
| `future_business_segment_profiles.csv` | Per-cluster size, training medians, holdout return rate, mean future revenue and revenue share | Segment profiles |
| `constraint_weight_sensitivity.csv` | Constrained model at penalty weights 0.1–5: quality scores, violation rates, η², agreement with standard K-means | Penalty weight |
| `constraint_set_robustness.csv` / `_summary.csv` | Constrained model on five re-sampled constraint sets (seeds 11–55), with mean and SD | Robustness |
| `pca_coords.csv` | First two principal components for every customer, with each method's labels | PCA figure |
| `centroids_standard_scaled.csv` | Standard K-means centroids in scaled feature units | — |
| `summary.json` | Data-cleaning counts, split dates, constraint statistics, violation and chance rates, agreement scores, number of reassigned customers, top-1% revenue share, run time and package versions | Used throughout the text |

---

## Data

The study uses the public **Online Retail** dataset from the UCI Machine Learning Repository: 541,909 transactions from a UK-based online gift retailer between 1 December 2010 and 9 December 2011.

- **Dataset page:** https://archive.ics.uci.edu/dataset/352/online+retail
- **Direct download:** https://archive.ics.uci.edu/static/public/352/online+retail.zip (about 23 MB; contains `Online Retail.xlsx`)
- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0)
- **Citation:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

The data file is not committed to this repository. Download and unzip it, then place `Online Retail.xlsx` in `data/`.

---

## How to run the experiments

### 1. Set up Python

Python 3.10 or newer is recommended. The results in the report were produced with Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9 and SciPy 1.18.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the data

```bash
mkdir -p data
curl -L -o data/online_retail.zip https://archive.ics.uci.edu/static/public/352/online+retail.zip
unzip data/online_retail.zip -d data/
```

On Windows, download the zip in a browser and extract `Online Retail.xlsx` into `data/`.

### 3. Run the full experiment

```bash
python run_analysis.py "data/Online Retail.xlsx" results
```

The two arguments are the path to the Excel file and the folder to write results to. The script:

1. cleans the data and splits it at 10 September 2011 into a training window and a 90-day holdout;
2. builds the seven customer features and five future outcomes;
3. fits standard, RFM and constrained K-means (*k* = 4);
4. computes the quality scores, cross-space scores, *k* = 2…8 scan, η², bootstrap intervals and robust effect sizes;
5. runs the penalty-weight sweep and the five re-sampled constraint sets;
6. writes the CSV files and `summary.json` listed above.

**Run time:** about 3–4 minutes on one CPU core. Reading the Excel file takes about a minute; the script then caches the data as `results/raw.pkl`, so later runs are faster. Progress is printed to the console.

All randomness uses fixed seeds (42 for models and constraints, 11–55 for the robustness check, 2026 for the bootstrap), so repeated runs give identical results. Other package versions may change results in the last decimal places.

### 4. (Optional) Run the original notebook

`notebooks/Masters_Constrained_Customer_Segmentation_Experiment.ipynb` is the original Google Colab notebook. It covers the core experiment (the three models, quality scores, η², penalty weight and constraint-set robustness). `run_analysis.py` reproduces it exactly and adds the further analyses used in the report. Open it in Colab and run all cells. It downloads the data itself, or asks you to upload `Online Retail.xlsx` if the download fails.

### 5. Build the report

Upload `report/main.tex` and the `report/figures/` folder to Overleaf (or compile locally) with **pdfLaTeX**. The figures are pre-rendered PDFs, so the report compiles quickly. To change a figure, edit its source in `report/figures/src/`, compile it with `pdflatex`, and copy the resulting PDF into `report/figures/`.

---

## Method summary

| Setting | Value |
|---|---|
| Training window | 1 Dec 2010 – 9 Sep 2011 (3,370 customers) |
| Holdout window | 10 Sep – 9 Dec 2011 (1,921 returning customers) |
| Features (extended) | Recency, Frequency, Monetary, average order value, total units, product diversity, mean days between purchases |
| Features (RFM) | Recency, Frequency, Monetary |
| Preprocessing | log(1 + x), then robust scaling (median / IQR) |
| Clusters | *k* = 4, k-means++ seeding, 20 initialisations |
| Constraints | 150 must-link + 150 cannot-link pairs from customers' modal country; penalty weight *w* = 1 |
| Future outcomes | Revenue, orders, average order value, units, product diversity |
| Business separation | η², log-scale η², Kruskal–Wallis ε², 1,000-resample bootstrap |

## Main findings

- **RFM K-means** scored best on all three quality scores and separated future order counts best (η² 0.192 vs 0.147 for standard K-means; 95% CI of the difference 0.020–0.086).
- **Constrained K-means** reassigned only 48 customers (1.4%) compared with standard K-means. Its small gains in business separation reversed in two of five re-sampled constraint sets.
- Clusters built without constraints already break the country constraints at about the rate random grouping would, so country carried no information about purchasing behaviour.

See the report for full results and discussion.

## Citation

If you use this code, please cite the report and the dataset:

```
N. Ngidi, "Customer Segmentation and Behavioural Analysis: A Comparison of Clustering
Methods for E-commerce Segmentation," COS 802 Research Project, University of Pretoria, 2026.

D. Chen, "Online Retail," UCI Machine Learning Repository, 2015. https://doi.org/10.24432/C5BW33
```
