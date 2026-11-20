# Customer Segmentation and Behavioural Analysis: Standard, RFM and Constrained K-Means

COS 802 Research Project, University of Pretoria
Author: Nqobile Ngidi (u15189181)

This repository contains the code and results for a study comparing three ways of segmenting e-commerce customers:

1. **Standard K-means** on seven behavioural features
2. **RFM K-means** on Recency, Frequency and Monetary value
3. **Soft constrained K-means**, which penalises broken must-link / cannot-link pairs built from customers' country

Customers are clustered on an earlier **training window** and judged on how well the segments separate their behaviour in a later **90-day holdout**.

## Repository contents
├── README.md
├── requirements.txt
├── run_analysis.py                    
├── notebooks/
│   └── Masters_Constrained_Customer_Segmentation_Experiment.ipynb   
├── data/
│   └── Online Retail.xlsx              # NOT included – see Data information below
├── results/                            # contents of analysis_results.zip


## Data
The study uses the public **Online Retail** dataset from the UCI Machine Learning Repository: 541,909 transactions from a UK-based online gift retailer between 1 December 2010 and 9 December 2011.
- **Dataset page:** https://archive.ics.uci.edu/dataset/352/online+retail
- **Direct download:** https://archive.ics.uci.edu/static/public/352/online+retail.zip (about 23 MB; contains Online Retail.xlsx)
- **Citation:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

## How to run the experiments

### 1. Set up Python
Python 3.10 or newer is recommended. The results in the report were produced with Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9 and SciPy 1.18.

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 2. Get the data
mkdir -p data
curl -L -o data/online_retail.zip https://archive.ics.uci.edu/static/public/352/online+retail.zip
unzip data/online_retail.zip -d data/

### 3. Run the full experiment

python run_analysis.py "data/Online Retail.xlsx" results

The two arguments are the path to the Excel file and the folder to write results to. The script:

1. cleans the data and splits it at 10 September 2011 into a training window and a 90-day holdout;
2. builds the seven customer features and five future outcomes;
3. fits standard, RFM and constrained K-means (*k* = 4);
4. computes the quality scores, cross-space scores, bootstrap intervals and robust effect sizes;
5. runs the penalty-weight sweep and the five re-sampled constraint sets;
6. writes the CSV files.

All randomness uses fixed seeds (42 for models and constraints, 11–55 for the robustness check, 2026 for the bootstrap)

## Main findings

- **RFM K-means** scored best on all three quality scores and separated future order counts best (η² 0.192 vs 0.147 for standard K-means; 95% CI of the difference 0.020–0.086).
- **Constrained K-means** reassigned only 48 customers (1.4%) compared with standard K-means. Its small gains in business separation reversed in two of five re-sampled constraint sets.
- Clusters built without constraints already break the country constraints at about the rate random grouping would, so country carried no information about purchasing behaviour.

See the report for full results and discussion.

