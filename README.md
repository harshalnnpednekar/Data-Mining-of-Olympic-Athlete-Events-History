# 🏅 Data Mining of Olympic Athlete Events History (1896–2016)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.2%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![mlxtend](https://img.shields.io/badge/mlxtend-v0.22%2B-green.svg)](http://rasbt.github.io/mlxtend/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade **Knowledge Discovery in Databases (KDD)** mini-project analyzing over **120 years of modern Olympic Games history (Athens 1896 to Rio 2016)**. This repository delivers a rigorous benchmarking study covering **Supervised Classification**, **Unsupervised Demographic Clustering**, and **Association Rule Mining & Frequent Pattern Analysis**.

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [Project Architecture & KDD Pipeline](#-project-architecture--kdd-pipeline)
- [Dataset Overview & Preprocessing](#-dataset-overview--preprocessing)
  - [Raw Attributes](#raw-attributes)
  - [Data Cleaning & Integrity Auditing](#data-cleaning--integrity-auditing)
  - [Target & Feature Engineering](#target--feature-engineering)
- [Supervised Classification: Predicting Olympic Medalists](#-supervised-classification-predicting-olympic-medalists)
  - [Feature Processing Pipeline](#feature-processing-pipeline)
  - [Class Imbalance Strategy](#class-imbalance-strategy)
  - [Evaluated Classifiers](#evaluated-classifiers)
  - [Classification Benchmark Results](#classification-benchmark-results)
  - [Feature Importance Analysis](#feature-importance-analysis)
- [Unsupervised Clustering: Athlete Demographic Segmentation](#-unsupervised-clustering-athlete-demographic-segmentation)
  - [Clustering Preprocessing](#clustering-preprocessing)
  - [Clustering Algorithms](#clustering-algorithms)
  - [Clustering Benchmark Results](#clustering-benchmark-results)
- [Association Rule Mining: Discovering Hidden Patterns](#-association-rule-mining-discovering-hidden-patterns)
  - [Transaction Construction](#transaction-construction)
  - [Frequent Pattern Algorithms](#frequent-pattern-algorithms)
  - [Association Algorithm Benchmark](#association-algorithm-benchmark)
  - [Key Association Rules Discovered](#key-association-rules-discovered)
- [Comparative Synthesis & Key Takeaways](#-comparative-synthesis--key-takeaways)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart Guide](#-installation--quickstart-guide)
- [Reproducibility & Execution](#-reproducibility--execution)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🎯 Executive Summary

The modern Olympic Games provide a rich historical laboratory of human performance, biometric evolution, and geopolitical dynamics. This project applies data mining methodologies to answer three core questions:
1. **Predictive Performance**: Can biological measurements (`Age`, `Height`, `Weight`) along with contextual descriptors (`Sport`, `Season`, `Year`, `Sex`) predict whether an athlete secures a podium finish (`Won_Medal`)?
2. **Biometric Topology**: Do athletes cluster into natural, well-defined physical archetypes across disparate sporting events, and how do density-based versus centroid-based clustering algorithms perform on standardized biometric data?
3. **Behavioral Co-occurrence**: What frequent attribute itemsets and association rules emerge across sports disciplines, genders, medal categories, and national representations? Furthermore, how do advanced hash-based frequent itemset algorithms (**PCY**, **Multistage PCY**, **Multihash PCY**) compare against canonical approaches (**Apriori**, **FP-Growth**)?

---

## 🏗 Project Architecture & KDD Pipeline

The end-to-end data mining pipeline follows the classical **Cross-Industry Standard Process for Data Mining (CRISP-DM)** and KDD methodology:

```
[Raw Olympic Athlete CSV (~271k records)]
                   │
                   ▼
       1. Ingestion & Audit
       ├── Deduplication & Whitespace Stripping
       ├── Anomaly Detection (<= 0 biometric values -> NaN)
       └── Target Engineering: Won_Medal in {0, 1}
                   │
                   ├────────────────────────┬─────────────────────────┐
                   ▼                        ▼                         ▼
        2. Supervised Learning   3. Unsupervised Clustering  4. Association Mining
        ├── Stratified 80/20     ├── Bio Subset (Age/Ht/Wt)  ├── Categorical Sample
        ├── Median/Mode Impute   ├── Median Impute + Scaler  ├── Bucket Top-25
        ├── One-Hot Encoding     ├── Silhouette & Elbow      ├── Basket Conversion
        ├── Model Zoo:           ├── Model Zoo:              ├── Model Zoo:
        │   • ID3-style (Entropy)│   • K-Means (K=4)         │   • Apriori
        │   • CART (Gini)        │   • Agglomerative (Ward)  │   • FP-Growth
        │   • Random Forest      │   • DBSCAN (eps, minPts)  │   • PCY
        │   • XGBoost            │                           │   • Multistage PCY
        │   • Gaussian Naïve Bayes                           │   • Multihash PCY
        └── Comprehensive Metrics ── PCA 2D/3D Projection   └── Hash & Pair Efficiency
                   │                        │                         │
                   └────────────────────────┼─────────────────────────┘
                                            ▼
                           5. Evaluation & Exported Insights
                           ├── Benchmark CSVs in Results/
                           └── Feature Importance & Trade-off Audits
```

---

## 📊 Dataset Overview & Preprocessing

### Raw Attributes
The project leverages the historical Olympic dataset spanning **1896 to 2016** (sourced from public archives):
- **Identification**: `ID`, `Name`
- **Demographics**: `Sex` (M/F), `Age` (years), `Height` (cm), `Weight` (kg)
- **Affiliation**: `Team`, `NOC` (National Olympic Committee 3-letter code)
- **Event Context**: `Games`, `Year`, `Season` (Summer/Winter), `City`, `Sport`, `Event`
- **Outcome**: `Medal` (`Gold`, `Silver`, `Bronze`, or `NaN`)

### Data Cleaning & Integrity Auditing
1. **Deduplication**: Exact duplicate records representing redundant athlete-event registrations were systematically purged.
2. **String Normalization**: All categorical features underwent whitespace stripping and explicit unicode string casting.
3. **Biological Validation**: Non-positive physiological values ($\le 0$) in `Age`, `Height`, and `Weight` were flagged as invalid entries and coerced to missing values (`np.nan`).
4. **Missing Value Categorization**: For descriptive exploration, missing medals were coded explicitly as `"No Medal"`, and missing contextual fields were flagged as `"Unknown"`.

### Target & Feature Engineering
- **Target Variable (`Won_Medal`)**: Formulated as a binary classification target:
  $$\text{Won\_Medal} = \begin{cases} 1, & \text{if } \text{Medal} \in \{\text{Gold}, \text{Silver}, \text{Bronze}\} \\ 0, & \text{if } \text{Medal is NaN} \end{cases}$$
- **Class Distribution**: The dataset exhibits extreme natural class imbalance:
  - **No Medal ($0$)**: $\approx 85.3\%$
  - **Won Medal ($1$)**: $\approx 14.7\%$

---

## 🤖 Supervised Classification: Predicting Olympic Medalists

### Feature Processing Pipeline
To guarantee complete isolation against data leakage, all imputation and feature encodings are encapsulated inside a `scikit-learn` `ColumnTransformer` embedded within model pipelines:
- **Numerical Features** (`Age`, `Height`, `Weight`, `Year`): Imputed using **Median** statistics and scaled via `StandardScaler` ($\mu = 0, \sigma = 1$).
- **Categorical Features** (`Sex`, `Season`, `Sport`): Imputed using **Most Frequent** strategy and transformed via `OneHotEncoder(handle_unknown="ignore", min_frequency=20)` to prevent rare category explosion.
- **Partitioning**: 80% Training set and 20% Testing set via stratified sampling (`stratify=y`, `random_state=42`).

### Class Imbalance Strategy
Given the $85:15$ skew, naive accuracy is a deceptive metric. Models such as Decision Trees and Random Forests were configured with cost-sensitive weighting (`class_weight="balanced"` / `class_weight="balanced_subsample"`), forcing algorithms to penalize false negatives on medalists heavily.

### Evaluated Classifiers
1. **ID3-Style Decision Tree**: Information Gain criterion (`criterion="entropy"`, `max_depth=12`, `min_samples_leaf=20`, `class_weight="balanced"`).
2. **CART Decision Tree**: Gini Impurity criterion (`criterion="gini"`, `max_depth=12`, `min_samples_leaf=20`, `class_weight="balanced"`).
3. **Random Forest Classifier**: Bagging ensemble of 150 randomized decision trees (`n_estimators=150`, `max_depth=16`, `min_samples_leaf=10`, `class_weight="balanced_subsample"`, `n_jobs=-1`).
4. **XGBoost (Extreme Gradient Boosting)**: Gradient boosted trees with regularized objective (`n_estimators=250`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `tree_method="hist"`).
5. **Gaussian Naïve Bayes**: Baseline probabilistic classifier utilizing dense array conversion via `FunctionTransformer`.

### Classification Benchmark Results

Evaluated on the held-out test partition ($20\%$):

| Algorithm | Accuracy | Precision | Recall | F1-Score | Training Time (sec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.8581** | **0.7270** | 0.0606 | 0.1119 | 3.80 s |
| **Naïve Bayes** | 0.8341 | 0.3192 | 0.1104 | 0.1640 | **0.90 s** |
| **Random Forest** | 0.6920 | 0.2641 | 0.6097 | **0.3686** | 70.05 s |
| **ID3-style (Entropy)** | 0.5894 | 0.2188 | 0.6940 | 0.3327 | 2.53 s |
| **CART (Gini)** | 0.5825 | 0.2162 | **0.6973** | 0.3300 | 2.54 s |

#### Analytical Insights:
- **Precision vs. Recall Trade-off**: Unweighted **XGBoost** targets global log-loss minimization, achieving high overall accuracy ($85.8\%$) and precision ($72.7\%$), but misses low-support medalists (low recall).
- **Balanced Ensembles Win on F1**: **Random Forest** with balanced subsampling establishes the highest harmonic mean (**F1 = 0.3686**), capturing over $60.9\%$ of actual medalists while filtering noise.
- **Decision Trees for High Sensitivity**: Both ID3 and CART identify nearly $70\%$ of all medal winners, serving as effective high-recall filters.

### Feature Importance Analysis

Extracted from the optimized Random Forest ensemble (`random_forest_feature_importance.csv`):

| Rank | Feature | Relative Importance | Interpretive Domain Insight |
| :---: | :--- | :---: | :--- |
| **1** | `num__Year` | **22.80%** | Evolution of Olympic competitiveness, squad sizes, and event expansions over 120 years. |
| **2** | `num__Weight` | **15.33%** | Essential physiological indicator across weight-class and leverage-heavy sports. |
| **3** | `num__Height` | **13.61%** | Critical physical requirement for rowing, basketball, athletics, and swimming. |
| **4** | `num__Age` | **11.27%** | Peak physical conditioning window vs. sport-specific longevity. |
| **5** | `cat__Sport_Gymnastics` | **4.28%** | Distinct age/weight biometric signature strongly correlating with performance. |
| **6** | `cat__Sport_Athletics` | **3.68%** | Largest medal category with diversified specialization profiles. |
| **7** | `cat__Sport_Rowing` | **3.03%** | High height/weight ratio strongly indicative of elite rowing competitors. |

---

## 🔍 Unsupervised Clustering: Athlete Demographic Segmentation

### Clustering Preprocessing
- **Feature Subspace**: Biological attributes (`Age`, `Height`, `Weight`) were extracted, median-imputed, and standardized.
- **Sampling**: A representative stratified sample of $N = 15,000$ athletes was selected to optimize computational geometry calculations and distance matrices.
- **Optimal Cluster Validation**: Evaluated via the **Elbow Method (Within-Cluster Sum of Squares / Inertia)** and **Silhouette Coefficients** across $K \in [2, 7]$. Optimal clustering stability was reached at $K = 4$.

### Clustering Algorithms
1. **K-Means Clustering**: Partitioning method minimizing centroid inertia with $K=4$ clusters.
2. **Agglomerative Hierarchical Clustering**: Bottom-up hierarchical tree construction evaluated with **Ward's Linkage** to minimize within-cluster variance.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Density reachability analysis using Euclidean neighborhood radius $\varepsilon = 0.65$ and $MinSamples = 12$.

### Clustering Benchmark Results

| Algorithm | Partitioned Clusters | Noise Points Identified | Silhouette Coefficient |
| :--- | :---: | :---: | :---: |
| **K-Means ($K=4$)** | 4 | 0 | **0.3397** |
| **Agglomerative Hierarchical** | 4 | 0 | 0.2932 |
| **DBSCAN** | 1 | 101 | N/A (single continuous dense core) |

#### Demographic Insights:
- **Cluster 0 (Endurance & Agility)**: Average height, lower body mass, mid-20s (e.g., long-distance runners, gymnasts).
- **Cluster 1 (Power & Heavyweight)**: High mass, tall stature (e.g., heavyweight weightlifting, shot put, super-heavyweight combat).
- **Cluster 2 (Tall & Lean Leverage)**: Exceptional height with moderate weight (e.g., basketball, volleyball, rowing, high jump).
- **Cluster 3 (Veteran Competitors)**: Athletes with ages exceeding 35–45 (e.g., equestrian disciplines, shooting, curling).
- **DBSCAN Observation**: Human athletic physiology forms a continuous multivariate Gaussian distribution without natural empty voids; hence, DBSCAN identifies a single dense mega-cluster while flagging 101 extreme biometric outliers.

---

## 🔗 Association Rule Mining: Discovering Hidden Patterns

### Transaction Construction
Records were converted into transactional baskets representing categorical co-occurrences:
- Categories: `Sport`, `Sex`, `Team`, `Season`, `Medal_Category`
- Vocabulary Control: High-cardinality features were bounded by retaining the **Top-25 most popular sports and teams**, grouping remaining categories into `"Other Sport"` and `"Other Team"`.
- Transaction Format:
  `['Sport=Gymnastics', 'Sex=M', 'Team=Other Team', 'Season=Summer', 'Medal=No Medal']`
- Parameters: Evaluated with minimum support threshold $s = 0.015$ ($1.5\%$) and maximum itemset length $k = 3$.

### Frequent Pattern Algorithms
1. **Apriori**: Classical breadth-first level-wise candidate generation exploiting the downward closure property.
2. **FP-Growth**: High-performance mining via recursive traversal of a compressed prefix tree (**FP-Tree**) without candidate pair generation.
3. **PCY (Park-Chen-Yu)**: Reduces candidate $C_2$ pairs using a hash table bucket array in Pass 1 and frequent-bucket bitmapping in Pass 2.
4. **Multistage PCY**: Extends PCY by applying sequential independent hash passes with filtered candidate pairs.
5. **Multihash PCY**: Utilizes simultaneous independent hash functions into multiple independent hash tables in a single pass.

### Association Algorithm Benchmark

Evaluated over $30,000$ standardized transactions:

| Algorithm | Mining Scope | Frequent Sets / Pairs | Execution Time (sec) | Relative Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Multihash PCY** | Frequent Pairs ($C_2$) | 132 | **0.4754 s** | **2.60x** vs PCY |
| **Apriori** | All Itemsets ($k \le 3$) | 308 | **0.4878 s** | **38.90x** vs FP-Growth |
| **Multistage PCY** | Frequent Pairs ($C_2$) | 132 | 0.5605 s | 2.20x vs PCY |
| **PCY** | Frequent Pairs ($C_2$) | 132 | 1.2344 s | Baseline Hash |
| **FP-Growth** | All Itemsets ($k \le 3$) | 308 | 18.9745 s | Dense tree traversal |

### Key Association Rules Discovered

Extracted rules reveal notable historical dynamics:
- **Summer Games Predominance**: $\approx 81.8\%$ of athlete participation is concentrated in Summer editions.
- **Participation Demographics**: `{Season=Summer, Sex=M}` accounts for $60.1\%$ aggregate support across transactions.
- **Sport-Specific Specialization**: High confidence rules associate team representations with consistent multi-event dominance (e.g., athletics and gymnastics delegations).

---

## 📈 Comparative Synthesis & Key Takeaways

```
                                  DATA MINING TAKEAWAYS
┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│     SUPERVISED PREDICTION       │     BIOMETRIC SEGMENTATION      │       ASSOCIATION MINING        │
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ • Random Forest achieved top    │ • Athlete biometrics segment    │ • Multihash PCY achieved the    │
│   F1-score (0.3686) with        │   into 4 distinct archetypes    │   fastest runtime (0.475s).     │
│   balanced subsampling.         │   (K-Means Silhouette = 0.34).  │                                 │
│ • Chronological 'Year' &        │ • DBSCAN confirmed continuity   │ • Both Apriori and FP-Growth    │
│   biometrics ('Weight',         │   in human physiology, flagging │   consistently identified 308   │
│   'Height') dominate splits.    │   101 distinct outliers.        │   frequent itemsets.            │
└─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

1. **Class Balancing is Mandatory**: Standard accuracy is uninformative for competitive athletic datasets where $>85\%$ of participants leave without a medal. Balanced tree ensembles are required for actionable sensitivity.
2. **Context Outweighs Raw Physiology Alone**: While Height and Weight dictate athletic classification, participation Year reflects systemic changes (training advancements, geopolitical delegations, event additions).
3. **Algorithm Suitability Depends on Sparsity**: Hash-based optimizations like Multihash PCY show substantial latency reductions for pair screening, demonstrating the utility of memory-efficient hashing in constrained environments.

---

## 📁 Repository Structure

```
Data-Mining-of-Olympic-Athlete-Events-History/
│
├── Data Mining Notebook.ipynb         # Master Jupyter/Colab notebook with all 27 KDD steps
├── LICENSE                            # Open-source MIT License
├── README.md                          # Comprehensive technical documentation
│
└── Results/                           # Benchmark artifacts and outputs
    ├── apriori_frequent_itemsets.csv          # 308 frequent itemsets generated via Apriori
    ├── fpgrowth_frequent_itemsets.csv         # Verified frequent itemsets from FP-Growth
    ├── pcy_frequent_pairs.csv                 # 132 frequent pairs mined using PCY
    ├── association_comparison.csv             # Execution times across all 5 association models
    ├── classification_comparison.csv          # Precision, Recall, F1, Accuracy for 5 classifiers
    ├── clustering_comparison.csv              # Silhouette scores and cluster counts
    └── random_forest_feature_importance.csv   # Normalized Gini importance rankings
```

---

## 🚀 Installation & Quickstart Guide

### Prerequisites
- Python `3.9` or higher
- Jupyter Notebook, JupyterLab, or Google Colab

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Data-Mining-of-Olympic-Athlete-Events-History.git
cd Data-Mining-of-Olympic-Athlete-Events-History
```

### 2. Set Up a Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn xgboost mlxtend scipy
```

---

## 💻 Reproducibility & Execution

To execute the notebook locally:

```bash
jupyter notebook "Data Mining Notebook.ipynb"
```

Or open the notebook directly in **Google Colab**:
1. Upload `Data Mining Notebook.ipynb` to Google Drive.
2. Select **Open with Google Colaboratory**.
3. Execute all cells (`Runtime -> Run all`).
4. Output tables will automatically regenerate and save into your local execution directory.

---

## 👥 Contributors

- **Harshal Pednekar** — Lead Author & Project Contributor ([GitHub](https://github.com/harshalnnpednekar))

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this codebase with proper attribution.
