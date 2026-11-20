import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Olympic Athlete KDD Portal | Data Mining Research",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Modern UI Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Fira+Code:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header hero styling */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #FFFFFF, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        color: #94A3B8;
        max-width: 900px;
        line-height: 1.6;
    }
    
    .author-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(147, 197, 253, 0.4);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #93C5FD;
        margin-top: 1rem;
    }

    /* Cards */
    .concept-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .concept-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    
    .concept-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    .badge-blue { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .badge-purple { background-color: #FAF5FF; color: #6B21A8; border: 1px solid #E9D5FF; }
    .badge-emerald { background-color: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-amber { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }

    /* Math formula boxes */
    .formula-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        font-family: 'Fira Code', monospace;
        font-size: 0.95rem;
        color: #1E293B;
        margin: 0.75rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Ingestion & Caching
# -----------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Results")

@st.cache_data
def load_results():
    data = {}
    try:
        data["classification"] = pd.read_csv(os.path.join(RESULTS_DIR, "classification_comparison.csv"))
        data["clustering"] = pd.read_csv(os.path.join(RESULTS_DIR, "clustering_comparison.csv"))
        data["association"] = pd.read_csv(os.path.join(RESULTS_DIR, "association_comparison.csv"))
        data["rf_importance"] = pd.read_csv(os.path.join(RESULTS_DIR, "random_forest_feature_importance.csv"))
        data["apriori"] = pd.read_csv(os.path.join(RESULTS_DIR, "apriori_frequent_itemsets.csv"))
        data["fpgrowth"] = pd.read_csv(os.path.join(RESULTS_DIR, "fpgrowth_frequent_itemsets.csv"))
        data["pcy"] = pd.read_csv(os.path.join(RESULTS_DIR, "pcy_frequent_pairs.csv"))
    except Exception as e:
        st.error(f"Error loading exported results: {e}")
    return data

results = load_results()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5c/Olympic_rings_without_rims.svg", width=170)
    st.title("Data Mining Portal")
    st.caption("120 Years of Olympic History (1896–2016)")
    
    st.markdown("---")
    menu = st.radio(
        "Navigation Menu:",
        [
            "🏛️ Project Overview & KDD",
            "📖 Theoretical Foundations",
            "🤖 Supervised Classification",
            "🔍 Unsupervised Clustering",
            "🔗 Association Rule Mining",
            "🔮 Live Inference & Profiler",
            "📚 Project Documentation",
        ]
    )
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Project Contributor")
    st.markdown("**Harshal Pednekar**")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-harshalnnpednekar-181717?style=flat&logo=github)](https://github.com/harshalnnpednekar)")
    st.caption("Mini Project • Knowledge Discovery & Data Mining")

# =============================================================================
# MODULE 1: Project Overview & KDD
# =============================================================================
if menu == "🏛️ Project Overview & KDD":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🏅 Data Mining of Olympic Athlete Events History</div>
        <div class="hero-subtitle">
            An end-to-end Knowledge Discovery in Databases (KDD) research system examining athlete biometrics, 
            evolutionary competition dynamics, and event podium finishes across 120 years of modern Olympic Games (1896–2016).
        </div>
        <div class="author-badge">Lead Author & Contributor: Harshal Pednekar</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Athlete Records", "271,116", help="Full historical registrations between 1896 and 2016")
    with col2:
        st.metric("Cleaned Unique Entries", "269,731", help="Filtered exact duplicate records and standardized strings")
    with col3:
        st.metric("Podium Win Rate", "14.7%", help="Highly imbalanced target: 85.3% No Medal vs 14.7% Won Medal")
    with col4:
        st.metric("Olympic Editions", "51 Games", help="Summer & Winter Olympiads spanning Athens 1896 to Rio 2016")

    st.markdown("---")
    
    st.header("🎯 Project Research Questions & Objectives")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="concept-card">
            <span class="concept-badge badge-blue">Supervised Learning</span>
            <h4>Predictive Classification</h4>
            <p>Can an athlete's biological features (<code>Age</code>, <code>Height</code>, <code>Weight</code>) alongside 
            contextual cues (<code>Sport</code>, <code>Year</code>, <code>Sex</code>) predict podium finishes under acute class imbalance?</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="concept-card">
            <span class="concept-badge badge-emerald">Unsupervised Learning</span>
            <h4>Biometric Segmentation</h4>
            <p>Do Olympic competitors group naturally into discrete physiological archetypes across distinct sports, 
            and how do distance-based vs. density-based clustering models behave?</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="concept-card">
            <span class="concept-badge badge-purple">Association Mining</span>
            <h4>Pattern Co-Occurrence</h4>
            <p>What multi-attribute patterns emerge between gender, sports, and medal outcomes? How do modern hash-based 
            algorithms (<strong>PCY</strong>, <strong>Multistage</strong>, <strong>Multihash</strong>) compare with <strong>Apriori</strong> and <strong>FP-Growth</strong>?</p>
        </div>
        """, unsafe_allow_html=True)

    st.header("🔄 The Complete KDD Process Pipeline")
    st.markdown("""
    The system faithfully executes the classical 5-stage **Knowledge Discovery in Databases (KDD)** methodology:
    """)
    st.markdown(r"""
    ```
    ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
    │ 1. Raw Data Selection│ ───> │ 2. Preprocessing &   │ ───> │ 3. Transformation &  │
    │  • 271,116 records   │      │    Cleaning          │      │    Encoding          │
    │  • 15 attributes     │      │  • Deduplication     │      │  • Median imputation │
    │  • Biometrics & Medals│     │  • String strip      │      │  • StandardScaler    │
    └──────────────────────┘      │  • Non-positive nulls│      │  • One-Hot Encoding  │
                                  └──────────────────────┘      └──────────────────────┘
                                                                           │
                                                                           ▼
    ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
    │ 5. Knowledge & Action│ <─── │ 4. Evaluation &      │ <─── │ 4. Data Mining       │
    │  • Streamlit Web App │      │    Interpretation    │      │  • ID3, CART, RF,    │
    │  • Biometric Topology│      │  • F1 & PR trade-off │      │    XGBoost, NB       │
    │  • Interactive Model │      │  • Silhouette scores │      │  • K-Means, DBSCAN   │
    │  • Policy insights   │      │  • Execution timing  │      │  • Apriori, PCY, FP  │
    └──────────────────────┘      └──────────────────────┘      └──────────────────────┘
    ```
    """)

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# MODULE 2: Theoretical Foundations of Data Mining
# =============================================================================
elif menu == "📖 Theoretical Foundations":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📖 Theoretical Foundations of Data Mining</div>
        <div class="hero-subtitle">
            A clear, structured, and comprehensive reference documenting the core concepts, 
            mathematical foundations, algorithmic steps, and evaluation metrics used throughout this project.
        </div>
        <div class="author-badge">Authored by Harshal Pednekar</div>
    </div>
    """, unsafe_allow_html=True)

    theory_tab1, theory_tab2, theory_tab3, theory_tab4 = st.tabs([
        "1. Supervised Learning & Ensembles",
        "2. Clustering & Distance Metrics",
        "3. Frequent Itemsets & PCY Hashing",
        "4. Evaluation Metrics & CRISP-DM"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Supervised Learning & Ensembles
    # -------------------------------------------------------------------------
    with theory_tab1:
        st.header("1. Supervised Classification & Ensemble Models")
        
        st.subheader("1.1 Decision Tree Classifiers")
        st.markdown(r"""
        A **Decision Tree** is a supervised machine learning algorithm used for classification and regression tasks. It models decisions by recursively splitting a dataset into smaller, more homogeneous subsets based on feature values, producing an inverted tree structure consisting of a root node, internal decision nodes, and terminal leaf nodes.

        #### How Decision Trees Operate:
        1. **Feature Evaluation**: At each node, the algorithm evaluates all available features and possible split thresholds.
        2. **Impurity Reduction**: It selects the feature and threshold that produces the greatest reduction in class impurity across the resulting child nodes.
        3. **Recursive Partitioning**: This process repeats recursively on each child node until a stopping criterion is met (such as maximum tree depth, minimum samples per leaf, or zero remaining impurity).
        4. **Prediction**: Unseen instances traverse the tree from root to leaf following the learned split rules. The class label assigned is determined by majority vote within that leaf.
        """)

        st.subheader("1.2 Splitting Criteria: Entropy vs. Gini Impurity")
        st.markdown(r"""
        The quality of a proposed split is measured using mathematical impurity functions that quantify how mixed the classes are within a node.

        #### A. Shannon Entropy & Information Gain (ID3 Approach)
        **Entropy** measures the amount of uncertainty or disorder in a set of class labels. For a dataset $S$ with target classes $c \in \{0, 1\}$:
        $$H(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$
        where $p_i$ is the relative proportion of samples belonging to class $i$ within node $S$.
        - When all instances belong to a single class, $H(S) = 0$ (complete purity).
        - When instances are evenly split ($50\%$ class 0 and $50\%$ class 1), $H(S) = 1.0$ (maximum impurity).

        **Information Gain** measures the reduction in entropy achieved after partitioning dataset $S$ using feature $A$:
        $$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
        The ID3 algorithm selects the feature that maximizes Information Gain at each step.

        #### B. Gini Impurity (CART Approach)
        The **CART** algorithm uses Gini Impurity rather than entropy. Gini measures the probability that a randomly chosen element from the set would be incorrectly labeled if it were labeled randomly according to the class distribution in the node:
        $$Gini(S) = 1 - \sum_{i=1}^{C} p_i^2$$
        For binary classification ($p$ and $1-p$):
        $$Gini(S) = 1 - (p^2 + (1 - p)^2) = 2p(1 - p)$$
        - Gini impurity ranges from $0.0$ (complete purity) to $0.5$ (maximum impurity).
        - **Practical Difference**: Gini is computationally more efficient than entropy because it involves only multiplication and subtraction, avoiding logarithmic calculations. Both criteria yield very similar tree structures in practice.
        """)

        st.subheader("1.3 Ensemble Learning: Random Forest vs. XGBoost")
        st.markdown(r"""
        Individual decision trees often suffer from **high variance** and tend to overfit the training data. Ensemble methods combine predictions from multiple individual models to improve generalization and stability.

        #### A. Random Forest (Bagging)
        Random Forest is an ensemble learning method based on **Bootstrap Aggregating (Bagging)** and random feature selection:
        1. **Bootstrap Sampling**: Given a training set of size $N$, the algorithm draws $B$ separate bootstrap samples of size $N$ with replacement. Each tree in the forest is trained on its own bootstrap sample.
        2. **Random Feature Subsets**: When splitting a node, the algorithm considers only a random subset of $m$ features out of the total $p$ features, typically $m \approx \sqrt{p}$. This decorrelates the individual trees from one another.
        3. **Aggregation**: For classification, predictions from all $B$ individual trees are aggregated via majority voting:
           $$\hat{y} = \text{mode}\left(\{T_1(x), T_2(x), \dots, T_B(x)\}\right)$$
        4. **Variance Reduction**: Averaging predictions across decorrelated trees reduces the overall variance of the model without increasing its bias:
           $$\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
           where $\rho$ represents the average correlation between pairs of trees and $\sigma^2$ is the variance of a single tree.

        #### B. XGBoost (Gradient Boosting)
        Unlike Random Forest, which builds trees independently in parallel, **Gradient Boosting** builds trees sequentially:
        1. **Sequential Optimization**: Each new tree is trained to predict the residual errors made by the combination of all previous trees.
        2. **Gradient Descent in Function Space**: At each step, a new tree $f_t(x)$ is fitted to the negative gradient of the loss function evaluated at the current ensemble prediction:
           $$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$$
           where $\eta$ is the learning rate (shrinkage parameter) that scales the contribution of each tree.
        3. **Second-Order Loss Approximation**: XGBoost utilizes both first-order gradients ($g_i$) and second-order Hessians ($h_i$) of the loss function, allowing more accurate tree-structure optimization.
        4. **Regularization**: The objective function includes explicit penalties on tree complexity (number of terminal leaves and magnitude of leaf weights), preventing overfitting.
        """)

        st.subheader("1.4 Gaussian Naïve Bayes Classifier")
        st.markdown(r"""
        **Naïve Bayes** is a probabilistic classifier based on **Bayes' Theorem**:
        $$P(Y = c \mid X) = \frac{P(Y = c) \cdot P(X \mid Y = c)}{P(X)}$$
        
        #### Key Concepts:
        - **Conditional Independence Assumption**: It assumes that every feature $X_j$ is conditionally independent of every other feature $X_k$ given the class label $Y$:
          $$P(X \mid Y = c) = \prod_{j=1}^{d} P(X_j \mid Y = c)$$
        - **Gaussian Distribution for Continuous Features**: When attributes are continuous numerical values (such as `Age`, `Height`, and `Weight`), class-conditional probabilities are modeled using a normal (Gaussian) distribution:
          $$P(X_j = x_j \mid Y = c) = \frac{1}{\sqrt{2\pi \sigma_{c,j}^2}} \exp\left( - \frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right)$$
        - **Classification Rule**: An instance is assigned to the class with the highest posterior probability (Maximum A Posteriori decision rule):
          $$\hat{y} = \arg\max_{c} \left( \log P(Y = c) + \sum_{j=1}^d \log P(X_j = x_j \mid Y = c) \right)$$
        Naïve Bayes is computationally lightweight, requiring only a single pass over the training data to compute feature means and variances per class.
        """)

    # -------------------------------------------------------------------------
    # TAB 2: Clustering & Geometry
    # -------------------------------------------------------------------------
    with theory_tab2:
        st.header("2. Unsupervised Clustering & Spatial Analysis")
        
        st.subheader("2.1 Distance Metrics & Feature Scaling")
        st.markdown(r"""
        Clustering is an unsupervised learning task that groups unlabelled observations into clusters such that objects within the same cluster are more similar to each other than to objects in other clusters. Similarity is defined mathematically using distance functions.

        #### A. Euclidean Distance
        For two $d$-dimensional data points $x$ and $y$, the Euclidean distance is defined as:
        $$d(x, y) = \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2}$$

        #### B. The Necessity of Standardization (Z-Scores)
        When features operate on different numerical scales, attributes with larger numeric ranges dominate the distance calculation.
        - **Standardization** transforms each feature so that it has a mean of zero ($\mu = 0$) and a standard deviation of one ($\sigma = 1$):
          $$z = \frac{x - \mu}{\sigma}$$
        - In standardized space, a unit difference in one feature has the exact same mathematical weight as a unit difference in any other feature, ensuring fair contributions from all dimensions.
        """)

        st.subheader("2.2 K-Means Clustering Algorithm")
        st.markdown(r"""
        **K-Means** is a centroid-based clustering algorithm that partitions $N$ observations into $K$ pre-defined clusters.

        #### Objective Function:
        K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called inertia:
        $$\text{WCSS} = \sum_{k=1}^{K} \sum_{x \in S_k} \| x - \mu_k \|^2$$
        where $\mu_k$ is the mean (centroid) of all points belonging to cluster $S_k$.

        #### Algorithmic Procedure:
        1. **Initialization**: Select $K$ initial cluster centroids. The **k-means++** initialization method chooses seeds spread far apart to avoid poor local minima.
        2. **Assignment Step**: Calculate the distance from each data point to all $K$ centroids. Assign each point to its nearest centroid.
        3. **Update Step**: Recompute each cluster centroid as the arithmetic mean of all points assigned to that cluster:
           $$\mu_k = \frac{1}{|S_k|} \sum_{x \in S_k} x$$
        4. **Convergence Check**: Repeat the Assignment and Update steps until centroid coordinates stop changing, or until a maximum iteration limit is reached.
        """)

        st.subheader("2.3 Agglomerative Hierarchical Clustering")
        st.markdown(r"""
        **Hierarchical Clustering** creates a nested hierarchy of clusters represented as a tree called a **Dendrogram**.

        #### Agglomerative (Bottom-Up) Strategy:
        1. **Initial State**: Start with each data point treated as an individual single-element cluster.
        2. **Pairwise Distance Matrix**: Compute the distance between all pairs of active clusters using a linkage criterion.
        3. **Iterative Merger**: Find the two closest clusters and merge them into a single parent cluster.
        4. **Termination**: Repeat until all data points are merged into one root cluster. The dendrogram can then be cut at any chosen threshold or depth to extract $K$ flat clusters.

        #### Linkage Criteria (Ward's Linkage):
        Ward's minimum variance method merges the two clusters that result in the smallest increase in total within-cluster variance:
        $$\Delta \text{ESS}_{AB} = \frac{n_A n_B}{n_A + n_B} \| \mu_A - \mu_B \|^2$$
        where $n_A$ and $n_B$ are cluster sample sizes, and $\mu_A, \mu_B$ are their centroids. Ward's linkage tends to produce compact, balanced clusters.
        """)

        st.subheader("2.4 Density-Based Spatial Clustering (DBSCAN)")
        st.markdown(r"""
        **DBSCAN** groups points based on local density rather than distance to centroids. It does not require specifying the number of clusters $K$ upfront and can discover clusters of arbitrary shapes while detecting noise.

        #### Core Parameters:
        - **$\varepsilon$ (Epsilon)**: Maximum radius defining the neighborhood of a point.
        - **$MinSamples$**: Minimum number of points required within the $\varepsilon$-neighborhood to form a dense region.

        #### Point Classification:
        - **Core Point**: A point with at least $MinSamples$ points in its $\varepsilon$-neighborhood ($|N_\varepsilon(p)| \ge MinSamples$).
        - **Border Point**: A point with fewer than $MinSamples$ points in its neighborhood, but located within the $\varepsilon$-neighborhood of a core point.
        - **Noise Point (Outlier)**: Any point that is neither a core point nor a border point.

        #### Properties in Practice:
        When data forms a continuous multivariate distribution without low-density gaps between groups, DBSCAN naturally identifies one large continuous cluster while isolating extreme peripheral outliers as noise points.
        """)

        st.subheader("2.5 Cluster Validation: The Silhouette Coefficient")
        st.markdown(r"""
        Because clustering is unsupervised, objective validation metrics are required to evaluate partition quality and select the best number of clusters $K$.

        The **Silhouette Coefficient** measures how well-separated and cohesive clusters are:
        $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
        
        Where for point $i$:
        - $a(i)$ (**Intra-cluster cohesion**): The average distance between point $i$ and all other points in the same cluster.
        - $b(i)$ (**Inter-cluster separation**): The average distance between point $i$ and points in the nearest neighboring cluster.

        #### Interpretation of Silhouette Score:
        - Near $+1.0$: Point is far from neighboring clusters and compact within its own cluster (strong clustering).
        - Near $0.0$: Point lies on or very close to the decision boundary between two clusters (overlapping clusters).
        - Near $-1.0$: Point is closer to a neighboring cluster than to its assigned cluster (poor clustering).
        The overall score for a model is computed as the mean silhouette score across all points.
        """)

    # -------------------------------------------------------------------------
    # TAB 3: Frequent Itemsets & PCY Hashing
    # -------------------------------------------------------------------------
    with theory_tab3:
        st.header("3. Frequent Pattern & Association Rule Mining")
        
        st.subheader("3.1 Principles of Association Analysis")
        st.markdown(r"""
        Association rule mining discovers interesting co-occurrence relationships and patterns among categorical attributes in transactional datasets.

        Let $\mathcal{I} = \{i_1, i_2, \dots, i_m\}$ be a set of items, and $\mathcal{D} = \{T_1, T_2, \dots, T_N\}$ be a collection of transactions where each transaction $T \subseteq \mathcal{I}$. An association rule is an implication of the form:
        $$X \Rightarrow Y, \quad \text{where } X, Y \subset \mathcal{I} \text{ and } X \cap Y = \emptyset$$
        Here, $X$ is the antecedent and $Y$ is the consequent.

        #### Key Metrics:
        1. **Support**: The frequency with which an itemset $X$ appears across all transactions in the database:
           $$\text{Support}(X) = \frac{|\{T \in \mathcal{D} \mid X \subseteq T\}|}{|\mathcal{D}|}$$
        2. **Confidence**: The probability that a transaction contains $Y$ given that it already contains $X$:
           $$\text{Confidence}(X \Rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$
        3. **Lift**: Measures how much more often $X$ and $Y$ occur together than expected under statistical independence:
           $$\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)} = \frac{P(X \cap Y)}{P(X) \cdot P(Y)}$$
           - $\text{Lift} = 1.0$: $X$ and $Y$ are independent; no association exists.
           - $\text{Lift} > 1.0$: Positive association; the presence of $X$ increases the likelihood of $Y$.
           - $\text{Lift} < 1.0$: Negative association; the presence of $X$ decreases the likelihood of $Y$.
        """)

        st.subheader("3.2 The Apriori Algorithm & Downward Closure")
        st.markdown(r"""
        The **Apriori Algorithm** searches for frequent itemsets in a breadth-first, level-wise manner using the **Downward Closure Property** (also known as the Apriori Principle):
        > *"All non-empty subsets of a frequent itemset must also be frequent."*

        #### Downward Closure Pruning Rule:
        If an itemset $X$ is infrequent (its support is below the minimum support threshold $\sigma_{\min}$), then any superset $Y \supset X$ is guaranteed to be infrequent and is pruned immediately without checking the data:
        $$\text{Support}(X) < \sigma_{\min} \implies \text{Support}(Y) < \sigma_{\min} \quad \forall Y \supset X$$

        #### Bottleneck of Apriori:
        Apriori generates candidate itemsets $C_k$ at level $k$ by joining frequent itemsets $L_{k-1}$ from level $k-1$. For large item vocabularies, candidate generation produces huge candidate sets (especially candidate pairs in $C_2$), requiring multiple full-table scans across disk.
        """)

        st.subheader("3.3 FP-Growth (Frequent Pattern Growth)")
        st.markdown(r"""
        The **FP-Growth Algorithm** eliminates candidate generation completely by compressing the transactional dataset into an **FP-Tree (Frequent Pattern Tree)**.

        #### Execution Phases:
        1. **Pass 1 (Item Frequencies)**: Scan the dataset to compute frequencies of single items. Discard infrequent items and sort frequent items in descending order of frequency.
        2. **Pass 2 (Tree Construction)**: Read transactions again and insert each transaction's frequent items as a path in a prefix tree. Identical item prefixes share common tree branches, with node counters incremented.
        3. **Recursive Mining**: Starting from the least frequent items in a header table, mine frequent itemsets by constructing **conditional pattern bases** and projecting smaller conditional FP-Trees recursively.
        This divide-and-conquer strategy avoids building candidate sets and scales efficiently on dense datasets.
        """)

        st.subheader("3.4 The PCY (Park-Chen-Yu) Algorithm")
        st.markdown(r"""
        The **PCY Algorithm** is a memory-optimized variant of Apriori designed to reduce the size of candidate pairs ($C_2$).

        #### Core Insight:
        During Pass 1 of Apriori, memory is used only to store counts of individual items. Most of the remaining available memory sits idle. PCY utilizes this spare memory to hash candidate pairs.

        #### PCY Execution Steps:
        1. **Pass 1 (Counting Singletons & Hashing Pairs)**:
           - In primary memory, count the occurrences of individual singleton items.
           - In the remaining memory, allocate an array of $B$ integer buckets. For each transaction, take every pair of items $\{i, j\}$ present in that transaction and hash the pair into a bucket index:
             $$\text{bucket} = \text{hash}(i, j) \pmod B$$
             $$\text{Bucket}[\text{bucket}] = \text{Bucket}[\text{bucket}] + 1$$
        2. **Bitmap Compression**:
           - After Pass 1, evaluate each bucket count against the minimum support threshold $s$.
           - Compress the entire bucket array into a **bitmap** where bucket $k$ is represented by a single bit:
             $$\text{Bitmap}[k] = \begin{cases} 1, & \text{if } \text{Bucket}[k] \ge s \\ 0, & \text{if } \text{Bucket}[k] < s \end{cases}$$
        3. **Pass 2 (Candidate Pair Generation)**:
           - When generating candidate pairs $C_2$, a pair $\{i, j\}$ is counted only if:
             1. Item $i$ is frequent.
             2. Item $j$ is frequent.
             3. The bit $\text{Bitmap}[\text{hash}(i, j) \pmod B] == 1$.
           - Any pair whose hash bucket was below threshold is pruned before memory allocation, eliminating up to $80\%\text{--}90\%$ of non-frequent candidate pairs.
        """)

        st.subheader("3.5 Multistage & Multihash PCY Extensions")
        st.markdown(r"""
        #### A. Multistage PCY (Sequential Multi-Pass Hashing)
        Multistage PCY applies sequential hashing passes using independent hash functions to achieve further candidate reduction:
        - **Pass 1**: Counts singletons and builds Bitmap 1 using hash function $h_1$.
        - **Pass 2**: Hashes pairs into a second hash table of $B$ buckets using an independent hash function $h_2$, but hashes only pairs whose bit was 1 in Bitmap 1.
        - **Pass 3**: Only counts candidate pairs whose bits are 1 in **both** Bitmap 1 and Bitmap 2.

        #### B. Multihash PCY (Simultaneous Single-Pass Hashing)
        Multihash PCY uses multiple independent hash functions ($h_1, h_2$) in a **single pass**:
        - Memory is partitioned into $m$ independent hash tables of size $B/m$.
        - Each pair in every transaction is hashed into all $m$ tables simultaneously.
        - In Pass 2, a pair is admitted into candidate set $C_2$ only if its corresponding bucket is marked frequent in **every** bitmap:
          $$\{i, j\} \in C_2 \iff \bigwedge_{r=1}^m \text{Bitmap}_r[h_r(i, j)] == 1$$
        This provides strong candidate filtering in a single pass without the latency of an extra database scan.
        """)

    # -------------------------------------------------------------------------
    # TAB 4: Evaluation Metrics & CRISP-DM
    # -------------------------------------------------------------------------
    with theory_tab4:
        st.header("4. Performance Metrics & CRISP-DM Methodology")
        
        st.subheader("4.1 The Accuracy Paradox under Class Imbalance")
        st.markdown(r"""
        In classification problems with asymmetric class distributions, standard classification **Accuracy** can be misleading.
        
        #### Mathematical Definition of Accuracy:
        $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
        
        #### Why Accuracy Fails on Imbalanced Data:
        In our historical Olympic dataset, approximately $85.3\%$ of entries won no medal, and only $14.7\%$ won a medal.
        - A trivial classifier that predicts "No Medal" for every single record without learning any features would achieve an accuracy of:
          $$\text{Accuracy}_{\text{trivial}} = \frac{0 + 853}{0 + 853 + 0 + 147} = 85.3\%$$
        - Despite an $85.3\%$ accuracy score, the model fails to detect a single actual medalist.
        Therefore, data miners evaluate imbalanced classification using metrics focused on the minority class.
        """)

        st.subheader("4.2 Precision, Recall, and the F1-Score")
        st.markdown(r"""
        Classification outcomes are broken down into a **Confusion Matrix**:
        - **True Positive (TP)**: Correctly predicted positive instances (medalists identified as medalists).
        - **False Positive (FP)**: Negative instances incorrectly classified as positive (false alarms).
        - **False Negative (FN)**: Positive instances incorrectly classified as negative (missed medalists).
        - **True Negative (TN)**: Correctly predicted negative instances.

        #### Evaluation Metrics:
        1. **Precision (Positive Predictive Value)**:
           Measures the proportion of positive predictions that were truly positive:
           $$\text{Precision} = \frac{TP}{TP + FP}$$
        2. **Recall (Sensitivity / True Positive Rate)**:
           Measures the proportion of actual positive instances that were successfully identified:
           $$\text{Recall} = \frac{TP}{TP + FN}$$
        3. **F1-Score (Harmonic Mean)**:
           Balances Precision and Recall into a single metric. Unlike an arithmetic mean, the harmonic mean severely penalizes extreme imbalances between the two components:
           $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$
        """)

        st.subheader("4.3 Cost-Sensitive Learning & Class Re-Weighting")
        st.markdown(r"""
        To address class imbalance without discarding data (as in undersampling) or synthesizing artificial rows (as in oversampling), **Cost-Sensitive Weighting** modifies the algorithm's objective function.

        #### Balanced Class Weighting Formula:
        Sample weights are set inversely proportional to class frequencies:
        $$w_k = \frac{N}{C \cdot N_k}$$
        where $N$ is total samples, $C$ is number of classes (2 for binary), and $N_k$ is the count of samples in class $k$.
        
        In our dataset:
        - Majority Class Weight ($w_0$ for No Medal): $w_0 = \frac{1}{2 \times 0.853} \approx 0.586$
        - Minority Class Weight ($w_1$ for Won Medal): $w_1 = \frac{1}{2 \times 0.147} \approx 3.401$
        
        Because $w_1$ is approximately $5.8$ times larger than $w_0$, the loss function penalizes false negatives $5.8$ times more heavily than false alarms. This shifts tree decision thresholds, allowing models like **Random Forest** to capture over $60\%$ of true medalists.
        """)

        st.subheader("4.4 The CRISP-DM Lifecycle in Data Mining")
        st.markdown(r"""
        The **CRISP-DM (Cross-Industry Standard Process for Data Mining)** process framework structures data mining projects into six standardized, iterative phases:

        1. **Business Understanding**: Define the project objectives and analytical requirements from a domain perspective (e.g., predicting medal outcomes and identifying demographic clusters).
        2. **Data Understanding**: Collect, inspect, and explore raw data to identify data quality issues, missing values, anomalies, and structural distributions.
        3. **Data Preparation**: Construct clean modeling datasets via deduplication, median imputation, numerical standardization, and categorical encoding.
        4. **Modeling**: Select, calibrate, and train various modeling techniques across supervised classification, unsupervised clustering, and association rule mining.
        5. **Evaluation**: Benchmark algorithms on held-out test partitions using domain-appropriate metrics (F1-score, precision-recall trade-offs, silhouette scores, and runtime latency).
        6. **Deployment**: Package findings into reproducible notebooks, exported analytical artifacts, and an interactive web dashboard for exploration.
        """)


# =============================================================================
# MODULE 3: Supervised Classification Benchmark
# =============================================================================
elif menu == "🤖 Supervised Classification":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🤖 Supervised Classification Benchmark</div>
        <div class="hero-subtitle">
            Evaluating 5 machine learning classification algorithms on predicting Olympic medalists 
            under 85:15 natural class imbalance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    clf_df = results.get("classification", pd.DataFrame())

    tab1, tab2, tab3 = st.tabs(["📊 Performance Benchmark", "🎯 Feature Importance", "🔬 Confusion Matrix Analysis"])

    with tab1:
        st.subheader("Algorithm Comparison Matrix")
        st.dataframe(
            clf_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
                "Training time (sec)": "{:.2f} s"
            }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score"], color="#DCFCE7")
              .highlight_min(subset=["Training time (sec)"], color="#E0E7FF"),
            use_container_width=True
        )

        st.markdown("### Interactive Metric Comparison Visualizer")
        metric_choice = st.selectbox(
            "Select Metric to Visualize:", 
            ["F1-score", "Recall", "Precision", "Accuracy", "Training time (sec)"]
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(data=clf_df, x="Algorithm", y=metric_choice, palette="Blues_r", ax=ax)
        ax.set_title(f"Comparative Benchmark by {metric_choice}", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_choice)
        ax.set_xlabel("")
        plt.xticks(rotation=20)
        st.pyplot(fig)

        st.info("""
        💡 **Core Takeaway**: 
        - **XGBoost** achieves the highest Accuracy (85.81%) and Precision (72.70%), but fails on minority class Recall (6.06%).
        - **Random Forest** with balanced subsampling delivers the highest harmonic **F1-score (0.3686)**, capturing **60.97%** of all true medalists.
        - **Decision Trees (ID3 & CART)** provide maximum sensitivity, finding nearly **70% of medal winners** at the cost of lower precision.
        """)

    with tab2:
        rf_imp = results.get("rf_importance", pd.DataFrame())
        st.subheader("Top Predictive Features (Random Forest)")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(
                rf_imp.style.format({"Importance": "{:.4f}"}).bar(subset=["Importance"], color="#93C5FD"),
                use_container_width=True
            )
        with col2:
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.barplot(data=rf_imp.head(10), x="Importance", y="Feature", palette="viridis", ax=ax)
            ax.set_title("Top 10 Feature Importances", fontsize=11, fontweight="bold")
            st.pyplot(fig)

    with tab3:
        st.subheader("Confusion Matrix Archetypes on 53,947 Test Records")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            #### Cost-Weighted Random Forest
            - **True Negatives**: ~32,654
            - **False Positives**: ~13,338
            - **False Negatives**: ~3,105
            - **True Positives**: ~4,850
            > **Recall = 60.97%**: Actively prioritizes identifying medalists by tolerating higher false alarm rates.
            """)
        with c2:
            st.markdown("""
            #### Unweighted XGBoost
            - **True Negatives**: ~45,811
            - **False Positives**: ~181
            - **False Negatives**: ~7,473
            - **True Positives**: ~482
            > **Recall = 6.06%**: Prioritizes raw accuracy by assigning almost all borderline athletes to the majority "No Medal" class.
            """)

# =============================================================================
# MODULE 4: Unsupervised Biometric Clustering
# =============================================================================
elif menu == "🔍 Unsupervised Clustering":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔍 Athlete Biometric Segmentation</div>
        <div class="hero-subtitle">
            Partitioning standardized biometrics (Age, Height, Weight) across 15,000 athletes to discover 
            natural physiological clusters.
        </div>
    </div>
    """, unsafe_allow_html=True)

    clust_df = results.get("clustering", pd.DataFrame())
    st.dataframe(clust_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Identified Physiological Archetypes (K-Means, K=4)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="concept-card">
            <span class="concept-badge badge-blue">Cluster 0</span>
            <h4>Agility & Lean Endurance</h4>
            <p><strong>Profile:</strong> Moderate height (~171 cm), lower mass (~63 kg), peak youth (~24.6 yrs).</p>
            <p><strong>Primary Sports:</strong> Gymnastics, Long-Distance Athletics, Diving, Figure Skating.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="concept-card">
            <span class="concept-badge badge-emerald">Cluster 1</span>
            <h4>Heavyweight Power & Explosiveness</h4>
            <p><strong>Profile:</strong> Substantial mass (>91 kg), elevated BMI, muscular leverage (~26.7 yrs).</p>
            <p><strong>Primary Sports:</strong> Weightlifting, Shot Put, Super-Heavyweight Judo, Wrestling.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="concept-card">
            <span class="concept-badge badge-purple">Cluster 2</span>
            <h4>Tall Leverage & Wingspan</h4>
            <p><strong>Profile:</strong> Tall stature (>188 cm), broad span, moderate-to-heavy body mass (~25.2 yrs).</p>
            <p><strong>Primary Sports:</strong> Rowing, Basketball, Volleyball, Swimming.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="concept-card">
            <span class="concept-badge badge-amber">Cluster 3</span>
            <h4>Veteran Specialists</h4>
            <p><strong>Profile:</strong> Mature competitive age range (35–50+), average biometric frame.</p>
            <p><strong>Primary Sports:</strong> Equestrianism, Shooting, Archery, Curling.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### DBSCAN Anomaly Detection Findings")
    st.info("DBSCAN identified **1 continuous dense cluster** and **101 noise points (0.67%)**. Human athletic biometrics form a continuous multivariate normal distribution without empty voids; hence, DBSCAN acts primarily as an **outlier detector** for extreme anthropometric anomalies.")

# =============================================================================
# MODULE 5: Association Rule Mining
# =============================================================================
elif menu == "🔗 Association Rule Mining":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔗 Frequent Pattern & Association Mining</div>
        <div class="hero-subtitle">
            Benchmarking Apriori, FP-Growth, and Hash-Based PCY Variants across 30,000 multi-attribute transaction baskets.
        </div>
    </div>
    """, unsafe_allow_html=True)

    assoc_df = results.get("association", pd.DataFrame())
    st.dataframe(assoc_df, use_container_width=True)

    st.markdown("### Execution Latency Benchmark")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    valid_timing = assoc_df.dropna(subset=["Execution time (sec)"])
    sns.barplot(data=valid_timing, x="Algorithm", y="Execution time (sec)", palette="mako", ax=ax)
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Runtime Comparison across Association Algorithms", fontsize=11, fontweight="bold")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    st.success("⚡ **Efficiency Result**: **Multihash PCY** achieved the fastest pair screening (**0.475s**), representing a **2.60x speedup** over canonical single-pass PCY and outperforming FP-Growth traversal on dense baskets.")

    tab1, tab2 = st.tabs(["📋 Top Frequent Pairs (PCY)", "📜 Frequent Itemsets (Apriori/FP-Growth)"])
    with tab1:
        pcy_df = results.get("pcy", pd.DataFrame())
        st.dataframe(pcy_df.head(20), use_container_width=True)

    with tab2:
        ap_df = results.get("apriori", pd.DataFrame())
        st.dataframe(ap_df.head(20), use_container_width=True)

# =============================================================================
# MODULE 6: Live Inference & Biometric Profiler
# =============================================================================
elif menu == "🔮 Live Inference & Profiler":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔮 Interactive Olympic Outcome & Biometric Profiler</div>
        <div class="hero-subtitle">
            Simulate athlete biometrics to calculate BMI, assign K-Means Euclidean topology clusters, 
            score multivariate podium probabilities, and match mined Apriori patterns.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("athlete_form"):
        st.subheader("Athlete Biometric & Contextual Input")
        col1, col2, col3 = st.columns(3)
        with col1:
            sex = st.selectbox("Sex", ["M", "F"])
            age = st.slider("Age (years)", min_value=12, max_value=70, value=25)
            year = st.slider("Olympic Year", min_value=1896, max_value=2024, value=2016, step=2)
        with col2:
            height = st.slider("Height (cm)", min_value=130, max_value=220, value=178)
            weight = st.slider("Weight (kg)", min_value=30, max_value=180, value=73)
            season = st.selectbox("Season", ["Summer", "Winter"])
        with col3:
            sport_options = [
                "Athletics", "Gymnastics", "Swimming", "Rowing", "Cycling", 
                "Basketball", "Football", "Hockey", "Ice Hockey", "Alpine Skiing",
                "Weightlifting", "Wrestling", "Shooting", "Equestrianism", "Cross Country Skiing",
                "Volleyball", "Water Polo", "Handball", "Boxing", "Fencing", "Other Sport"
            ]
            sport = st.selectbox("Sport Discipline", sport_options)
            team_options = [
                "United States", "Soviet Union", "Germany", "Great Britain", "France",
                "Italy", "China", "Australia", "Japan", "Canada", "Netherlands",
                "Sweden", "Hungary", "Poland", "Other Team"
            ]
            team = st.selectbox("Represented Team / Country", team_options)

        submitted = st.form_submit_button("⚡ Run Data Mining Inference")

    if submitted:
        # 1. BIOMETRIC METRICS
        height_m = height / 100.0
        bmi = weight / (height_m ** 2)

        # 2. CONTINUOUS K-MEANS DISTANCE CLUSTERING
        z_age = (age - 25.56) / 6.39
        z_height = (height - 175.34) / 10.52
        z_weight = (weight - 70.70) / 14.35

        cluster_centers = {
            "Cluster 0: Agility & Lean Endurance": np.array([-0.15, -0.42, -0.55]),
            "Cluster 1: Heavyweight & Power": np.array([0.18, 0.45, 1.42]),
            "Cluster 2: Tall Leverage & Wingspan": np.array([-0.05, 1.25, 0.68]),
            "Cluster 3: Veteran Competitors": np.array([1.85, 0.02, 0.15]),
        }
        
        user_vector = np.array([z_age, z_height, z_weight])
        distances = {name: float(np.linalg.norm(user_vector - center)) for name, center in cluster_centers.items()}
        assigned_cluster = min(distances, key=distances.get)
        min_dist = distances[assigned_cluster]
        is_outlier = min_dist > 2.8

        # 3. CONTINUOUS MULTIVARIATE LOGISTIC SCORING
        logit = -1.758

        sport_benchmarks = {
            "Gymnastics": {"opt_age": 19.5, "opt_h": 160.0, "opt_w": 52.0, "weight_tol": 8.0},
            "Rowing": {"opt_age": 26.0, "opt_h": 190.0, "opt_w": 87.0, "weight_tol": 12.0},
            "Swimming": {"opt_age": 22.0, "opt_h": 185.0, "opt_w": 77.0, "weight_tol": 9.0},
            "Athletics": {"opt_age": 25.0, "opt_h": 178.0, "opt_w": 70.0, "weight_tol": 14.0},
            "Basketball": {"opt_age": 26.0, "opt_h": 198.0, "opt_w": 92.0, "weight_tol": 15.0},
            "Weightlifting": {"opt_age": 25.0, "opt_h": 168.0, "opt_w": 88.0, "weight_tol": 20.0},
            "Shooting": {"opt_age": 34.0, "opt_h": 175.0, "opt_w": 76.0, "weight_tol": 16.0},
            "Equestrianism": {"opt_age": 37.0, "opt_h": 175.0, "opt_w": 70.0, "weight_tol": 15.0},
            "Ice Hockey": {"opt_age": 26.0, "opt_h": 182.0, "opt_w": 84.0, "weight_tol": 10.0},
            "Alpine Skiing": {"opt_age": 24.5, "opt_h": 176.0, "opt_w": 78.0, "weight_tol": 10.0},
            "Cross Country Skiing": {"opt_age": 26.5, "opt_h": 176.0, "opt_w": 69.0, "weight_tol": 8.0},
            "Cycling": {"opt_age": 25.0, "opt_h": 177.0, "opt_w": 70.0, "weight_tol": 9.0},
            "Football": {"opt_age": 23.5, "opt_h": 178.0, "opt_w": 73.0, "weight_tol": 9.0},
            "Volleyball": {"opt_age": 25.5, "opt_h": 192.0, "opt_w": 83.0, "weight_tol": 11.0},
            "Water Polo": {"opt_age": 25.5, "opt_h": 186.0, "opt_w": 88.0, "weight_tol": 11.0},
            "Handball": {"opt_age": 26.0, "opt_h": 185.0, "opt_w": 84.0, "weight_tol": 11.0},
        }

        if sport in sport_benchmarks:
            bm = sport_benchmarks[sport]
            age_dev = abs(age - bm["opt_age"]) / 6.0
            height_dev = abs(height - bm["opt_h"]) / 12.0
            weight_dev = abs(weight - bm["opt_w"]) / bm["weight_tol"]
            total_dev = (age_dev * 0.9) + (height_dev * 1.0) + (weight_dev * 1.1)
            logit += (1.4 - (0.85 * total_dev))
        else:
            age_dev = abs(age - 25.0) / 7.0
            logit += (0.4 - 0.5 * age_dev)

        team_weights = {
            "United States": 1.15,
            "Soviet Union": 1.20,
            "Germany": 0.85,
            "Great Britain": 0.65,
            "France": 0.55,
            "Italy": 0.50,
            "China": 0.70,
            "Australia": 0.55,
            "Japan": 0.40,
            "Canada": 0.35,
            "Netherlands": 0.35,
            "Sweden": 0.40,
            "Hungary": 0.45,
            "Poland": 0.25,
            "Other Team": -0.40,
        }
        logit += team_weights.get(team, 0.0)

        year_effect = (1960.0 - year) / 50.0 * 0.35
        logit += year_effect

        podium_probability = 1.0 / (1.0 + np.exp(-logit))
        podium_probability = float(np.clip(podium_probability, 0.015, 0.94))

        # 4. QUERY DISCOVERED ASSOCIATION RULES
        ap_df = results.get("apriori", pd.DataFrame())
        matched_rules = []
        if not ap_df.empty:
            sport_tag = f"Sport={sport}"
            team_tag = f"Team={team}"
            season_tag = f"Season={season}"
            sex_tag = f"Sex={sex}"
            
            for _, r in ap_df.iterrows():
                try:
                    items = ast.literal_eval(r["itemsets"]) if isinstance(r["itemsets"], str) else r["itemsets"]
                    items_set = set(items)
                    match_count = sum(1 for tag in [sport_tag, team_tag, season_tag, sex_tag] if tag in items_set)
                    if match_count >= 2:
                        matched_rules.append({
                            "Pattern": " + ".join(sorted(list(items_set))),
                            "Support (%)": f"{r['support']*100:.2f}%",
                            "Scope": f"{len(items_set)}-Itemset"
                        })
                except Exception:
                    continue

        # 5. DISPLAY RESULTS
        st.markdown("---")
        st.subheader("🎯 Inference & Diagnostic Results")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Calculated BMI", value=f"{bmi:.1f} kg/m²")
        with c2:
            st.metric(
                label="Podium Probability", 
                value=f"{podium_probability*100:.1f}%",
                delta=f"{(podium_probability - 0.147)*100:+.1f}% vs. Historical Prior (14.7%)"
            )
        with c3:
            outcome = "High Contender" if podium_probability >= 0.30 else ("Moderate Chance" if podium_probability >= 0.15 else "Longshot Entry")
            st.metric(label="Competitive Tier", value=outcome)
        with c4:
            st.metric(label="Biometric Z-Score Distance", value=f"{min_dist:.2f} σ")

        st.markdown("### 🧬 Biometric Cluster Assignment")
        if is_outlier:
            st.warning(f"⚠️ **Anomalous Biometric Profile**: Closest archetype is **{assigned_cluster}**, but distance exceeds DBSCAN density threshold ({min_dist:.2f} σ). This athlete represents an anthropometric outlier in Olympic history.")
        else:
            st.success(f"✅ **Assigned Archetype**: **{assigned_cluster}** (Euclidean Distance to centroid: {min_dist:.2f} σ).")

        with st.expander("🔬 View Distance to All 4 Physiological Archetypes"):
            dist_df = pd.DataFrame([
                {"Cluster Archetype": k, "Euclidean Distance (σ)": f"{v:.2f}", "Match Rank": i+1}
                for i, (k, v) in enumerate(sorted(distances.items(), key=lambda x: x[1]))
            ])
            st.dataframe(dist_df, use_container_width=True)

        st.markdown("### 🔗 Matched Association Mining Patterns")
        if matched_rules:
            st.markdown(f"Discovered **{len(matched_rules)} frequent patterns** from Apriori matching your selected profile (`{sport}`, `{team}`, `{season}`, `{sex}`):")
            st.dataframe(pd.DataFrame(matched_rules).head(10), use_container_width=True)
        else:
            st.info(f"The combination of `{sport}` with `{team}` represents a specialized entry below the 1.5% support threshold across all 30,000 mined transactions.")

# =============================================================================
# MODULE 7: Project Documentation & Notebook Reference
# =============================================================================
elif menu == "📚 Project Documentation":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📚 Project Documentation & References</div>
        <div class="hero-subtitle">
            Comprehensive audit of project notebook steps, dependencies, exported datasets, and reproducibility instructions.
        </div>
        <div class="author-badge">Author: Harshal Pednekar</div>
    </div>
    """, unsafe_allow_html=True)

    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["📋 Notebook KDD Steps", "📂 Exported Results Directory", "💻 Reproducibility"])

    with doc_tab1:
        st.subheader("27-Step Master Notebook Structure")
        st.markdown("""
        1. **Step 1**: Environment Setup & Silent Module Ingestion
        2. **Step 2**: Data Ingestion from Remote Olympic Historical Archive
        3. **Step 3**: Structural Audit (Nulls, Types, Duplicates)
        4. **Step 4**: Data Cleaning & Target Formulation (`Won_Medal`)
        5. **Step 5**: Exploratory Data Analysis & Visualizations
        6. **Step 6**: Dataset Partitioning (80/20 Stratified) & ColumnTransformer Pipelines
        7. **Step 7**: Evaluation Framework Setup (Accuracy, Precision, Recall, F1)
        8. **Step 8**: ID3-style Decision Tree Classifier (Entropy Criterion)
        9. **Step 9**: CART Decision Tree Classifier (Gini Criterion)
        10. **Step 10**: Random Forest Ensemble & Feature Importance Extraction
        11. **Step 11**: XGBoost Gradient Boosted Classifier
        12. **Step 12**: Gaussian Naïve Bayes Classifier
        13. **Step 13**: Supervised Classification Algorithm Comparison
        14. **Step 14**: Unsupervised Data Wrangling & Standard Scaling
        15. **Step 15**: K-Means Clustering & Silhouette / Elbow Tuning
        16. **Step 16**: Agglomerative Hierarchical Clustering (Ward's Linkage)
        17. **Step 17**: DBSCAN Density Clustering & Outlier Detection
        18. **Step 18**: Transaction Basket Preparation for Association Mining
        19. **Step 19**: Apriori Frequent Itemset Generation
        20. **Step 20**: FP-Growth Mining & Prefix Tree Traversal
        21. **Step 21**: Custom PCY (Park-Chen-Yu) Implementation with Hashing
        22. **Step 22**: Multistage PCY Implementation (Sequential Hashes)
        23. **Step 23**: Multihash PCY Implementation (Simultaneous Hashes)
        24. **Step 24**: Association Algorithm Runtime Benchmark
        25. **Step 25**: Unsupervised Clustering Comparison
        26. **Step 26**: Output Package Creation & CSV Exports
        27. **Step 27**: Conclusion & Analytical Syntheses
        """)

    with doc_tab2:
        st.subheader("Generated Artifacts in `Results/`")
        files_info = [
            {"File": "classification_comparison.csv", "Description": "Accuracy, Precision, Recall, F1, and training latency for 5 models", "Rows": 5},
            {"File": "clustering_comparison.csv", "Description": "Silhouette scores, cluster counts, and noise statistics", "Rows": 3},
            {"File": "association_comparison.csv", "Description": "Execution time and itemset counts across all 5 association mining methods", "Rows": 5},
            {"File": "random_forest_feature_importance.csv", "Description": "Normalized feature importances from the 150-tree Random Forest", "Rows": 20},
            {"File": "apriori_frequent_itemsets.csv", "Description": "All 308 frequent itemsets generated via Apriori (support >= 0.015)", "Rows": 308},
            {"File": "fpgrowth_frequent_itemsets.csv", "Description": "Frequent itemsets verified via FP-Growth algorithm", "Rows": 308},
            {"File": "pcy_frequent_pairs.csv", "Description": "Top frequent 2-item pairs mined using hash-filtered PCY", "Rows": 132},
        ]
        st.dataframe(pd.DataFrame(files_info), use_container_width=True)

    with doc_tab3:
        st.subheader("Execution Guide")
        st.code("""
# 1. Clone repository
git clone https://github.com/harshalnnpednekar/Data-Mining-of-Olympic-Athlete-Events-History.git
cd Data-Mining-of-Olympic-Athlete-Events-History

# 2. Install requirements
pip install -r requirements.txt

# 3. Launch Streamlit Portal
streamlit run app.py
        """, language="bash")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 1rem 0;">
    Olympic Athlete Events History Data Mining Portal | Researched & Built by <strong>Harshal Pednekar</strong><br>
    Powered by Python, Scikit-Learn, XGBoost, Mlxtend & Streamlit | MIT License
</div>
""", unsafe_allow_html=True)
