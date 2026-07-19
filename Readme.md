# Consumer Complaint Segmentation (NLP + Clustering)

## Overview
This project segments consumer financial complaints filed with the **Consumer Financial Protection Bureau (CFPB)** into actionable groups, combining narrative text embeddings, sentiment, and complaint metadata. The goal: move beyond generic complaint categories and surface **data-driven segments** with meaningfully different outcomes — specifically, relief rates — so under-served complaint groups can be identified and prioritized.

## Data
- **49,850** consumer complaints from the CFPB public complaints database
- Spanning three product categories: **Debt Collection**, **Checking/Savings**, and **Credit Card**
- Each complaint includes free-text narrative, product/sub-product metadata, and outcome (whether relief was granted)

## Methodology

The pipeline combines three complementary signal types before clustering, and runs clustering **separately within each product category** rather than on the combined dataset — since complaint patterns differ meaningfully by product.

1. **Feature construction**
   - **Text embeddings** — dense vector representation of each complaint narrative
   - **Sentiment score** — captures emotional tone of the complaint text
   - **Metadata features** — categorical/numeric complaint attributes
   - All three combined into a single feature matrix per complaint

2. **Dimensionality reduction**
   - **UMAP** (cosine metric, reduced to 10 components) applied to the combined feature matrix, making clustering tractable and less noisy

3. **Clustering — per product category**
   - **KMeans** as the primary method, with the number of clusters (*k*) chosen per product category via **silhouette score** (final k = 7, 3, and 4 across the three product groups)
   - **HDBSCAN** explored as an alternative density-based approach for the Credit Card segment

4. **Cluster interpretation**
   - **TF-IDF** used to extract top distinguishing keywords per cluster, making each segment interpretable (not just a numbered group) for labeling and downstream action

5. **Statistical robustness checks**
   - Clusters below a minimum size threshold (n < 500) are **excluded from headline comparisons**, since relief-rate estimates on very small clusters are too noisy to trust — this avoids overstating the spread using an outlier small cluster
   - **Bootstrap confidence intervals** (2,000 resamples) computed per segment to validate that relief-rate differences between segments are stable, not sampling noise

## Results
- **14 total actionable segments** identified across the three product categories
- Among robustly-sized segments (n ≥ 500), relief rates show a **4.00x spread** — from **33.0%** (Debt_1, n=1,188) down to **8.3%** (Debt_6, n=654)
- **21,000+ complaints** flagged as belonging to consistently under-served, low-relief segments, highlighting where consumer outcomes may warrant closer review
- Segment-level analysis surfaced a further nuance: the *same* company can show very different relief rates across segments (e.g., one collections agency showed 81.2% relief in one segment vs. 31.6% in another) — suggesting complaint framing/context matters beyond company identity alone

## Notes & Limitations
- Clustering was performed independently per product category; cross-product segment comparisons should be interpreted with that in mind
- Very small clusters (n < 500) are deliberately excluded from the headline spread statistic; their relief rates are reported but flagged as unreliable given limited sample size
- The specific embedding model used to generate narrative vectors is not documented in this notebook; embeddings were loaded pre-computed rather than generated inline

## Tech Stack
Python · pandas · scikit-learn (KMeans, TF-IDF) · UMAP · HDBSCAN
