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

## Results
- **14 total actionable segments** identified across the three product categories
- Segments show a **3.6x spread in relief rates** (9.8% – 35%) — meaning otherwise-similar complaints receive very different outcomes depending on segment
- **21,000+ complaints** flagged as belonging to consistently under-served segments (low relief-rate clusters), highlighting where consumer outcomes may warrant closer review

## Notes & Limitations
- Clustering was performed independently per product category; cross-product segment comparisons should be interpreted with that in mind
- With only 14 segments across 3 product groups, some individual clusters are relatively small — segment-level relief-rate estimates should be treated as directional, not statistically definitive, without further validation
- The specific embedding model used to generate narrative vectors is not documented in this notebook; embeddings were loaded pre-computed rather than generated inline

## Tech Stack
Python · pandas · scikit-learn (KMeans, TF-IDF) · UMAP · HDBSCAN
