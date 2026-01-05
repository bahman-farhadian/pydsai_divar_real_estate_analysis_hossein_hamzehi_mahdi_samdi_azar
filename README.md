# Divar Real Estate Market Analysis

**Course**: Data Science and AI Introductory Course  
**Institution**: School of Data Processing and Analysis Daghigheh  
**Team**: Bahman Farhadian, Mahdi Samadi Azar  
**Date**: January 2026

---

## Project Overview

This project analyzes Iranian real estate market data from Divar platform to extract actionable insights for market stakeholders, specifically buyers and sellers. The analysis spans data quality assessment, exploratory analysis, market segmentation, price modeling, and text classification.

### Guiding Principle

This project prioritizes practical value over technical complexity. As emphasized in the course:

> "This question's goal is NOT for you to just write code with correct output. Here you need to frame a problem that is valuable outside this data world."

> "The stakeholders listed here don't necessarily know data and statistics. Your presentation style, how you communicate information - all of it matters."

Our approach focuses on simple, interpretable analysis that creates real value for non-technical stakeholders.

---

## Dataset

**Source**: Divar Real Estate Advertisements  
**Size**: ~1.6 million listings  
**Coverage**: Multiple cities across Iran  
**Time Period**: 2024

Key fields include property type, location, price, size, construction year, amenities, and advertisement text.

---

## Phase 1: Data Quality Assessment

**Objective**: Identify data recording errors and determine which fields are reliable for analysis.

> "Where you expect a number around 50-100, do you suddenly see a negative number? A 5-digit number? See if data is reasonable and logical."

This phase examines missing values across all columns, identifies duplicate records, and detects logical inconsistencies such as negative prices, impossible building ages, or floor numbers exceeding total floors. Fields with critical issues are excluded from core analysis or given peripheral roles only.

**Output**: Cleaned dataset with documented exclusions and quality assessment summary.

---

## Phase 2: Exploratory Data Analysis

**Objective**: Understand data distributions and relationships to generate hypotheses for market analysis.

This phase covers distribution analysis for numerical variables, summary statistics, correlation analysis between key variables, and identification of notable patterns. The findings from this phase directly inform the stakeholder-focused market analysis.

**Output**: Statistical summaries, correlation insights, and documented hypotheses.

---

## Phase 3: Market Analysis for Stakeholders

**Objective**: Deliver actionable insights for buyers and sellers in the Iranian real estate market.

> "Don't go for very technical format or very high-level complex models. This question doesn't have that capacity. The audience will reject it if you try to show them that way."

> "You don't need extensive market study. You need MORE THINKING."

### Target Stakeholders

**Buyers**: Individuals seeking to purchase residential property who need to understand fair pricing, valuable locations, and what features affect price.

**Sellers**: Property owners looking to list their property who need guidance on competitive pricing and understanding what adds value to their listing.

### Analysis Focus

The analysis addresses practical questions these stakeholders face: Which areas offer best value per square meter? How do amenities affect listing prices? What distinguishes different market segments? Where are pricing opportunities?

**Output**: Clear visualizations and recommendations in non-technical language.

---

## Phase 4: Clustering Analysis

**Objective**: Segment the real estate market into distinct, interpretable categories.

> "If you're using features with vastly different scales in mean or variance, this greatly affects clustering. You must normalize if needed."

### Methodology

Preprocessing includes normalization of features, log transformation for skewed variables, and outlier handling. Clustering is performed using K-Means with optimal cluster count determined through Elbow and Silhouette methods.

### Dimensionality Reduction

Two approaches are compared: clustering on original features with PCA for visualization, and clustering on principal components with subsequent comparison.

> "Do dimensionality reduction with PCA first, then cluster on principal components, then interpret again. See what differences you find."

> "If you use t-SNE, be aware it has stochastic behavior. Each run gives slightly different results. Only make claims about robust patterns that don't change between runs."

### Interpretation

Each cluster is given a business-meaningful name such as "Budget apartments under 80 sqm", "Luxury new constructions", or "Old properties for renovation". This segmentation provides market understanding beyond simple statistics.

**Output**: Market segments with business interpretations and visual representations.

---

## Phase 5: Price Prediction

**Objective**: Build price prediction models and identify over-valued and under-valued listings.

> "Your output variable is probably PRICE PER SQUARE METER. You could use total price, but price per sqm is more logical."

### Target Variable

Price per square meter is used as the target variable rather than total price, providing more meaningful comparisons across different property sizes.

### Models

Three regression approaches are implemented and compared: Linear Regression as interpretable baseline, tree-based methods (Random Forest) for feature importance analysis, and distance-based or neural network methods for comparison.

Models are evaluated using R-squared, RMSE, and MAE metrics.

### Value Classification

Listings are categorized as over-valued, normal, or under-valued based on prediction residuals:

- **Over-valued**: Priced significantly higher than predicted, representing poor value for buyers
- **Normal**: Priced as expected given features
- **Under-valued**: Priced below predicted value, representing potential opportunities

> "If you find limitations for your model, report them. Where does it work? Where doesn't it?"

**Output**: Trained models, performance comparison, and value classification for all listings with documented limitations.

---

## Phase 6: Text Classification

**Objective**: Extract structured information from advertisement text.

### Part A: Property Type Classification

Using combined title and description text, models are trained to predict property type (cat3_slug). Text preprocessing includes normalization of Persian characters, removal of special characters and noise, and vectorization using both Bag-of-Words (TF-IDF) and embedding approaches.

Three classification models are compared with evaluation through accuracy, F1-score, and confusion matrix analysis.

### Part B: User Type Classification

This task predicts whether an advertisement was posted by an individual or a real estate agent (user_type). The challenge is harder due to high NULL rate and class imbalance.

> "This column has many NULLs, so labeled data is less. The challenge is harder. You need a stronger model."

Class imbalance is addressed through appropriate techniques, and the best model is used to predict user type for records with missing values.

> "After predicting, check manually - does it make sense? Is the model's prediction reasonable?"

**Output**: Classification models for both tasks, filled NULL values for user_type, and manual validation of predictions.

---

## Project Structure

```
.
├── README.md
├── data/
│   ├── processed/
│   └── raw/
│       ├── README.md
│       ├── divar_real_estate_ads.csv
│       └── sampled_data.csv
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_market_analysis.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_price_prediction.ipynb
│   ├── 06_text_classification.ipynb
│   ├── outputs/
│   │   ├── figures/
│   │   └── models/
│   └── src/
└── reports/
    ├── report.md
    └── presentation.pdf
```

---

## Deliverables

| Item | Description |
|------|-------------|
| Presentation | Results and findings for stakeholder audience |
| Written Report | Methodology details and decision rationale |
| Code | Jupyter notebooks with analysis implementation |
| Final Package | All materials in ZIP format |

---

## Technical Environment

- Python 3.x
- pandas, numpy for data manipulation
- matplotlib, seaborn for visualization
- scikit-learn for machine learning
- hazm for Persian text processing (optional)