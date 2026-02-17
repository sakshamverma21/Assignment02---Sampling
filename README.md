# Sampling Assignment - Credit Card Fraud Detection

## Objective
Understand how different sampling techniques affect machine learning model performance on imbalanced datasets.

## Problem Statement
Credit card fraud dataset with 85:1 imbalance ratio (763 non-fraud vs 9 fraud cases). Task: Balance dataset using 5 sampling techniques and evaluate performance across 5 ML models.

## Dataset
- **Source**: Credit card transaction data
- **Total Records**: 772
- **Features**: 30 numerical features
- **Target Variable**: Class (0 = Non-fraud, 1 = Fraud)
- **Imbalance Ratio**: 84.78:1 (763 vs 9)

## Sampling Techniques Used

1. **Sampling1: Random Over-Sampling** - Duplicates minority class samples (1,068 training samples)
2. **Sampling2: Random Under-Sampling** - Reduces majority class samples (12 training samples)
3. **Sampling3: SMOTE** - Creates synthetic minority samples through interpolation (1,068 training samples)
4. **Sampling4: ADASYN** - Adaptive synthetic sampling focusing on difficult cases (1,068 training samples)
5. **Sampling5: SMOTETomek** - SMOTE + Tomek Links for cleaning boundaries (1,036 training samples)

## Machine Learning Models

- **M1**: Logistic Regression
- **M2**: Decision Tree Classifier
- **M3**: Random Forest Classifier
- **M4**: Support Vector Machine (SVM)
- **M5**: K-Nearest Neighbors (KNN)

## Methodology

To prevent data leakage:
1. Split data into train (70%) and test (30%) sets
2. Apply sampling ONLY to training set
3. Evaluate on original imbalanced test set

This ensures realistic performance metrics.

## Results Summary

| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|-------|-----------|-----------|-----------|-----------|-----------|
| M1    | 91.81%    | 57.76%    | 93.53%    | 92.67%    | 92.67%    |
| M2    | 96.98%    | 38.79%    | 98.71%    | 98.28%    | 97.84%    |
| M3    | 99.14%    | 66.81%    | 98.71%    | 99.14%    | 98.71%    |
| M4    | 91.81%    | 60.78%    | 92.67%    | 92.67%    | 91.81%    |
| M5    | 97.84%    | 75.00%    | 72.41%    | 72.84%    | 73.71%    |

## Analysis and Discussion

### Best Combinations

**Top Performers (99.14% accuracy):**
- Random Forest + Random Over-Sampling
- Random Forest + ADASYN

**Best Sampling per Model:**
- M1 (Logistic Regression): SMOTE (93.53%)
- M2 (Decision Tree): SMOTE (98.71%)
- M3 (Random Forest): Over-Sampling/ADASYN (99.14%)
- M4 (SVM): SMOTE/ADASYN/SMOTETomek (92.67%)
- M5 (KNN): Over-Sampling (97.84%)

### Key Insights

1. **Random Forest** emerged as the most robust model (67-99% range across all techniques)

2. **SMOTE-based techniques** (SMOTE, ADASYN, SMOTETomek) provided consistent high performance (92-99%) across most models

3. **Random Under-Sampling** failed on this small dataset (39-75% accuracy) due to only 12 training samples

4. **KNN anomaly**: Performed excellently with Over-Sampling (97.84%) but poorly with synthetic samples (72-73%), suggesting distance-based algorithms struggle with interpolated features

5. **Methodology matters**: Splitting before sampling prevents data leakage and ensures realistic metrics

## Conclusion

Sampling techniques significantly impact model performance on imbalanced datasets. SMOTE-based techniques (SMOTE, ADASYN, SMOTETomek) provided the most consistent performance (92-99%), with Random Forest being the most robust model. Under-sampling failed on this small dataset. Proper methodology (split first, then sample only training data) is crucial to prevent data leakage and achieve realistic metrics.

**Optimal solution for this dataset**: Random Forest + ADASYN/SMOTE (99.14% accuracy)

## How to Run

```bash
pip install -r requirements.txt
python sampling_assignment.py
```

**Dependencies**: pandas, numpy, scikit-learn, imbalanced-learn

---

