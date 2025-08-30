# Credit Card Fraud Detection Predictive Models

This project explores and compares several predictive models for detecting fraudulent credit card transactions. The dataset obtained from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), contains credit card transactions from European cardholders in September 2013. A key challenge is its high imbalance, with only **492 frauds out of 284,807 transactions**.

---

## Dataset

The dataset includes numerical features resulting from a PCA transformation due to confidentiality. The only features not transformed are `Time` and `Amount`.

| Feature | Description |
|---------|-------------|
| V1 to V28 | Principal components from PCA |
| Time | Seconds elapsed between each transaction and the first transaction in the dataset |
| Amount | Transaction amount |
| Class | Response variable: `1` indicates a fraudulent transaction, `0` indicates a legitimate transaction |

Fraudulent transactions make up only **0.172%** of all transactions, making the dataset highly unbalanced.

---

## Data Exploration

The project includes a detailed exploration phase to understand the dataset's characteristics:

- **Missing Data:** No missing data was found.  
- **Data Imbalance:** Severe imbalance between fraudulent and non-fraudulent transactions was visualized.  
- **Time and Amount Analysis:** Fraudulent transactions have a more even distribution over time. Legitimate transactions show fewer transactions during certain hours, likely nighttime in Europe.  
- **Feature Correlation:** PCA features (V1-V28) show little correlation with each other. Some features show correlation with `Time` (inverse correlation with V3) and `Amount` (direct correlation with V7 and V20, inverse with V1 and V5).  
- **Feature Density Plots:** Features like V4 and V11 have clearly separated distributions for fraudulent vs. legitimate transactions, making them highly predictive.

---

## Predictive Models

The project evaluated five machine learning models using **ROC-AUC** as the primary metric, suitable for imbalanced datasets:

| Model | Validation AUC Score | Test AUC Score |
|-------|-------------------|---------------|
| RandomForestClassifier | 0.85 | - |
| AdaBoostClassifier | 0.83 | - |
| CatBoostClassifier | 0.86 | - |
| XGBoost | 0.984 (best) | 0.974 |
| LightGBM | 0.974 (best) | 0.946 |

### Key Findings

- RandomForestClassifier and AdaBoostClassifier provided good but lower performance.  
- CatBoostClassifier showed a slight improvement over the previous two.  
- XGBoost and LightGBM, both gradient boosting algorithms, demonstrated the highest performance.  
- **XGBoost** achieved the best test score of **0.974**.  
- Models consistently identified `V14`, `V10`, `V12`, and `V17` as the most important features.

---

## Conclusion

Gradient boosting models like **XGBoost** and **LightGBM** are highly effective at detecting credit card fraud, outperforming simpler ensemble methods. The high ROC-AUC scores indicate strong ability to handle data imbalance and accurately distinguish between fraudulent and legitimate transactions.
