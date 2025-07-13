# 📊 Customer Churn Prediction

Predict customer churn for a telecommunications company using supervised machine learning models. This project identifies customers likely to cancel their subscriptions, helping businesses take proactive steps to retain them.

---

## 📌 Project Overview

This project focuses on predicting customer churn using classification algorithms. By analyzing customer behavior, demographics, and service usage, we aim to identify those most at risk of leaving.

---

## 🎯 Business Value

- 🔍 **Identify At-Risk Customers**: Detect potential churners early.
- 💼 **Enhance Retention Strategies**: Leverage data to improve customer loyalty.
- 💰 **Increase Revenue**: Focus on retaining high-value customers.
- 🎯 **Optimize Marketing Spend**: Target customers more effectively.

---

## 📂 Dataset

- **Source**: [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Features**:
  - Demographics: Gender, Senior Citizen, Partner, Dependents
  - Services: Phone, Internet, Streaming, etc.
  - Account Info: Tenure, Contract, Payment Method
  - Churn: Target variable (`Yes` / `No`)
- **Size**: 7,043 customers | 21 features

---

## 🛠️ Models Implemented

| Model                    | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------------|----------|-----------|--------|----------|---------|
| ✅ Random Forest (Tuned) | **80.12%** | 72.35%    | 79.21% | 75.62%   | **85.04%** |
| Gradient Boosting       | 79.35%   | 71.82%    | 78.45% | 74.98%   | 84.12% |
| Random Forest (Default) | 78.94%   | 71.25%    | 77.89% | 74.42%   | 83.76% |
| Logistic Regression     | 77.42%   | 69.87%    | 75.32% | 72.48%   | 81.95% |
| Decision Tree           | 74.82%   | 67.45%    | 72.18% | 69.71%   | 78.23% |
| KNN                     | 74.41%   | 66.92%    | 71.54% | 69.12%   | 77.65% |
| SVM                     | 73.68%   | 66.34%    | 70.89% | 68.52%   | 76.87% |

---

## 🏆 Best Model: Tuned Random Forest

- ✅ Highest Accuracy and ROC-AUC
- ✅ Balanced Precision & Recall
- ✅ Handles imbalanced data well
- ✅ Provides feature importance
- ✅ Robust to overfitting

---

## 📊 Key Insights

**Top Factors Influencing Churn**:
- 📉 Contract Type: Month-to-month customers are more likely to churn.
- 🌐 Internet Service: Fiber optic users churn more than DSL users.
- ⏳ Tenure: New customers show a higher churn tendency.
- 💸 Monthly Charges: Higher charges correlate with churn.
- 🛡️ Lack of Tech Support & Online Security contributes to churn.

**Business Recommendations**:
- 🎁 Offer loyalty benefits to month-to-month users.
- ⚙️ Improve fiber optic service quality.
- 📦 Bundle tech support and security features.
- 💳 Offer discounts for long-term contracts.
- 👥 Monitor high-spending customers closely.

---

