# Customer Churn Analysis & Prediction

## Project Overview

Customer churn is a major challenge for subscription-based businesses. This project focuses on analyzing customer behavior and predicting churn using historical customer data. The goal is to identify key factors influencing churn and propose data-driven retention strategies.

The dataset used is the IBM Telco Customer Churn dataset, containing customer demographics, service usage, account information, and churn status.

## Objectives

Understand customer churn patterns through Exploratory Data Analysis (EDA)

Identify key drivers influencing customer churn

Build machine learning models to predict churn

Provide actionable business recommendations to reduce customer attrition


##  Dataset Description

Rows: 7,043 customers

Target Variable: Churn (Yes / No)

Features include:

Demographics (gender, senior citizen, partner, dependents)

Services (internet, phone, streaming, tech support, security)

Account details (tenure, contract type, payment method)

Billing information (monthly charges, total charges)


## Tools & Technologies

Programming Language: Python

Libraries:

Pandas, NumPy

Matplotlib

Scikit-learn

Environment: Jupyter Notebook & Python scripts

Version Control: Git & GitHub


## Project Workflow
#### Data Cleaning & Preprocessing

Handled missing values in TotalCharges

Converted categorical variables into numerical format using one-hot encoding

Prepared two datasets:

Raw cleaned data for EDA

Encoded data for machine learning models

#### Exploratory Data Analysis (EDA)

Key analyses included:

Overall churn distribution

Churn vs contract type

Churn vs monthly charges

Churn vs tenure

Churn vs internet service & tech support

Churn vs payment method

#### Key EDA Insights:

Month-to-month customers churn significantly more

Customers with high monthly charges are more likely to churn

New customers (low tenure) show higher churn risk

Customers without tech support churn more frequently

#### Model Building (Stage 4)

Two models were developed and evaluated:

🔹 Logistic Regression (Baseline Model)

Accuracy: ~81%

Recall (Churn): ~55%

High interpretability, suitable for business decisions

🔹 Random Forest Classifier

Accuracy: ~79%

Captures non-linear relationships

Lower recall for churn without extensive tuning

## Model Selection Insight:
Logistic Regression was preferred due to higher recall for churn customers, which is more critical in retention-focused use cases.

📈 Model Evaluation Metrics

Accuracy

Precision

Recall (emphasized for churn = 1)

Confusion Matrix

Feature Importance (Random Forest)

## Business Insights & Recommendations (Stage 5)
### High-Risk Customers

Month-to-month contract users

High monthly charges

Low tenure customers

Fiber optic users without tech support

Customers using electronic check payments

#### Recommended Retention Strategies

Encourage long-term contracts with discounts

Implement a strong onboarding program for new customers

Offer bundled or discounted tech support

Optimize pricing for high-bill customers

Promote auto-payment methods

### Project Structure
Churn-Rate-Analysis/
│
├── EDA.py                 # Exploratory Data Analysis
├── Modeling.py            # Model training & evaluation
├── df_encoded.csv         # Cleaned & encoded dataset
├── Telco-Customer-Churn.csv
├── README.md

## Key Learnings

Churn prediction requires focusing on recall, not just accuracy

Business understanding is as important as modeling

Simple, interpretable models can outperform complex ones in real-world scenarios

## Future Improvements

Hyperparameter tuning for Random Forest

Threshold optimization for churn prediction

Use advanced models like XGBoost or LightGBM

Deploy results via a Streamlit dashboard

## Conclusion

This project demonstrates an end-to-end data analytics workflow, from raw data to business insights. By combining EDA, machine learning, and strategic recommendations, it provides a practical framework for understanding and reducing customer churn.