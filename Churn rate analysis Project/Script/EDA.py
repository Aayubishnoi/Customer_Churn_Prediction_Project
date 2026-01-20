import pandas as pd
import matplotlib.pyplot as plt

print("EDA on encoded data started")

# Load encoded data
df_encoded = pd.read_csv(
    r"C:\Users\Asus\OneDrive\Documents\DATA ANALYST\Churn rate analysis Project\Clean_Telco-Customer-Churn.csv"
)

# =============================
# 1. Overall Churn Distribution
# =============================
df_encoded['Churn'].value_counts().plot(kind='bar')
plt.title("Overall Customer Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Customer Count")
plt.show()

# =============================
# 2. Churn Rate for Top Features
# =============================
important_features = [
    'Contract_One year',
    'Contract_Two year',
    'InternetService_Fiber optic',
    'TechSupport_Yes',
    'PaperlessBilling_Yes',
    'SeniorCitizen'
]

for feature in important_features:
    churn_rate = df_encoded.groupby(feature)['Churn'].mean()
    churn_rate.plot(kind='bar')
    plt.title(f"Churn Rate by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Churn Rate")
    plt.show()

# =============================
# 3. Correlation with Churn
# =============================
corr = df_encoded.corr()['Churn'].sort_values()

corr.tail(10).plot(kind='barh')
plt.title("Top Positive Correlations with Churn")
plt.show()

corr.head(10).plot(kind='barh')
plt.title("Top Negative Correlations with Churn")
plt.show()

print("Encoded EDA completed")
