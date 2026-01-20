# =============================
# Customer Churn Modeling
# =============================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Modeling started...")

# =============================
# Load Encoded Data
# =============================
df = pd.read_csv(
    r"C:\Users\Asus\OneDrive\Documents\DATA ANALYST\Churn rate analysis Project\Clean_Telco-Customer-Churn.csv"
)

print("Data shape:", df.shape)

# =============================
# Features & Target
# =============================
X = df.drop('Churn', axis=1)
y = df['Churn']

# =============================
# Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =============================
# Model 1: Logistic Regression
# =============================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# =============================
# Model 2: Random Forest
# =============================
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# =============================
# Feature Importance
# =============================
importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importances.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()

print("Modeling completed successfully")
