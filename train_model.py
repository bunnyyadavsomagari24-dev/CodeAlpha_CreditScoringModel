import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/credit_data.csv")

# Features and Target
X = df.drop("CreditScore", axis=1)
y = df["CreditScore"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("=" * 35)
print("MODEL PERFORMANCE")
print("=" * 35)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall   : {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "models/credit_model.pkl")

print("\n✅ Model saved successfully!")

# Show confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()