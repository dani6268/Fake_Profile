import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"D:\project\fake profile\fake_profile_dataset.csv")

# Features and label
X = df.drop(columns=["fake"])
y = df["fake"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Train model
xgb = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
xgb.fit(X_train_sc, y_train)

# Evaluate model
y_pred = xgb.predict(X_test_sc)
print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
plt.figure(figsize=(8,5))
pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).plot(kind="bar")
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()

# ---- User Input Prediction ----
print("\n🔍 Enter Instagram profile details to predict (0 = No, 1 = Yes):")

profile_pic = int(input("Profile picture present (1/0): "))
nums_length_username = float(input("Numbers/Length in username (0-1): "))
fullname_words = int(input("Fullname words count: "))
nums_length_fullname = float(input("Numbers/Length in fullname (0-1): "))
name_equals_username = int(input("Name equals username (1/0): "))
description_length = int(input("Description length (chars): "))
external_url = int(input("Has external URL (1/0): "))
private = int(input("Private account (1/0): "))
posts = int(input("Number of posts: "))
followers = int(input("Number of followers: "))
follows = int(input("Number of follows: "))

# Create a DataFrame from input
user_input = pd.DataFrame([[
    profile_pic, nums_length_username, fullname_words,
    nums_length_fullname, name_equals_username,
    description_length, external_url, private,
    posts, followers, follows
]], columns=X.columns)

# Scale and predict
user_scaled = scaler.transform(user_input)
pred = xgb.predict(user_scaled)[0]
prob = xgb.predict_proba(user_scaled)[0][1]

print("\n==============================")
print(f"Fake probability: {prob*100:.2f}%")
print("✅ Result:", "Fake Profile" if pred == 1 else "Real Profile")
print("==============================")
