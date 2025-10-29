import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("fake_profile_dataset.csv")

print("Columns in dataset:", df.columns.tolist())

# 2️⃣ Define features and target (Assume column 'fake' or 'label' is target)
target_col = "fake" if "fake" in df.columns else "label"
X = df.drop(columns=[target_col])
y = df[target_col]

# 3️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4️⃣ Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluate model
y_pred = model.predict(X_test)
print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6️⃣ Save model
joblib.dump(model, "fake_profile_model.pkl")
print("\n✅ Model saved successfully as 'fake_profile_model.pkl'")
print("Feature order:", list(X.columns))
