import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the trained model
model = joblib.load("fake_profile_model.pkl")

# Get user input
print("🔍 Enter Instagram profile details (0 = No, 1 = Yes):")
profile_pic = int(input("Profile picture present (1/0): "))
username_ratio = float(input("Numbers/Length in username (0–1): "))
fullname_words = int(input("Fullname words count: "))
fullname_ratio = float(input("Numbers/Length in fullname (0–1): "))
name_equals_username = int(input("Name equals username (1/0): "))
desc_len = int(input("Description length (chars): "))
has_url = int(input("Has external URL (1/0): "))
private = int(input("Private account (1/0): "))
posts = int(input("Number of posts: "))
followers = int(input("Number of followers: "))
following = int(input("Number of follows: "))

# Combine into array
features = np.array([[profile_pic, username_ratio, fullname_words, fullname_ratio,
                      name_equals_username, desc_len, has_url, private,
                      posts, followers, following]])

# Predict
prob = model.predict_proba(features)[0][1] * 100
prediction = model.predict(features)[0]

print("\n==============================")
print(f"Fake probability: {prob:.2f}%")
print("✅ Result:", "Fake Profile" if prediction == 1 else "Real Profile")
