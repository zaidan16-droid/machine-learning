import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# === Dataset dummy ===
# Fitur: age, income, loan_amt
# Label: 0 = lancar, 1 = default
data = {
    "age": [22, 25, 30, 40, 45, 50],
    "income": [2000, 3000, 4000, 6000, 7000, 10000],
    "loan_amt": [500, 1000, 1500, 2000, 2500, 5000],
    "default": [0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["age", "income", "loan_amt"]]
y = df["default"]

# === Model ringan ===
model = LogisticRegression()
model.fit(X, y)

# Simpan model ke file
joblib.dump(model, "model.joblib")
print("✅ Model disimpan ke model.joblib")
