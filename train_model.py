import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

data = {
    "age": [25,45,36,50,22,40,30,29],
    "income": [3000,7000,4000,10000,2000,6000,3500,3200],
    "loan_amt": [1000,2000,1500,5000,500,1200,2000,1700],
    "default": [0,0,0,1,0,0,1,0]
}
df = pd.DataFrame(data)
X = df[["age","income","loan_amt"]]
y = df["default"]

pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, random_state=42))
pipe.fit(X, y)

joblib.dump(pipe, "model.joblib")
print("Model saved -> model.joblib")
