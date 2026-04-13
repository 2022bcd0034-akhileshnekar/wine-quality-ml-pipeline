import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/winequality-red.csv")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

metrics = {"mse": mse, "r2": r2}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Run completed by: Akhilesh Nekar - 2022BCD0034")
print(f"MSE: {mse}")
print(f"R2: {r2}")