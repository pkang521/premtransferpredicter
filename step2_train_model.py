# STEP 2: Train a linear regression and check how good it is.

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("player_seasons.csv")

# ---------------------------------------------------------------
# Choose the features and target
# ---------------------------------------------------------------
# Market values are very skewed (most players are cheap, a few cost €100m+)
# Taking the log makes the data much better behaved for linear regression
df["log_value"] = np.log(df["market_value"])

# Value rises until the mid-20s, then falls.
df["age_squared"] = df["age"] ** 2

# Turn position (text) into 0/1 columns the model can use.
df = pd.get_dummies(df, columns=["position"], drop_first=True, dtype=int)
position_cols = [c for c in df.columns if c.startswith("position_")]

features = ["age", "age_squared", "minutes", "goals_per_90", "assists_per_90",
            "club_points_per_game"] + position_cols
target = "log_value"

# ---------------------------------------------------------------
# Split into training and test data BY SEASON
# ---------------------------------------------------------------
# Training on older seasons and test on the most recent one.
last_season = df["season"].max()
train = df[df["season"] < last_season]
test = df[df["season"] == last_season]
print(f"Training on {len(train)} rows, testing on {len(test)} rows (season {last_season})")

# ---------------------------------------------------------------
# Train
# ---------------------------------------------------------------
model = LinearRegression()
model.fit(train[features], train[target])

# ---------------------------------------------------------------
# Evaluate on the test season
# ---------------------------------------------------------------
test = test.copy()
test["predicted_value"] = np.exp(model.predict(test[features]))   # undos the log
r2 = r2_score(test[target], np.log(test["predicted_value"]))

# How far off are we, in %?
test["error_pct"] = (test["predicted_value"] - test["market_value"]).abs() / test["market_value"]

print(f"\nR² on the test season: {r2:.2f}")
print(f"Median error: {test['error_pct'].median():.0%}")
print("(R² = 1 would be perfect; 0 = no better than guessing the average)")

#exp(coef) - 1 = % change in value per unit of the feature
print("\nWhat each feature does to value:")
for name, coef in zip(features, model.coef_):
    print(f"  {name:22s} {coef:+.4f}")

# ---------------------------------------------------------------
# Save the model and the predictions for the app
# ---------------------------------------------------------------
joblib.dump({"model": model, "features": features}, "model.joblib")
test.to_csv("predictions.csv", index=False)
print("\nSaved model.joblib and predictions.csv")
