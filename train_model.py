# ==========================================================
# Train XGBoost Model and Save for Deployment
# ==========================================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 1. Load dataset
df = pd.read_csv("data/diabetes.csv")

# 2. Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features (IMPORTANT for production consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 6. Tuned XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

# 7. Train model
model.fit(X_train, y_train)

# 8. Save model
model.save_model("models/xgboost_model.json")

# 9. Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")