import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_excel(r"C:\Users\DELL 5520\Downloads\Car_Purchasing_Data.xlsx")

# removes personal data
df = df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)

# Features + target
X = df[["Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"]]
y = df["Car Purchase Amount"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(), # Support Vector Regression
    "KNN": KNeighborsRegressor(n_neighbors=5) # K-Nearest Neighbors Regression
}

results = []

# Train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, r2])

    print(f"{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("-" * 30)

# comparison table
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2 Score"])
results_df

# find best model based on R2 Score
best_model = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]
print("Best Model:")
print(best_model)

# Baseline prediction
baseline_pred = np.full_like(y_test, y_train.mean())

# Evaluate baseline
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y_test, baseline_pred)

print("Baseline Results")
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2:", baseline_r2)

# Comparison
print("\nBaseline vs Best Model Comparison")
print("Baseline R2:", baseline_r2)
print("Best Model R2:", results_df["R2 Score"].max())

# Train best model on full dataset
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

# Metrics - TRAIN
train_r2 = r2_score(y_train, train_pred)
train_mae = mean_absolute_error(y_train, train_pred)

# Metrics - TEST
test_r2 = r2_score(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print("Training R2:", train_r2)
print("Training MAE:", train_mae)
print("Testing R2:", test_r2)
print("Testing MAE:", test_mae)

# Save model
joblib.dump(best_model, "best_car_purchase_model.pkl")

# Save scaler
joblib.dump(scaler, "car_purchase_scaler.pkl")

# plot actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, label="Predictions")

# Median values
median_actual = np.median(y_test)
median_pred = np.median(y_pred)

# Median lines
plt.axvline(median_actual, color='red', linestyle='--', label='Median Actual')
plt.axhline(median_pred, color='blue', linestyle='--', label='Median Predicted')

# Perfect prediction line (y = x)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))

plt.plot([min_val, max_val], [min_val, max_val],
         color='green', linestyle='-', label='Perfect Prediction (y=x)')

# Labels
plt.xlabel("Actual Car Purchase Amount")
plt.ylabel("Predicted Car Purchase Amount")
plt.title("Actual vs Predicted Car Purchase Amount")
plt.legend()
plt.grid(True)

plt.show()

# Error Chart
# Prepare features and target
X = df[['Age', 'Annual Salary']]  # Example features
y = df['Car Purchase Amount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Errors
errors = y_test - y_pred

# Histogram
plt.hist(errors, bins=25, alpha=0.7, edgecolor='black', label="Error Distribution")

# Zero error line (perfect prediction reference)
plt.axvline(0, color='green', linestyle='-', linewidth=2, label="Zero Error (Perfect Prediction)")

# Labels
plt.title("Advanced Prediction Error Distribution")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

plt.show()
