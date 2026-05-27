import pandas as pd
import numpy as np

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

# New customer prediction
new_customer = [[1, 35, 90000, 5000, 200000]]
new_customer_scaled = scaler.transform(new_customer)

print("Prediction (Random Forest):",
      models["Random Forest"].predict(new_customer_scaled)[0])