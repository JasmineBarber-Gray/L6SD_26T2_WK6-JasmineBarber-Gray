# libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_excel(r"C:\Users\DELL 5520\Downloads\Car_Purchasing_Data.xlsx")

# Remove personal data
df = df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)

# Features and target
# Features
X = df[["Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"]].values

# Target
y = df['Car Purchase Amount'].values

# Normalize features
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_scaled = (X - X_min) / (X_max - X_min)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# Feature Scaling
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    print("\n", name)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))
    
# new customer Prediction
new_customer = [[1, 35, 90000, 5000, 200000]]
new_customer_scaled = scaler.transform(new_customer)

print("Prediction (Random Forest):",
      models["Random Forest"].predict(new_customer_scaled)[0])