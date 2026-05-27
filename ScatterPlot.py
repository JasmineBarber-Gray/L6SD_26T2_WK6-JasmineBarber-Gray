import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_excel(r"C:\Users\DELL 5520\Downloads\Car_Purchasing_Data.xlsx")

# Remove useless columns
df = df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)

# Features and target
X = df[["Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"]].values
y = df["Car Purchase Amount"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

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