# libraries
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel(r"C:\Users\DELL 5520\Downloads\Car_Purchasing_Data.xlsx")

# Remove personal data
df = df.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)

# Features and target
# Features
X = df[['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']].values

# Target
y = df['Car Purchase Amount'].values

# Normalize features
X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_scaled = (X - X_min) / (X_max - X_min)

# Split dataset into training and testing sets

split = int(0.8 * len(X_scaled))

X_train = X_scaled[:split]
X_test = X_scaled[split:]

y_train = y[:split]
y_test = y[split:]

# linear regression using normal equation
# θ = (X^T X)^-1 X^T y

theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predict on test set
y_pred = X_test @ theta

# Evaluate model
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# new customer prediction
new_customer = np.array([[35, 90000, 5000, 200000]])

new_customer_scaled = (new_customer - X_min) / (X_max - X_min)

new_customer_scaled = np.c_[np.ones(1), new_customer_scaled]

prediction = new_customer_scaled @ theta

print("Predicted Car Purchase Amount:", prediction[0])
