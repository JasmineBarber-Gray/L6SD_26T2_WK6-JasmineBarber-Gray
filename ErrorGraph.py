import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_excel(r"C:\Users\DELL 5520\Downloads\Car_Purchasing_Data.xlsx")

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