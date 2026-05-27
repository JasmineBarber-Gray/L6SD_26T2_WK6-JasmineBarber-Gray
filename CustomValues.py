import joblib
import numpy as np

# Load saved model and scaler
loaded_model = joblib.load("best_car_purchase_model.pkl")
loaded_scaler = joblib.load("car_purchase_scaler.pkl")

# Custom customer input
# Order MUST match training data:
# Gender, Age, Annual Salary, Credit Card Debt, Net Worth

custom_customer = np.array([[1, 35, 75000, 5000, 250000]])

# Scale input
custom_customer_scaled = loaded_scaler.transform(custom_customer)

# Predict
predicted_amount = loaded_model.predict(custom_customer_scaled)

print("Predicted Car Purchase Amount:", predicted_amount[0])