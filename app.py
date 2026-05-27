import tkinter as tk
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_car_purchase_model.pkl")
scaler = joblib.load("car_purchase_scaler.pkl")

# Create window
window = tk.Tk()
window.title("Car Purchase Predictor")
window.geometry("350x400")

# Labels + inputs
tk.Label(window, text="Gender (0 = Female, 1 = Male)").pack()
gender_entry = tk.Entry(window)
gender_entry.pack()

tk.Label(window, text="Age").pack()
age_entry = tk.Entry(window)
age_entry.pack()

tk.Label(window, text="Annual Salary").pack()
salary_entry = tk.Entry(window)
salary_entry.pack()

tk.Label(window, text="Credit Card Debt").pack()
debt_entry = tk.Entry(window)
debt_entry.pack()

tk.Label(window, text="Net Worth").pack()
networth_entry = tk.Entry(window)
networth_entry.pack()

# Result label
result_label = tk.Label(window, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=20)

# Predict function
def predict():
    features = np.array([[
        float(gender_entry.get()),
        float(age_entry.get()),
        float(salary_entry.get()),
        float(debt_entry.get()),
        float(networth_entry.get())
    ]])

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    result_label.config(text=f"Predicted: ${prediction:,.2f}")

# Button
tk.Button(window, text="Predict", command=predict).pack()

# Run app
window.mainloop()