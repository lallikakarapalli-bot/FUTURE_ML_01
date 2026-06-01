import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Sales dataset
sales_data = {
    "Month": [1, 2, 3, 4, 5, 6],
    "Sales": [100, 120, 130, 150, 170, 200]
}

# Create DataFrame
df = pd.DataFrame(sales_data)

# Features and target
X = df[["Month"]]
y = df["Sales"]

# Build and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Future months for prediction
future_months = pd.DataFrame({"Month": [7, 8, 9]})

# Predict future sales
future_sales = lr_model.predict(future_months)

print("Predicted Sales for Upcoming Months:")
for month, sales in zip(future_months["Month"], future_sales):
    print(f"Month {month}: {sales:.2f}")

# Predictions on training data
predicted_sales = lr_model.predict(X)

# Model evaluation
mae = mean_absolute_error(y, predicted_sales)
print(f"\nMean Absolute Error (MAE): {mae:.2f}")

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(df["Month"], df["Sales"], label="Actual Sales")
plt.plot(df["Month"], predicted_sales, label="Regression Line")

# Labels and title
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting Using Linear Regression")
plt.legend()
plt.grid(True)

# Display graph
plt.show()