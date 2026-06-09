import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle

# Load dataset
data = pd.read_csv("house_data.csv")

# Display first rows
print("Dataset Preview:")
print(data.head())

# Features and Target
X = data[['Area', 'Bedrooms']]
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy check
mae = mean_absolute_error(y_test, predictions)

print("\nModel Trained Successfully")
print("Mean Absolute Error:", mae)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")

# Sample prediction
sample = [[2500, 5]]
predicted_price = model.predict(sample)

print("\nSample Prediction")
print("Area:", sample[0][0])
print("Bedrooms:", sample[0][1])
print("Predicted Price:", predicted_price[0])