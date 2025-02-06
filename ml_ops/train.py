import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("aws_cloud_lockin_dataset.csv")

# Split features and target variable
X = df.drop(columns=["Cloud Lock-in Score"])
y = df["Cloud Lock-in Score"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Save the trained model
joblib.dump(model, "cloud_lockin_model.pkl")
print("Model saved as cloud_lockin_model.pkl")