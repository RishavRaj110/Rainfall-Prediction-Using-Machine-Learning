import pandas as pd
print("\n" + "=" * 50)
print("       RAINFALL PREDICTION PROJECT")
print("=" * 50)

print("\nPROJECT OBJECTIVES:")
print("1. To analyze historical rainfall data.")
print("2. To identify rainfall patterns and trends.")
print("3. To predict annual rainfall using Machine Learning.")
print("4. To compare the performance of different ML algorithms.")
print("5. To evaluate prediction accuracy using performance metrics.")

print("=" * 50)
import numpy as np
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
import pandas as pd

df = pd.read_excel(r"C:\Users\RISHAV RAJ\OneDrive\Desktop\Rainfall_Prediction_Model_Ready.xlsx", sheet_name="Annual_Model_Data")

print(df.head())

# Data cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip()

df["TARGET_YEAR"] = pd.to_numeric(df["TARGET_YEAR"], errors="coerce")
df["RAIN_MM_PREVIOUS_YEAR"] = pd.to_numeric(df["RAIN_MM_PREVIOUS_YEAR"], errors="coerce")
df["TARGET_ANNUAL_RAIN_MM"] = pd.to_numeric(df["TARGET_ANNUAL_RAIN_MM"], errors="coerce")

df = df.dropna(subset=["SUBDIVISION", "TARGET_YEAR", "TARGET_ANNUAL_RAIN_MM"])
df = df[(df["TARGET_ANNUAL_RAIN_MM"] >= 0) & (df["RAIN_MM_PREVIOUS_YEAR"] >= 0)]

# Charts
plt.hist(df["TARGET_ANNUAL_RAIN_MM"], bins=30)
plt.title("Annual Rainfall Distribution")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.show()

df.groupby("TARGET_YEAR")["TARGET_ANNUAL_RAIN_MM"].mean().plot()
plt.title("Annual Rainfall Trend")
plt.ylabel("Average Rainfall (mm)")
plt.show()

# Prepare data
df["SUBDIVISION"] = LabelEncoder().fit_transform(df["SUBDIVISION"])

X = df[["SUBDIVISION", "TARGET_YEAR", "RAIN_MM_PREVIOUS_YEAR"]]
y = df["TARGET_ANNUAL_RAIN_MM"]

# Chronological split
split_year = int(df["TARGET_YEAR"].quantile(0.8))
train = df["TARGET_YEAR"] < split_year

X_train = X[train]
X_test = X[~train]
y_train = y[train]
y_test = y[~train]

# Five ML algorithms
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": make_pipeline(
        PolynomialFeatures(2), StandardScaler(), LinearRegression()
    ),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": make_pipeline(StandardScaler(), SVR())
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred

    results.append([
        name,
        mean_absolute_error(y_test, pred),
        mean_squared_error(y_test, pred),
        np.sqrt(mean_squared_error(y_test, pred)),
        r2_score(y_test, pred)
    ])

results = pd.DataFrame(
    results, columns=["Algorithm", "MAE", "MSE", "RMSE", "R2 Score"]
)

print("\nMODEL COMPARISON")
print(results.round(3))

# Comparison chart
plt.bar(results["Algorithm"], results["RMSE"])
plt.title("Model Comparison using RMSE")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# Best model
best_index = results["RMSE"].idxmin()
best_name = results.loc[best_index, "Algorithm"]
best_prediction = predictions[best_name]

print("\nBest Model:", best_name)

# Actual vs predicted chart
plt.scatter(y_test, best_prediction)
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")
plt.show()

print("\nProject completed!")
