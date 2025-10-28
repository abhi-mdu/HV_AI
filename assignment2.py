import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# Load your data
housing = load_housing_data()
housing.dropna(subset=["total_bedrooms"])
housing_labels = housing['median_house_value']
housing_features = housing.drop('median_house_value', axis=1)

from sklearn.impute import SimpleImputer

# Identify numeric and categorical columns
num_attribs = housing_features.select_dtypes(include=['int64', 'float64']).columns
cat_attribs = housing_features.select_dtypes(include=['object']).columns

# Numeric pipeline with imputer
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Full pipeline for preprocessing
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing_features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(housing_prepared, housing_labels, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(kernel='rbf', C=100, gamma=0.1),
    "SVR": SVR(kernel='linear', C=100, gamma=0.1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    results[name] = mse

# Visualize results
plt.figure(figsize=(7,5))
plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
plt.ylabel('Mean Squared Error')
plt.title('Model Comparison: MSE')
plt.tight_layout()
plt.show()

# Print results for discussion
#for name, mse in results.items():
#    print(f"{name}: Mean Squared Error = {mse:.2f}")

from sklearn.metrics import mean_absolute_error
results_rmse = {}
results_mae = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    results[name] = mse
    results_rmse[name] = rmse
    results_mae[name] = mae
# Visualize RMSE results
plt.figure(figsize=(7,5))
plt.bar(results_rmse.keys(), results_rmse.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
plt.ylabel('Root Mean Squared Error')   
plt.title('Model Comparison: RMSE')
plt.tight_layout()
plt.show()
# Visualize MAE results
plt.figure(figsize=(7,5))
plt.bar(results_mae.keys(), results_mae.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
plt.ylabel('Mean Absolute Error')
plt.title('Model Comparison: MAE')
plt.tight_layout()
plt.show()

