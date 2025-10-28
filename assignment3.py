import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
import urllib.request

# X_train, y_train 
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

# Assume X_train, y_train are already prepared

# Linear Regression (no major hyperparameters, but you can try fit_intercept)
lr = LinearRegression()
lr_params = {'fit_intercept': [True, False]}
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='neg_mean_squared_error')
lr_grid.fit(X_train, y_train)

lr_params2 = {'fit_intercept': [True, False]}
lr_grid2 = RandomizedSearchCV(lr, lr_params2, cv=5, scoring='neg_mean_squared_error')
lr_grid2.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeRegressor()
dt_params = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
dt_rand = RandomizedSearchCV(dt, dt_params, n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42)
dt_rand.fit(X_train, y_train)

# Random Forest
rf = RandomForestRegressor()
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
rf_rand = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=5, scoring='neg_mean_squared_error', random_state=42)
rf_rand.fit(X_train, y_train)

# SVR
svr = SVR()
svr_params = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100], 'gamma': ['scale', 'auto']}
svr_grid = GridSearchCV(svr, svr_params, cv=5, scoring='neg_mean_squared_error')
svr_grid.fit(X_train, y_train)

# Collect best scores and parameters
results = {
    'LR ': (lr_grid.best_params_, -lr_grid.best_score_),
    'LR(Random)': (lr_grid2.best_params_, -lr_grid2.best_score_),
    'DecisionTree': (dt_rand.best_params_, -dt_rand.best_score_),
    'RandomForest': (rf_rand.best_params_, -rf_rand.best_score_),
    'SVR': (svr_grid.best_params_, -svr_grid.best_score_)
}
models = list(results.keys())
mse_scores = [score for _, score in results.values()]

plt.bar(models, mse_scores, color=['blue', 'purple','orange', 'green', 'red'])
plt.ylabel('Best CV Mean Squared Error')
plt.title('Fine-Tuned Model Comparison')
plt.tight_layout()
plt.show()
