from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

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
