import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']
X = X.astype('float32') / 255.0
y = y.astype('int64')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Grid Search
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=5, n_jobs=-1)
knn_grid.fit(X_train, y_train)
knn_best = knn_grid.best_estimator_
y_pred_knn = knn_best.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

# Random Forest Grid Search
rf_params = {'n_estimators': [100], 'max_depth': [None, 10, 20]}
rf = RandomForestClassifier()
rf_grid = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# SGD Grid Search
sgd_params = {'alpha': [0.0001, 0.001], 'max_iter': [1000], 'tol': [1e-3]}
sgd = SGDClassifier(loss='log_loss', random_state=42)
sgd_grid = GridSearchCV(sgd, sgd_params, cv=5, n_jobs=-1)
sgd_grid.fit(X_train, y_train)
sgd_best = sgd_grid.best_estimator_
y_pred_sgd = sgd_best.predict(X_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)

# Print accuracies
print(f"KNN Test Accuracy: {acc_knn:.4f}")
print(f"Random Forest Test Accuracy: {acc_rf:.4f}")
print(f"SGD Test Accuracy: {acc_sgd:.4f}")

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, y_pred, title in zip(axes, [y_pred_knn, y_pred_rf, y_pred_sgd], ['KNN', 'Random Forest', 'SGD']):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{title} Confusion Matrix")
plt.tight_layout()
plt.show()
