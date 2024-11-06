from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from data.data import DataSets
from data.analysis import get_confusion_matrix
from data.helpers import find_n_componenents, get_best_params, drop_cols_with_low_var


x_train = DataSets.x_train.dropna()
y_train = DataSets.y_train.dropna()
x_test = DataSets.x_test
y_test = DataSets.y_test

x_train = drop_cols_with_low_var(df=x_train, variance=0.95)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

smote = SMOTE(random_state=42)
x_train_transformed_data, y_train_transformed_data = smote.fit_resample(x_train_scaled, y_train)

n_components = find_n_componenents(df=x_train_transformed_data, threshold=0.95)

pca = PCA(n_components=n_components)
x_train_transformed_data = pca.fit_transform(x_train_transformed_data)
x_test = pca.transform(x_test_scaled)

print(f"Optimal number of PCA components: {n_components}")

rf_param_grid = {
    'n_estimators': [30, 40, 50, 60],
    'max_depth': [15, 20, 25, 30],
}
best_rf_params, best_rf_score = get_best_params(
    RandomForestClassifier, rf_param_grid, X=x_train_transformed_data, y=y_train_transformed_data,
    static_params={'class_weight': 'balanced'}
)

print("Best Random Forest Parameters:", best_rf_params)

knn_param_grid = {
    'n_neighbors': [5, 7, 9, 11, 13],
    'leaf_size': [5, 7, 9, 11, 13]
}
best_knn_params, best_knn_score = get_best_params(
    KNeighborsClassifier, knn_param_grid, X=x_train_transformed_data, y=y_train_transformed_data
)

print("Best KNN Parameters:", best_knn_params)

# Train and evaluate the RandomForest model with the best parameters
rf_model = RandomForestClassifier(**best_rf_params)
rf_model.fit(x_train_transformed_data, y_train_transformed_data)

rf_prob = rf_model.predict_proba(x_test)[:, 1]
# Adjust threshold to improve sensitivity
threshold = 0.25  # Experiment with different thresholds
rf_pred = (rf_prob >= threshold).astype(int)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_matrix = get_confusion_matrix(y_true=y_test, y_pred=rf_pred)
print(f"Accuracy score RF: {rf_accuracy}")

# Train and evaluate the KNeighborsClassifier model with the best parameters
knn_model = KNeighborsClassifier(**best_knn_params)
knn_model.fit(x_train_transformed_data, y_train_transformed_data)

# Predict and evaluate on the test set
knn_pred = knn_model.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_matrix = get_confusion_matrix(y_true=y_test, y_pred=knn_pred)
print(f"Accuracy score knn: {knn_accuracy}")
