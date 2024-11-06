from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score

from data.analysis import get_confusion_matrix
from data.data import DataSets
from data.eda import eda_bar_scatter_graphs
from data.helpers import drop_cols_with_low_var, find_n_componenents, get_best_params
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from data.labels import TARGET

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/rf', methods=['GET', 'POST'])
def random_forest_route():
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

    rf_param_grid = {
        'n_estimators': [30, 40, 50, 60],
        'max_depth': [15, 20, 25, 30],
    }
    best_rf_params, best_rf_score = get_best_params(
        RandomForestClassifier, rf_param_grid, X=x_train_transformed_data, y=y_train_transformed_data,
        static_params={'class_weight': 'balanced'}
    )

    rf_model = RandomForestClassifier(**best_rf_params)
    rf_model.fit(x_train_transformed_data, y_train_transformed_data)

    rf_prob = rf_model.predict_proba(x_test)[:, 1]
    threshold = 0.25  # Experiment with different thresholds
    rf_pred = (rf_prob >= threshold).astype(int)

    rf_accuracy = accuracy_score(y_test, rf_pred)
    fig = get_confusion_matrix(y_true=y_test, y_pred=rf_pred)

    return render_template('rfanalysis.html', score=rf_accuracy, plot=fig)

@app.route('/knn', methods=['GET', 'POST'])
def knn_route():
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

    knn_param_grid = {
        'n_neighbors': [5, 7, 9, 11, 13],
        'leaf_size': [5, 7, 9, 11, 13]
    }
    best_knn_params, best_knn_score = get_best_params(
        KNeighborsClassifier, knn_param_grid, X=x_train_transformed_data, y=y_train_transformed_data
    )

    print("Best KNN Parameters:", best_knn_params)

    knn_model = KNeighborsClassifier(**best_knn_params)
    knn_model.fit(x_train_transformed_data, y_train_transformed_data)

    knn_pred = knn_model.predict(x_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    fig = get_confusion_matrix(y_true=y_test, y_pred=knn_pred)

    return render_template('knnanalysis.html', score=knn_accuracy, plot=fig)

if __name__ == '__main__':
    app.run(debug=True)


