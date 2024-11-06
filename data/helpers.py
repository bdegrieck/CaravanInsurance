import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from itertools import product

from data.data import DataSets
from data.eda import eda_max_extraction
from data.labels import L0_DICT, L1_DICT, L2_DICT, L0_col, L1_col, L2_col, L3_col, L3_DICT, L4_col, L4_DICT


def load_main_traits():
    data_loader = DataSets()
    train_df = data_loader.train_df

    feature_max = eda_max_extraction(feature_df=train_df, target_column="Target")

    feature_max[L0_col] = L0_DICT[feature_max[L0_col]]
    feature_max[L1_col] = L1_DICT[feature_max[L1_col]]
    feature_max[L2_col] = L2_DICT[feature_max[L2_col]]

    for col_3 in L3_col:
        feature_max[col_3] = L3_DICT[feature_max[col_3]]

    for col_4 in L4_col:
        feature_max[col_4] = L4_DICT[feature_max[col_4]]

    for feature, statistic in feature_max.items():
        print(f"Feature: {feature} - Most common insurance buyers: {statistic}")

def drop_cols_with_low_var(df: pd.DataFrame, variance: float) -> pd.DataFrame:
    columns_to_drop = []
    for column in df.columns.to_list():
        val_cal = df[column].value_counts().sort_values(ascending=False)
        sum = val_cal.sum()
        second_largest_val = val_cal.iloc[1]
        if second_largest_val / sum < variance:
            columns_to_drop.append(column)
            df.drop(columns=column)
    return df

def find_n_componenents(df: pd.DataFrame, threshold: float) -> int:
    pca_model_1 = PCA()
    pca_model_1.fit(df)
    cum_var = np.cumsum(pca_model_1.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= threshold) + 1
    return n_components


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, folds: int) -> list:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Model: {model} Average Accuracy: {score.mean() * 100:.2f} ± {score.std():.2f}")
    return score


def get_best_params(model_class, param_grid: dict[str, list], X: pd.DataFrame, y: pd.Series, static_params: dict = {}, folds = 5):
    best_score = -np.inf
    best_params = None
    param_combinations = list(product(*param_grid.values()))

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        model = model_class(**param_dict, **static_params)

        score = evaluate_model(model=model, folds=folds, X=X, y=y)

        if score.mean() > best_score:
            best_score = score.mean()
            best_params = param_dict

    return best_params, best_score