import pandas as pd
from typing import Any
from data.plots import bar_graphs, scatter_graphs


def eda_bar_scatter_graphs(feature_df: pd.DataFrame, target_column: str):
    feature_df = feature_df.rename(columns={
        feature_df.columns[48]: feature_df.columns[48].replace("/", " "),
        feature_df.columns[69]: feature_df.columns[69].replace("/", " ")
        }
    )
    columns_to_group = feature_df.columns.tolist()[:-1]
    bar_plots = []
    scatter_plots = []
    for col in columns_to_group:
        df = feature_df.groupby(by=col)[target_column].sum().reset_index()
        bar_plt = bar_graphs(feature_series=df[col], target_series=df[target_column], column_name=col)
        scatter_plt = scatter_graphs(feature_series=df[col], target_series=df[target_column], column_name=col)
        bar_plots.append(bar_plt)
        scatter_plots.append(scatter_plt)
    return bar_plots, scatter_plots

def eda_max_extraction(feature_df: pd.DataFrame, target_column: str):
    feature_max: dict[str, Any]  = {}
    columns_to_group = feature_df.columns.tolist()[:-1]
    for col in columns_to_group:
        df = feature_df.groupby(by=col)[target_column].sum().reset_index()
        feature_max[col] = df.iloc[df[target_column].idxmax()][col]
    return feature_max
