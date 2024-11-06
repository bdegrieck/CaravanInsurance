import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def bar_graphs(feature_series: pd.Series, target_series: pd.Series, column_name: str):
    # Create a bar plot using Plotly
    fig = go.Figure(data=[go.Bar(x=feature_series, y=target_series)])

    # Add titles and labels
    fig.update_layout(
        title=f"{column_name} Insurance Count",
        xaxis_title=column_name,
        yaxis_title="Insurance Count",
        template="plotly_white"
    )

    # Save the figure to an HTML ffile
    html_str = pio.to_html(fig, full_html=False)

    return html_str


def scatter_graphs(feature_series: pd.Series, target_series: pd.Series, column_name: str):
    # Create a scatter plot using Plotly
    fig = go.Figure(data=[go.Scatter(x=feature_series, y=target_series, mode='markers')])

    # Add titles and labels
    fig.update_layout(
        title=f"{column_name} vs Insurance Count",
        xaxis_title=column_name,
        yaxis_title="Insurance Count",
        template="plotly_white"
    )

    html_str = pio.to_html(fig, full_html=False)

    return html_str

