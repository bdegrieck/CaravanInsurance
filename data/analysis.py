import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.io as pio

def get_confusion_matrix(y_true: pd.Series, y_pred: pd.Series):
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Create a Plotly heatmap for the confusion matrix
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f'Predicted {i}' for i in range(matrix.shape[1])],
        y=[f'True {i}' for i in range(matrix.shape[0])],
        colorscale='Blues',
        text=matrix,
        texttemplate='%{text}',
        hoverinfo='z'
    ))

    # Add titles and labels
    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted Label'),
        yaxis=dict(title='True Label')
    )

    # Convert the Plotly figure to an HTML string
    html_str = pio.to_html(fig, full_html=False)

    return html_str