import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objs as go

def preprocess_data(file_path):
    """
    Membaca file input dan memvalidasi kolom.
    """
    data = pd.read_csv(file_path)
    if "time" not in data.columns:
        raise ValueError("Uploaded file must contain a 'time' column.")
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns found for forecasting.")
    return data, numeric_columns

def train_and_forecast(file_path, column, horizon):
    """
    Melatih model Holt-Winters dengan data baru dan melakukan forecasting.
    """
    data, numeric_columns = preprocess_data(file_path)

    # Jika "overall" dipilih, gunakan kolom "total load actual"
    if column == "overall":
        if "total load actual" not in data.columns:
            raise ValueError("The dataset does not contain a 'total load actual' column for overall forecasting.")
        series = data["total load actual"].resample("D").sum()
        column_name = "Total Load Actual"
    else:
        series = data[column].resample("D").sum()
        column_name = column

    # Train Holt-Winters Model
    model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=365)
    fit = model.fit()

    # Generate Forecast
    steps = 180 if horizon == "6_months" else 1825
    forecast = fit.forecast(steps=steps)
    forecast_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps)

    # Plot dengan Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name="Historical Data", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode="lines", name="Forecast", line=dict(color="red", dash="dash")))
    fig.update_layout(
        title=f"Forecast for {column_name} ({horizon})",
        xaxis_title="Date",
        yaxis_title="Values",
        template="plotly_white"
    )
    return fig.to_html(full_html=False)