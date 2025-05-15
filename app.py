import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Load models on app start
with open('covid_models.pkl', 'rb') as f:
    models = pickle.load(f)

poly = models['polynomial_transformer']
lin_reg_poly = models['polynomial_regression']

# Replace this with your dataset's min date (yyyy-mm-dd)
first_date = pd.to_datetime("2020-01-22")

def predict_cases(date_str):
    try:
        date = pd.to_datetime(date_str)
    except Exception:
        return "Invalid date format. Please enter YYYY-MM-DD"

    days_since = (date - first_date).days
    if days_since < 0:
        return "Date must be after 2020-01-22"
    x_input = poly.transform([[days_since]])
    pred = lin_reg_poly.predict(x_input)[0]
    return f"Predicted confirmed COVID cases: {int(pred)}"

# Gradio interface
iface = gr.Interface(
    fn=predict_cases,
    inputs=gr.Textbox(label="Enter date (YYYY-MM-DD)"),
    outputs="text",
    title="COVID Confirmed Cases Predictor (Polynomial Regression)",
    description="Predict COVID confirmed cases for a given date based on polynomial regression."
)

iface.launch()
