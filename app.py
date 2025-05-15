import gradio as gr
import pickle
import numpy as np

# Load models dictionary
with open("covid_models.pkl", "rb") as f:
    models = pickle.load(f)

# Use polynomial regression (or classification) model if available
model = models['polynomial_regression']  # replace with your exact key from models

def predict(age, fever, body_pain, runny_nose, diff_breathing):
    # Convert Yes/No radio inputs to 1/0
    body_pain = 1 if body_pain == "Yes" else 0
    runny_nose = 1 if runny_nose == "Yes" else 0
    diff_breathing = 1 if diff_breathing == "Yes" else 0

    # Create input feature array as model expects
    input_data = np.array([[age, fever, body_pain, runny_nose, diff_breathing]])

    # Predict (assuming model is a classifier, threshold at 0.5)
    prediction = model.predict(input_data)[0]
    result = "Positive" if prediction >= 0.5 else "Negative"
    return result

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(1, 100, step=1, label="Age"),
        gr.Slider(95.0, 105.0, step=0.1, label="Fever (°F)"),
        gr.Radio(["No", "Yes"], label="Body Pain"),
        gr.Radio(["No", "Yes"], label="Runny Nose"),
        gr.Radio(["No", "Yes"], label="Difficulty in Breathing")
    ],
    outputs="text",
    title="COVID-19 Prediction App",
    description="Predict COVID-19 status based on symptoms."
)

if __name__ == "__main__":
    demo.launch()
