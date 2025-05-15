import streamlit as st
import pickle
import numpy as np

# Load models
with open("covid_models.pkl", "rb") as f:
    models = pickle.load(f)

lin_reg = models["linear_regression"]
lin_reg_poly = models["polynomial_regression"]
svm = models["support_vector_regression"]
poly = models["polynomial_transformer"]

st.title("COVID-19 Case Predictor")

# User input for days since first observation
days = st.slider("Days Since First Report", 0, 500, 100)

# Make predictions
input_day = np.array(days).reshape(-1, 1)
input_poly = poly.transform(input_day)

lr_pred = lin_reg.predict(input_day)[0]
poly_pred = lin_reg_poly.predict(input_poly)[0]
svm_pred = svm.predict(input_day)[0]

# Show results
st.subheader("Predicted Confirmed Cases:")
st.write(f"Linear Regression: {int(lr_pred)}")
st.write(f"Polynomial Regression: {int(poly_pred)}")
st.write(f"SVM Regression: {int(svm_pred)}")
