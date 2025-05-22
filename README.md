# 🦠 COVID-19 Case Prediction using Polynomial Regression

A machine learning project that predicts the number of COVID-19 cases based on a given future date using **Polynomial Regression**. Built using Python and deployed with an interactive UI on **Hugging Face Spaces**.

---

## Table of Contents

- [About the Project](#-about-the-project)  
- [Live Demo](#-live-demo)  
- [Technologies Used](#-technologies-used)  
- [Installation & Usage](#-installation--usage)  
- [Model Overview](#-model-overview)  
- [Deployment Guide](#-deployment-guide)  
- [Project Structure](#-project-structure)  
---

Here is the reference the overleaf code:https://www.overleaf.com/read/cqsrmbdqndxs#154651

##  About the Project

This project provides a web-based tool for predicting future COVID-19 case numbers.

- **Input**: Date (e.g., `2025-06-01`)
- **Output**: Predicted number of COVID-19 cases

The model is trained on historical case data and uses polynomial regression to fit the curve and forecast future values.

---

##  Live Demo

Try the live version hosted on Hugging Face Spaces:  
[kiyoya123/covid19](https://huggingface.co/spaces/kiyoya123/covid19)

---

## Technologies Used

- **Python** – Programming language  
- **Scikit-learn** – Machine learning model  
- **Pandas** – Data manipulation  
- **NumPy** – Numerical operations  
- **Matplotlib** – Data visualization  
- **Gradio** – Web-based UI for model interaction  

---

## Installation & Usage

Follow the steps below to run the project locally:

### 1. Clone the Repository
git clone https://github.com/Saksham6395/covid19.git
cd covid19

 Install Dependencies : pip install -r requirements.txt

 Run the App  : python app.py

## Model Overview
Model Type: Polynomial Regression (using Scikit-learn)

Features:

Transforms date into numerical format for fitting

Applies polynomial features to capture non-linear trends

Predicts based on best-fit curve of historical data

Training Data: Historical COVID-19 case records (dataset.csv)

Deployment: Saved model loaded using pickle


## Deployment Guide
To deploy this project on Hugging Face Spaces:

Push your repository to GitHub.

Create a new Space at:https://huggingface.co

Link your GitHub repo.

Ensure you include:

app.py

requirements.txt

README.md

Model and dataset files

Refer to the official configuration guide here:https://huggingface.co/docs/hub/spaces-config-reference
here is the drive link for complete process video and apk link:https://drive.google.com/drive/folders/1GG-OLPPlkFwJ1-UmPMMRFrnA50pbl22d?usp=drive_link


##  Project Structure

| File/Folder       | Description                               |
|-------------------|-------------------------------------------|
| `app.py`          | Main Gradio app script                    |
| `model.pkl`       | Trained regression model (saved as pickle)|
| `dataset.csv`     | Historical COVID-19 case data             |
| `requirements.txt`| Python dependencies                       |


