# Breast Cancer disease-prediction

## Project Overview

The Breast Cancer Prediction System is a machine learning application that predicts whether a breast tumor is Benign (Non-Cancerous) 
or Malignant (Cancerous) using medical measurements from the Breast Cancer Wisconsin Dataset.
The project applies multiple machine learning classification algorithms, compares their performance, and deploys the best-performing model through an interactive Streamlit web application.

 ## Objectives

- Predict breast cancer using patient tumor characteristics.
- Compare the performance of multiple machine learning algorithms.
- Deploy the best-performing model using Streamlit.
- Provide a simple and interactive interface for users.

 ## Dataset

 - Dataset: Breast Cancer Wisconsin Diagnostic Dataset
 - Instances: 569
 - Features: 30 numerical features
 - Target Variable:
    * Benign (0)
    * Malignant (1)

The dataset contains measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

## Exploratory Data Analysis

The following analyses were performed:

- Dataset overview
- Missing value analysis
- Duplicate value check
- Class distribution
- Correlation heatmap
- Histograms
- Boxplots
- Outlier analysis

## Machine Learning Models

The following classification algorithms were implemented and evaluated:
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Random Forest
* XGBoost


## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 97.4% | 0.976 | 0.953 | 0.965 | 0.997 |
| K-Nearest Neighbors (KNN) | 94.7% | 0.930 | 0.930 | 0.930 | 0.982 |
| **Support Vector Machine (SVM)** | **98.2%** | **1.000** | **0.953** | **0.976** | **0.997** |
| Random Forest | 96.5% | 0.976 | 0.930 | 0.952 | 0.995 |
| XGBoost | 96.5% | 0.976 | 0.930 | 0.952 | 0.993 |

**🏆 Best Performing Model:** **Support Vector Machine (SVM)** with an accuracy of **98.2%**.


## Streamlit Application

The application provides:

* Project Overview
* Feature Explanation
* Interactive Prediction Interface
* Prediction Confidence Score
* Prediction Probability Table
* Medical Disclaimer

Users can enter the required tumor measurements, and the trained SVM model predicts whether the tumor is Benign or Malignant.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* Joblib

## 📁 Project Structure

```text
Breast_Cancer_Prediction/
│
├── app.py
├── pages/
│   └── predict.py
├── models/
│   ├── svm_model.pkl
│   └── scaler.pkl
├── notebooks/
├── data.csv
├── requirements.txt
└── README.md
```
