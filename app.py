import streamlit as st
st.title("Breast Cancer Prediction System")

st.header("Project Overview")
st.write("The Breast Cancer Prediction System is a machine learning application " \
" that classifies breast tumors as Benign (non-cancerous) or Malignant (cancerous) " \
"based on patient data. It analyzes 30 medical features, such as tumor radius, texture, " \
"perimeter, area, smoothness, compactness, concavity, and symmetry, to assist in early " \
"breast cancer diagnosis.he goal is to support early diagnosis by providing a fast and accurate "\
"classification based on medical measurements.")

st.subheader("Features of Breast Cancer prediction system")
st.markdown("""
- Predicts whether a breast tumor is **Benign** or **Malignant**.
- Displays Exploratory Data Analysis (EDA) including distribution plots, correlation 
   heatmap, and outlier analysis.
- Compares the performance of multiple machine learning models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
- Uses the best-performing **Support Vector Machine (SVM)** model for prediction.
- Evaluates models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Interactive and user-friendly interface built with Streamlit.
""")

st.markdown("### Technologies Used")
st.markdown("- Python")
st.markdown("- Pandas")
st.markdown("- NumPy")
st.markdown("- Scikit-learn")
st.markdown("- Streamlit")

st.markdown("---")
st.caption("BreastCancer Prediction System | Streamlit + machine learning")
