import joblib
import streamlit as st
import pandas as pd
import numpy as np

model= joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
st.success("Model loaded successfully")

st.info("Enter the tumor characteristics below to predict whether the tumor "
"is Benign or Malignant.")
with st.expander("ℹ️ Feature Explanation"):

    st.markdown("""
### Understanding the Input Features

The model uses **30 tumor characteristics** divided into three groups:

- **Mean Features:** Average value of each tumor measurement.
- **Standard Error (SE):** Measures the variation or uncertainty in each feature.
- **Worst Features:** Largest (most severe) value recorded for each measurement.

### Main Tumor Characteristics

- **Radius:** Distance from the center of the tumor to its boundary.
- **Texture:** Variation in the gray-scale appearance of the tumor.
- **Perimeter:** Length of the tumor boundary.
- **Area:** Size of the tumor.
- **Smoothness:** Degree of smoothness of the tumor surface.
- **Compactness:** How compact or dense the tumor shape is.
- **Concavity:** Extent of inward curves in the tumor.
- **Concave Points:** Number of inward indentations on the boundary.
- **Symmetry:** Symmetry of the tumor shape.
- **Fractal Dimension:** Complexity of the tumor boundary.
""")
    

st.subheader("Mean Features")
with st.expander("Mean Features", expanded=True):
 col1, col2 = st.columns(2)

 with col1:
    radius_mean = st.number_input("Radius Mean", value=14.0)
    perimeter_mean = st.number_input("Perimeter Mean", value=90.0)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.09)
    concavity_mean = st.number_input("Concavity Mean", value=0.08)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.18)

 with col2:
    texture_mean = st.number_input("Texture Mean", value=19.0)
    area_mean = st.number_input("Area Mean", value=650.0)
    compactness_mean = st.number_input("Compactness Mean", value=0.10)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.05)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.06)

st.subheader("Standard Error (SE) Features")
with st.expander("Standard error features",expanded=True):
 col1, col2 = st.columns(2)

 with col1:
    radius_se = st.number_input("Radius SE", value=0.40)
    perimeter_se = st.number_input("Perimeter SE", value=2.80)
    smoothness_se = st.number_input("Smoothness SE", value=0.007)
    concavity_se = st.number_input("Concavity SE", value=0.03)
    symmetry_se = st.number_input("Symmetry SE", value=0.02)

 with col2:
    texture_se = st.number_input("Texture SE", value=1.20)
    area_se = st.number_input("Area SE", value=40.0)
    compactness_se = st.number_input("Compactness SE", value=0.025)
    concave_points_se = st.number_input("Concave Points SE", value=0.012)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.004)

st.subheader("Worst Features")
with st.expander("Worst Features",expanded=True):
 col1, col2 = st.columns(2)

 with col1:
    radius_worst = st.number_input("Radius Worst", value=16.0)
    perimeter_worst = st.number_input("Perimeter Worst", value=107.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.13)
    concavity_worst = st.number_input("Concavity Worst", value=0.27)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.29)

 with col2:
    texture_worst = st.number_input("Texture Worst", value=25.0)
    area_worst = st.number_input("Area Worst", value=880.0)
    compactness_worst = st.number_input("Compactness Worst", value=0.25)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.11)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)

input_data = np.array([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,
    compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,                                 
    radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,
    concave_points_se,symmetry_se,fractal_dimension_se,
    radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,
    concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst
]])

if st.button("Predict"):

    with st.spinner("Analyzing tumor characteristics..."):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

    st.success("Prediction completed!")

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(" Malignant Tumor Detected")
    else:
        st.success(" Benign Tumor Detected")

    benign_prob = probability[0][0] * 100
    malignant_prob = probability[0][1] * 100
    confidence = max(benign_prob, malignant_prob)

    st.subheader("Confidence")
    st.progress(confidence / 100)
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Probability (%)": [
            round(benign_prob, 2),
            round(malignant_prob, 2)
        ]
    })

    st.dataframe(prob_df, hide_index=True, use_container_width=True)