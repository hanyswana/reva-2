import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES
from scipy import sparse
import numpy as np

# st.markdown("""
# <style>
# .custom-font {font-size: 16px; font-weight: bold;}
# </style> """, unsafe_allow_html=True)

# st.markdown('<p class="custom-font">Absorbance data :</p>', unsafe_allow_html=True)

def csv_data(uploaded_file):
    if uploaded_file is not None:
        # Assuming the uploaded file is a CSV, read it into a DataFrame
        df = pd.read_csv(uploaded_file, usecols=range(3, 12))  # Columns D to M have indexes 3 to 11

        # Convert to numeric, handling errors by coercing invalid values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # You might want to process or manipulate the dataframe here as per your requirements
        wavelengths = df.columns

        return df, wavelengths
    else:
        st.write("No file uploaded.")
        return None, None
        

def load_model(model_dir):
    model = tf.saved_model.load(model_dir)
    return model

def predict_with_model(model, input_data):

    input_array = input_data.to_numpy(dtype='float64')
    input_array_reshaped = input_array.reshape(-1, 19)  # Adjust to match the number of features your model expects
    input_tensor = tf.convert_to_tensor(input_array_reshaped, dtype=tf.float64)
    predictions = model(input_tensor)
    return predictions.numpy()  # Convert predictions to numpy array if needed

def main():

    # Simulate CSV upload through Streamlit file_uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Get data from uploaded CSV
    absorbance_data, wavelengths = csv_data(uploaded_file)
    
    # Define model paths with labels
    model_paths_with_labels = [
        ('R39', 'reva-lablink-hb-125-(original-data).csv_r2_0.39_2024-02-15_11-55-27'),
        ('Normalized Manhattan (R38)', 'lablink-hb-norm-manh.csv_best_model_2024-02-23_00-52-51_r38'),
        ('Normalized Manhattan (R40)', 'lablink-hb-norm-manh.csv_best_model_2024-02-22_02-09-42_r40'),
        ('SNV (R49)', 'snv_transformed-1.csv_best_model_2024-02-29_22-15-55')
    ]

    for label, model_path in model_paths_with_labels:
        # Load the model
        model = load_model(model_path)
        # st.write(model)

                # Example conditional block for selecting the correct dataset based on the model's intended use:
        if label == 'R39':
            input_data = absorbance_data
        elif label == 'Normalized Manhattan (R38)':
            input_data = absorbance_normalized_manh_data
        elif label == 'Normalized Manhattan (R40)':
            input_data = absorbance_normalized_manh_data
        elif label == 'SNV (R49)':
            input_data = absorbance_snv_data
        else:
            continue  # Skip if label does not match expected values

        predictions = predict_with_model(model, input_data)
        predictions_value = predictions[0][0] 
        
    
        st.markdown("""
        <style>
        .label {font-size: 16px; font-weight: bold; color: black;}
        .value {font-size: 60px; font-weight: bold; color: blue;}
        .high-value {font-size: 60px; font-weight: bold; color: red;}
        </style> """, unsafe_allow_html=True)

                # Add condition for prediction value
        if predictions_value > 25:
            display_value = f'<span class="high-value">High value : ({predictions_value:.1f} g/dL)</span>'
        else:
            display_value = f'<span class="value">{predictions_value:.1f} g/dL</span>'
        
        # Display label and prediction value
        st.markdown(f'<span class="label">Haemoglobin ({label}):</span><br>{display_value}</p>', unsafe_allow_html=True)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, absorbance_data.iloc[0], marker='o', linestyle='-', color='b')
    plt.xlabel('Wavelength (nm)', fontweight='bold', fontsize=14)
    plt.ylabel('Absorbance', fontweight='bold', fontsize=14)
    plt.xticks(rotation='vertical', fontweight='bold', fontsize=12)
    plt.yticks(fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
    
if __name__ == "__main__":
    main()
