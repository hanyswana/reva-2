import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# st.markdown("""
# <style>
# .custom-font {font-size: 16px; font-weight: bold;}
# </style> """, unsafe_allow_html=True)

# st.markdown('<p class="custom-font">Absorbance data :</p>', unsafe_allow_html=True)

def csv_data():
    file_path = 'SNV & br 25 data.csv'  # Adjust the path if the file is in a specific folder
    df = pd.read_csv(file_path, usecols=range(3, 13))  # Columns D to M have indexes 3 to 11

    # Convert to numeric, handling errors by coercing invalid values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    st.write(df)
    # Extracting wavelengths or column names if needed
    wavelengths = df.columns

    # absorbance_data = df.iloc[13]
    # # st.write(df)
    # st.write(absorbance_data)

    return df, wavelengths

def load_model(model_dir):
    if model_dir.endswith('.tflite'):  # Check if model is a TensorFlow Lite model
        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path=model_dir)
        interpreter.allocate_tensors()
        return interpreter
    else:
        # Load TensorFlow SavedModel
        model = tf.saved_model.load(model_dir)
        return model


def predict_with_model(model, input_data):
    if isinstance(model, tf.lite.Interpreter):  # Check if model is TensorFlow Lite Interpreter
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        input_data = input_data.astype('float32')
        input_data = np.expand_dims(input_data, axis=0)
        
        # Assuming input_data is already in the correct shape and type
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        return predictions  # This will be a numpy array
    else:
        # Existing prediction code for TensorFlow SavedModel
        input_array = input_data.to_numpy(dtype='float32')
        input_array_reshaped = input_array.reshape(-1, 10)  # Adjust to match the number of features your model expects
        input_tensor = tf.convert_to_tensor(input_array_reshaped, dtype=tf.float32)
        predictions = model(input_tensor)
        return predictions.numpy()  # Convert predictions to numpy array if needed
    
def main():
        # Define model paths with labels
    model_paths_with_labels = [
        ('SNV + br (R49)', 'snv_baseline_removed_pls_top_10_float32.parquet_best_model_2024-03-31_13-29-57'),
        ('TFLite', 'tflite_model_snv_br_10.tflite'),
        ('TFLite Q', 'tflite_model_snv_br_10_quant.tflite'),
        ('TFLite Q-weight', 'tflite_model_snv_br_10_quant_weight.tflite')
    ]
    
    
    # Get data from server (simulated here)
    df, wavelengths = csv_data()

    for label, model_path in model_paths_with_labels:

        # Load the model
        model = load_model(model_path)
        # st.write(model)
        
                # Now process each row in df
        for index, row in df.iterrows():
            predictions = predict_with_model(model, row)  # Assuming predict_with_model can handle a single row of DataFrame
            predictions_value = predictions[0][0]  # Assuming each prediction returns a single value

            # Display logic remains the same
            if predictions_value > 25:
                display_value = f'<span class="high-value">High value : ({predictions_value:.1f} g/dL)</span>'
            else:
                display_value = f'<span class="value">{predictions_value:.1f} g/dL</span>'

            st.markdown(f'<span class="label">Haemoglobin ({label}) - Sample {index+1}:</span><br>{display_value}</p>', unsafe_allow_html=True)
        # # Predict
        # predictions = predict_with_model(model, df)
        # predictions_value = predictions[0][0]
    
        # st.markdown("""
        # <style>
        # .label {font-size: 16px; font-weight: bold; color: black;}
        # .value {font-size: 60px; font-weight: bold; color: blue;}
        # .high-value {font-size: 60px; font-weight: bold; color: red;}
        # </style> """, unsafe_allow_html=True)
    
        # # Add condition for prediction value
        # if predictions_value > 25:
        #     display_value = f'<span class="high-value">High value : ({predictions_value:.1f} g/dL)</span>'
        # else:
        #     display_value = f'<span class="value">{predictions_value:.1f} g/dL</span>'
        
        # # Display label and prediction value
        # st.markdown(f'<span class="label">Haemoglobin ({label}):</span><br>{display_value}</p>', unsafe_allow_html=True)

    # # Plotting
    # plt.figure(figsize=(10, 4))
    # plt.plot(wavelengths, absorbance_data.iloc[0], marker='o', linestyle='-', color='b')
    # plt.xlabel('Wavelength (nm)', fontweight='bold', fontsize=14)
    # plt.ylabel('Absorbance', fontweight='bold', fontsize=14)
    # plt.xticks(rotation='vertical', fontweight='bold', fontsize=12)
    # plt.yticks(fontweight='bold', fontsize=12)
    # plt.tight_layout()
    # plt.show()
    # st.pyplot(plt)
    
if __name__ == "__main__":
    main()
