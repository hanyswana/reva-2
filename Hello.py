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
from datetime import datetime
import pytz

csv_file_path = 'golden_lablink_snv_norm_euc_baseline_each_batch.csv'
golden_df = pd.read_csv(csv_file_path)
golden_values = golden_df.iloc[0].values

range_csv_file_path = 'range_lablink_snv_norm_euc_baseline_each_batch.csv'
range_df = pd.read_csv(range_csv_file_path)

utc_now = datetime.now(pytz.utc)
singapore_time = utc_now.astimezone(pytz.timezone('Asia/Singapore'))
formatted_time = singapore_time.strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"Time: {formatted_time}")

# Custom Baseline Removal Transformer
class BaselineRemover(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!')
        return self

    def transform(self, X, copy=None):
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        X = self.remove_baseline(X.T).T
        return X

    def remove_baseline(self, spectra):
        return spectra - spectra.mean(axis=0)

    def _more_tags(self):
        return {'allow_nan': True}

def snv(input_data):
    # Mean centering and scaling by standard deviation for each spectrum
    mean_corrected = input_data - np.mean(input_data, axis=1, keepdims=True)
    snv_transformed = mean_corrected / np.std(mean_corrected, axis=1, keepdims=True)
    return snv_transformed
        
def json_data():
    # First API call
    api_url1 = "https://x8ki-letl-twmt.n7.xano.io/api:3Ws6ADLi/bgdata"
    payload1 = {}
    response1 = requests.get(api_url1, params=payload1)

    if response1.status_code == 200:
        data1 = response1.json()
    else:
        st.write("Error in first API call:", response1.status_code)
        return None

    # Second API call
    api_url2 = "https://x8ki-letl-twmt.n7.xano.io/api:Qc5crfn2/spectraldata"
    payload2 = {}
    response2 = requests.get(api_url2, params=payload2)

    if response2.status_code == 200:
        data2 = response2.json()
    else:
        st.write("Error in second API call:", response2.status_code)
        return None

    # Extract first line of data from both API responses and convert to numeric
    df1 = pd.DataFrame(data1).iloc[:1].apply(pd.to_numeric, errors='coerce')
    df2 = pd.DataFrame(data2).iloc[:1].apply(pd.to_numeric, errors='coerce')
    wavelengths = df1.columns
    absorbance_df = df2.div(df1.values).pow(0.5)
    # st.write(absorbance_df)

    # Apply SNV to the absorbance data after baseline removal
    absorbance_snv = snv(absorbance_df.values)
    absorbance_snv_df = pd.DataFrame(absorbance_snv, columns=absorbance_df.columns)
    
    # # Normalize the absorbance data using Euclidean normalization
    normalizer = Normalizer(norm='l2')  # Euclidean normalization
    absorbance_normalized_euc = normalizer.transform(absorbance_snv_df)
    absorbance_normalized_euc_df = pd.DataFrame(absorbance_normalized_euc, columns=absorbance_df.columns)

    # # Normalize the absorbance data using Manhattan normalization
    # normalizer = Normalizer(norm='l1')  # Manhattan normalization
    # absorbance_normalized_manh = normalizer.transform(absorbance_df)
    # absorbance_normalized_manh_df = pd.DataFrame(absorbance_normalized_manh, columns=absorbance_df.columns)

    # Apply baseline removal to the absorbance data
    baseline_remover = BaselineRemover()
    absorbance_baseline_removed = baseline_remover.transform(absorbance_normalized_euc_df)
    absorbance_baseline_removed_df = pd.DataFrame(absorbance_baseline_removed, columns=absorbance_df.columns)
    
    absorbance_snv_normalized_euc_baseline_removed_df = absorbance_baseline_removed_df

    # First row of absorbance data
    absorbance_data = absorbance_df.iloc[0]  
 
    return absorbance_df, absorbance_snv_df, absorbance_normalized_euc_df, absorbance_baseline_removed_df, absorbance_snv_normalized_euc_baseline_removed_df, wavelengths

def select_for_prediction(absorbance_df, selected_wavelengths):
    return absorbance_df[selected_wavelengths]
    
def load_model(model_dir):
    if model_dir.endswith('.tflite'):
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
        
        # Ensure input data is 2D: [batch_size, num_features]
        input_data = input_data.values.astype('float32')  # Convert DataFrame to numpy and ensure dtype
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)  # Reshape if single row input
        
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        return predictions  # This will be a numpy array
    else:
        # Assuming TensorFlow SavedModel prediction logic
        input_data = input_data.values.astype('float32').reshape(-1, 10)  # Adjust based on your model's expected input
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        predictions = model(input_tensor)
        return predictions.numpy()

def main():
    # Define model paths with labels
    model_paths_with_labels = [
        # ('SNV + BR (R47)', 'Lablink_134_SNV_Baseline_pls_top_10.parquet_best_model_2024-05-09_20-22-34_R45_77%')
        # ('SNV + BR (R56)', 'Lablink_134_SNV_Baseline_pls_top_10.parquet_best_model_2024-05-11_02-11-44_R56_81%')
        # ('SNV + BR (R50)', 'Lablink_134_SNV_Baseline_pls_top_10.parquet_best_model_2024-05-18_04-08-04_R50_78%')
        ('SNV + BR + norm euc (R52)', 'Lablink_134_SNV_norm_eucl_Baseline_pls_top_10.parquet_best_model_2024-05-24_05-21-44_R52_78%')
        
    ]
    Min = range_df.iloc[0, 1:].values
    Max = range_df.iloc[1, 1:].values

    absorbance_df, absorbance_snv_df, absorbance_normalized_euc_df, absorbance_baseline_removed_df, absorbance_snv_normalized_euc_baseline_removed_df, wavelengths = json_data()

    for label, model_path in model_paths_with_labels:

        selected_wavelengths = ['_415nm', '_445nm', '_515nm', '_555nm', '_560nm', '_610nm', '_680nm', '_730nm', '_900nm', '_940nm']
        prediction_data = select_for_prediction(absorbance_snv_normalized_euc_baseline_removed_df, selected_wavelengths)
        # st.write(prediction_data)
        
        model = load_model(model_path)
        
        # # Predict with original absorbance data
        # predictions_original = predict_with_model(model, absorbance_df)
        # predictions_value_original = predictions_original[0][0]

        predictions = predict_with_model(model, prediction_data)
        predictions_value = predictions[0][0]

        # Calculate correlation with 'golden' values
        correlation = np.corrcoef(absorbance_snv_normalized_euc_baseline_removed_df.iloc[0], golden_values)[0, 1]

        Min = np.array(Min, dtype=float)
        Max = np.array(Max, dtype=float)

        # Ensure absorbance_snv_baseline_removed_df values are numpy array
        absorbance_values = absorbance_snv_normalized_euc_baseline_removed_df.values

        out_of_range = (absorbance_values < Min) | (absorbance_values > Max)
        count_out_of_range = np.sum(out_of_range)
        total_values = absorbance_values.size
        in_range_percentage = 100 - ((count_out_of_range / total_values) * 100)

        st.markdown("""
        <style>
        .label {font-size: 20px; font-weight: bold; color: black;}
        .value {font-size: 40px; font-weight: bold; color: blue;}
        # .high-value {color: red;}
        </style> """, unsafe_allow_html=True)

        if predictions_value > 100:
            display_text = 'Above 100 g/dL'
        elif predictions_value < 0:
            display_text = 'Below 0 g/dL'
        else:
            display_text = f'{predictions_value:.1f} g/dL'
            
        # Format the display value with consistent styling
        display_value = f'<span class="value">{display_text}</span>'

        # # Display label and prediction value
        st.markdown(f'<span class="label">Haemoglobin:</span><br>{display_value}</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="label">Similarity to training data:</span><br><span class="value">{in_range_percentage:.0f} %</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="label">Correlation:</span><br><span class="value">{correlation:.2f}</span>', unsafe_allow_html=True)


    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, absorbance_snv_df.iloc[0], marker='o', linestyle='-', color='b', label='Pp sample (SNV)')
    plt.plot(wavelengths, absorbance_normalized_euc_df.iloc[0], marker='d', linestyle='-', color='r', label='Pp sample (SNV + norm euc)')
    plt.plot(wavelengths, absorbance_baseline_removed_df.iloc[0], marker='s', linestyle='-', color='g', label='Pp (SNV + norm euc + baseline)')
    plt.plot(wavelengths, absorbance_df.iloc[0], marker='o', linestyle='--', color='b', label='Raw sample')
    plt.plot(wavelengths, Min, linestyle='--', color='r', label='Min')
    plt.plot(wavelengths, Max, linestyle='--', color='y', label='Max')
    plt.title('Absorbance', fontweight='bold', fontsize=20)
    plt.xlabel('Wavelength (nm)', fontweight='bold', fontsize=14)
    plt.ylabel('Absorbance', fontweight='bold', fontsize=14)
    plt.xticks(rotation='vertical', fontweight='bold', fontsize=12)
    plt.yticks(fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.legend()
    plt.show()
    st.pyplot(plt)
    
if __name__ == "__main__":
    main()
