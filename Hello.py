import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import requests, pytz, pickle, joblib, torch, os, json, tempfile, zipfile, onnxruntime as ort
import scipy.io as sio
import xml.etree.ElementTree as ET
from scipy import sparse, io
from datetime import datetime
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from pytorch_tabnet.tab_model import TabNetRegressor 


utc_now = datetime.now(pytz.utc)
singapore_time = utc_now.astimezone(pytz.timezone('Asia/Singapore'))
formatted_time = singapore_time.strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"Time: {formatted_time}")


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
    mean_corrected = input_data - np.mean(input_data, axis=1, keepdims=True)
    snv_transformed = mean_corrected / np.std(mean_corrected, axis=1, keepdims=True)
    return snv_transformed
    

def pds_transform(input_data, pds_model_path):
    if pds_model_path.endswith('.xml'):
        tree = ET.parse(pds_model_path)
        root = tree.getroot()
        stdmat_elements = root.find(".//stdmat").text.strip().split(';')
        stdvect_elements = root.find(".//stdvect").text.strip().split(',')
        stdmat = np.array([list(map(float, row.split(','))) for row in stdmat_elements if row])
        stdvect = np.array(list(map(float, stdvect_elements)))
        transformed_data = np.dot(input_data, stdmat) + stdvect
        return transformed_data
    elif pds_model_path.endswith('.joblib'):
        pds_model = joblib.load(pds_model_path)
        F, a = pds_model
        transformed_data = input_data.dot(F) + a
        return transformed_data
    else:
        raise ValueError("Unsupported model format")


def custom_transform(input_data, pds_models):
    transformed_data = np.zeros_like(input_data)

    for start, end, model in pds_models:
        slave_segment = input_data[:, start:end]
        transformed_segment = model.predict(slave_segment)
        transformed_data[:, start:end] = transformed_segment

    return transformed_data


def json_data():
    # API --------------------------------------------------------------------------------------------------------------------
    # # First API call
    # api_url1 = "https://x8ki-letl-twmt.n7.xano.io/api:5r4pCOor/bgdata_hb"
    # payload1 = {}
    # response1 = requests.get(api_url1, params=payload1)

    # if response1.status_code == 200:
    #     data1 = response1.json()
    # else:
    #     st.write("Error in first API call:", response1.status_code)
    #     return None

    # # Second API call
    # api_url2 = "https://x8ki-letl-twmt.n7.xano.io/api:UpqVw9TY/spectraldata_hb"
    # payload2 = {}
    # response2 = requests.get(api_url2, params=payload2)

    # if response2.status_code == 200:
    #     data2 = response2.json()
    # else:
    #     st.write("Error in second API call:", response2.status_code)
    #     return None

    # # Extract first line of data from both API responses and convert to numeric
    # df1 = pd.DataFrame(data1).iloc[:1].apply(pd.to_numeric, errors='coerce')
    # df2 = pd.DataFrame(data2).iloc[:1].apply(pd.to_numeric, errors='coerce')
    # wavelengths = df1.columns
    # absorbance_df = df2.div(df1.values).pow(0.5)
    # # st.write('19 raw data :')
    # # st.write(absorbance_df)


    # CSV ------------------------------------------------------------------------------------------------------------------
    # # file_path = 'correct-data/Test_raw_sample1.csv'
    # file_path = 'correct-data/test_no_ble_alpha_10.csv'
    file_path = 'lablink-data-2024/REVA LABLINK 2024_125.csv'
    # df = pd.read_csv(file_path, usecols=range(0, 19))
    df = pd.read_csv(file_path, usecols=range(3, 22))
    sample_ids = pd.read_csv(file_path, usecols=[0])['Sample ID']
    wavelengths = df.columns
    absorbance_df = df.apply(pd.to_numeric, errors='coerce')
    # absorbance_data = df.iloc[13]
    st.write('19 raw data :')
    st.write(absorbance_df)


    # CALIBRATION TRANSFER ------------------------------------------------------------------------------------------------------------------
    # PDS transformation
    
    # # pycharm ---------------------
    # # pds_model_path = 'calibration-transfer-model/py_U6_ori_pds_model.joblib'

    # # solo ------------------------
    # pds_model_path = 'calibration-transfer-model/pds-model-u6.xml'
    
    # absorbance_transformed = pds_transform(absorbance_df.values, pds_model_path)
    # absorbance_transformed_df = pd.DataFrame(absorbance_transformed, columns=absorbance_df.columns)
    # absorbance_df = absorbance_transformed_df
    # st.write('19 raw data after calibration transfer:')
    # st.write(absorbance_transformed_df)
    


    # PREPROCESS ------------------------------------------------------------------------------------------------------------------
    # 1. SNV
    absorbance_snv = snv(absorbance_df.values)
    absorbance_snv_df = pd.DataFrame(absorbance_snv, columns=absorbance_df.columns)
    
    # # 2. Euclidean normalization
    # normalizer = Normalizer(norm='l2')  # Euclidean normalization
    # absorbance_normalized_euc = normalizer.transform(absorbance_snv_df)
    # absorbance_normalized_euc_df = pd.DataFrame(absorbance_normalized_euc, columns=absorbance_df.columns)

    # # 3. Manhattan normalization
    # normalizer = Normalizer(norm='l1')  # Manhattan normalization
    # absorbance_normalized_manh = normalizer.transform(absorbance_snv_df)
    # absorbance_normalized_manh_df = pd.DataFrame(absorbance_normalized_manh, columns=absorbance_df.columns)

    # # 4. Baseline removal
    # baseline_remover = BaselineRemover()
    # absorbance_baseline_removed = baseline_remover.transform(absorbance_normalized_euc_df)
    # absorbance_baseline_removed_df = pd.DataFrame(absorbance_baseline_removed, columns=absorbance_df.columns)

    # pds_model = joblib.load('pds_model_U11_snv_baseline.joblib')
    # with open('pds_model_U6_snv_baseline.pkl', 'rb') as f:
    #     pds_model = pickle.load(f)

    # absorbance_transformed = pds_transform(absorbance_baseline_removed_df.values, pds_model)
    # absorbance_transformed_df = pd.DataFrame(absorbance_transformed, columns=absorbance_df.columns)
    # absorbance_all_pp_df = absorbance_transformed_df

    absorbance_all_pp_df = absorbance_snv_df
    
    st.write('19 preprocessed data :')
    st.write(absorbance_all_pp_df)

    reference_file_path = 'correct-data/corrected-lablink-128-hb_SNV.csv'
    # reference_file_path = 'correct-data/corrected-lablink-128-hb_SNV_Baseline.csv'
    # reference_file_path = 'correct-data/corrected-lablink-128-hb_SNV_norm_manh_Baseline.csv'
    # reference_file_path = 'correct-data/corrected-lablink-128-hb_SNV_norm_eucl_Baseline.csv'
    reference_df = pd.read_csv(reference_file_path, usecols=range(3, 22))
    reference_df = reference_df.apply(pd.to_numeric, errors='coerce')
    
    golden_values = reference_df.mean().values
    Min = reference_df.min().values
    Max = reference_df.max().values
 
    return absorbance_df, absorbance_all_pp_df, wavelengths, golden_values, Min, Max, sample_ids
    

def create_csv(golden_values, Min, Max, wavelengths):
    data = {
        'Wavelength': wavelengths,
        'Golden Values': golden_values,
        'Min': Min,
        'Max': Max
    }
    df = pd.DataFrame(data).T
    df.to_csv('golden_values_min_max.csv', index=False)
    # st.write(df)
    

def select_for_prediction(absorbance_df, selected_wavelengths):
    return absorbance_df[selected_wavelengths]


# TF/TFLITE MODEL ------------------------------------------------------------------------------------------------------------------

# def load_model(model_dir):
#     if model_dir.endswith('.tflite'):
#         interpreter = tf.lite.Interpreter(model_path=model_dir)
#         interpreter.allocate_tensors()
#         return interpreter
#     else:
#         model = tf.saved_model.load(model_dir)
#         return model


# def predict_with_model(model, input_data):
#     if isinstance(model, tf.lite.Interpreter):
#         input_details = model.get_input_details()
#         output_details = model.get_output_details()
        
#         # Ensure input data is 2D: [batch_size, num_features]
#         input_data = input_data.values.astype('float32')
#         if input_data.ndim == 1:
#             input_data = input_data.reshape(1, -1)  # Reshape if single row input
        
#         model.set_tensor(input_details[0]['index'], input_data)
#         model.invoke()
#         predictions = model.get_tensor(output_details[0]['index'])
#         return predictions
#     else:
#         input_data = input_data.values.astype('float32').reshape(-1, 10)
#         input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
#         predictions = model(input_tensor)
#         return predictions.numpy()


# TABNET MODEL ------------------------------------------------------------------------------------------------------------------

# def load_tabnet_model(model_path):
#     model = TabNetRegressor()
#     model.load_model(model_path)
#     return model


# def predict_with_tabnet_model(model, input_data):
#     input_data = torch.tensor(input_data.values, dtype=torch.float32)  # Convert DataFrame to NumPy array and then to tensor
#     with torch.no_grad():
#         predictions = model.predict(input_data)
#     return predictions



# TF/TFLITE/TABNET MODEL -----------------------------------------------------------------------------------------------------------------


def load_model(model_dir):
    if model_dir.endswith('.tflite'):
        interpreter = tf.lite.Interpreter(model_path=model_dir)
        interpreter.allocate_tensors()
        return interpreter
    elif model_dir.endswith('.pt.zip') or model_dir.endswith('.pt'):
        model = TabNetRegressor()
        model.load_model(model_dir)
        return model
    elif model_dir.endswith('.onnx'):
        session = ort.InferenceSession(model_dir)
        return session
    else:
        model = tf.saved_model.load(model_dir)
        return model


def predict_with_model(model, input_data):
    if isinstance(model, tf.lite.Interpreter):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        input_data = input_data.values.astype('float32')
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        return predictions
    elif isinstance(model, TabNetRegressor):
        input_data = torch.tensor(input_data.values, dtype=torch.float32)
        with torch.no_grad():
            predictions = model.predict(input_data)
        return predictions
    elif isinstance(model, ort.InferenceSession):
        input_name = model.get_inputs()[0].name
        input_data = input_data.values.astype('float32')
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        predictions = model.run(None, {input_name: input_data})[0]
        return predictions
    else:
        input_data = input_data.values.astype('float32').reshape(-1, 10)
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        predictions = model(input_tensor)
        return predictions.numpy()

        

def main():

    model_paths_with_labels = [
        # ('SNV + BR (tf-R45)', 'incorrect-model-lablink/Lablink_134_SNV_Baseline_pls_top_10.parquet_best_model_2024-05-09_20-22-34_R45_77%'),
        # ('SNV + BR (tf)', 'tabnet-model/model_snv_br_2024-06-06_14-42-37')
        # ('SNV + norm euc + BR (tf-R53)', 'corrected-model-lablink-128/corrected-lablink-128-hb_SNV_norm_eucl_Baseline_top_10.parquet_best_model_2024-07-09_22-18-50_R53_88%') # correct dataset
        ('SNV (tf-R59)', 'corrected-model-lablink-128/corrected-lablink-128-hb_SNV_top_10.parquet_best_model_2024-07-12_14-12-24_R59_88%') # correct dataset
    ]
    
    absorbance_df, absorbance_all_pp_df, wavelengths, golden_values, Min, Max, sample_ids = json_data()

    create_csv(golden_values, Min, Max, wavelengths)
    
    for label, model_path in model_paths_with_labels:

        # selected_wavelengths = ['_415nm', '_515nm', '_555nm', '_560nm', '_585nm', '_590nm',  '_610nm', '_680nm', '_730nm', '_900nm'] # for API (SNV) - latest new
        # selected_wavelengths = ['_415nm', '_445nm', '_515nm', '_555nm', '_560nm', '_610nm', '_680nm', '_730nm', '_900nm', '_940nm'] # for API (SNV + BR) - new
        # selected_wavelengths = ['_415nm', '_515nm', '_555nm', '_560nm', '_585nm', '_590nm', '_610nm', '_680nm', '_730nm', '_900nm'] # for API (SNV + euc + BR) - new
        # selected_wavelengths = ['_445nm', '_515nm', '_555nm', '_560nm', '_585nm', '_610nm', '_680nm', '_730nm', '_900nm', '_940nm'] # for API (SNV + manh + BR) - new
        
        selected_wavelengths = ['Spec-1', 'Spec-4', 'Spec-5', 'Spec-6', 'Spec-7', 'Spec-8', 'Spec-9', 'Spec-12', 'Spec-14', 'Spec-18'] # for CSV (SNV / SNV + euc + BR) - new lablink 2024
        
        # selected_wavelengths = ['415 nm', '445 nm', '515 nm', '555 nm', '560 nm', '610 nm', '680 nm', '730 nm', '900 nm', '940 nm'] # for CSV (SNV + BR) - new
        # selected_wavelengths = ['415 nm', '515 nm', '555 nm', '560 nm', '585 nm', '590 nm', '610 nm', '680 nm', '730 nm', '900 nm'] # for CSV (SNV + euc + BR) - new
        
        prediction_data = select_for_prediction(absorbance_all_pp_df, selected_wavelengths)
        st.write('10 selected preprocessed data :')
        st.write(prediction_data)

        # TF/TFLITE &&& TF/TFLITE/TABNET MODEL ---------------------------------------------------------------------------------
        model = load_model(model_path)
        predictions = predict_with_model(model, prediction_data)

        # TABNET MODEL ---------------------------------------------------------------------------------
        # model = load_tabnet_model(model_path)
        # predictions = predict_with_tabnet_model(model, prediction_data)
        
        predictions_value = predictions[0][0]
        
        correlation = np.corrcoef(absorbance_all_pp_df.iloc[0], golden_values)[0, 1]

        Min = np.array(Min, dtype=float)
        Max = np.array(Max, dtype=float)
        absorbance_values = (absorbance_all_pp_df.values)

        out_of_range = (absorbance_values < Min) | (absorbance_values > Max)
        count_out_of_range = np.sum(out_of_range)
        total_values = absorbance_values.size
        in_range_percentage = 100 - ((count_out_of_range / total_values) * 100)


        # PREDICT ALL ROWS IN CSV ----------------------------------------------------------------------
        rounded_predictions = np.round(predictions, 1)
        st.write(f'Predictions for {label}:')
        for i, prediction in enumerate(rounded_predictions):
            sample_id = sample_ids.iloc[i]

            # Calculate in-range percentage as similarity score
            absorbance_values = absorbance_all_pp_df.iloc[i].values
            out_of_range = (absorbance_values < Min) | (absorbance_values > Max)
            count_out_of_range = np.sum(out_of_range)
            total_values = absorbance_values.size
            in_range_percentage = 100 - ((count_out_of_range / total_values) * 100)
            in_range_percentage = round(in_range_percentage, 0)  # Round to 0 decimal places

            st.write(f'Sample {sample_id}: {prediction[0]} g/dL, Similarity Score: {in_range_percentage}%')

        
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
            
        display_value = f'<span class="value">{display_text}</span>'

        # st.markdown(f'<span class="label">Haemoglobin ({label}):</span><br>{display_value}</p>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Similarity score ({label}):</span><br><span class="value">{in_range_percentage:.0f} %</span>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Haemoglobin:</span><br>{display_value}</p>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Similarity score:</span><br><span class="value">{in_range_percentage:.0f} %</span>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Correlation:</span><br><span class="value">{correlation:.2f}</span>', unsafe_allow_html=True)

    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, absorbance_all_pp_df.iloc[0], marker='o', linestyle='-', color='b', label='Sample')
    # plt.plot(wavelengths, absorbance_df.iloc[0], marker='o', linestyle='--', color='b', label='Raw sample')
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
