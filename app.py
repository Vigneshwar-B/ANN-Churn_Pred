import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import seaborn as sns

# Load the trained model with compile=False to avoid optimizer-related errors
model = tf.keras.models.load_model('model.h5', compile=False)

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title and icon
st.set_page_config(page_title='Customer Churn Prediction', page_icon=':chart_with_upwards_trend:')
st.title('Customer Churn Prediction')
st.subheader('Predict the likelihood of customer churn based on various factors.')

# Create a sidebar for user inputs
st.sidebar.header('User Input Parameters')
geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92, value=30)
balance = st.sidebar.number_input('Balance', value=0.0)
credit_score = st.sidebar.number_input('Credit Score', value=600)
estimated_salary = st.sidebar.number_input('Estimated Salary', value=50000.0)
tenure = st.sidebar.slider('Tenure (Years)', 0, 10)
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the 'Geography' input
geo_encoded = onehot_encoder_geo.transform([[geography]])
if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Predict churn probability
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display results in the main area
st.markdown("---")
st.header('Prediction Results')
st.write(f'**Churn Probability:** {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.error('The customer is likely to churn.')
else:
    st.success('The customer is not likely to churn.')

# Add some styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f5;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<footer style='text-align: center;'>Built with ❤️ by Vigneshwar B.</footer>", unsafe_allow_html=True)
