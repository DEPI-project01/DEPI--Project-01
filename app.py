import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the trained model
model = load_model('best_models/NN_with_smote.h5')

# Load the preprocessing components
label_encoder = joblib.load('best_models/label_encoder.pkl')
scaler = joblib.load('best_models/scaler.pkl')

# Define the countries for one-hot encoding
countries = ['France', 'Germany', 'Spain']

# Set the title of the app
st.title("Bank Customer Churn Prediction")
st.markdown("Welcome to the customer churn prediction tool. Enter the details below to check if a customer is likely to churn.")

# Organize the layout with two columns
col1, col2 = st.columns(2)

# Collect user input in the first column
with col1:
    st.header("Customer Details")
    country = st.selectbox("Country", countries)  # Added country selection
    gender = st.selectbox("Gender", ["Male", "Female"])  # Added gender selection
    age = st.number_input("Age", min_value=18, max_value=100, value=30)  # Age input
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)

# Continue collecting more data in the second column
with col2:
    st.header("Account Details")
    products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
    credit_card = st.selectbox("Has Credit Card?", [0, 1])
    active_member = st.selectbox("Active Member?", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Prepare input data for the model
input_data = {
    "credit_score": credit_score,
    "age": age,
    "balance": balance,
    "estimated_salary": estimated_salary,
    "gender": gender,
    "country": country,
    "products_number": products_number,
    "tenure": tenure,
    "credit_card": credit_card,
    "active_member": active_member,
}


# Preprocessing input data
def preprocess_input(input_data):
    # Label encode gender
    input_data['gender'] = label_encoder.transform([input_data['gender']])[0]

    # Create a DataFrame for processing
    df_input = pd.DataFrame({
        'credit_score': [input_data['credit_score']],
        'gender': [input_data['gender']],
        'age': [input_data['age']],
        'tenure': [input_data['tenure']],
        'balance': [input_data['balance']],
        'products_number': [input_data['products_number']],
        'credit_card': [input_data['credit_card']],
        'active_member': [input_data['active_member']],
        'estimated_salary': [input_data['estimated_salary']],
    })

    # One-hot encode the country columns
    for country_name in countries:
        df_input[country_name] = 1 if input_data['country'] == country_name else 0

    # Scale the necessary columns
    scaled_values = scaler.transform(df_input[['credit_score', 'age', 'balance', 'estimated_salary']])
    df_input[['credit_score', 'age', 'balance', 'estimated_salary']] = scaled_values

    # Convert to float32
    return df_input.astype(np.float32)

# Prediction button
if st.button("Predict Churn"):
    preprocessed_input = preprocess_input(input_data)
    #print(preprocessed_input)
    #print("preprocessed_input", preprocessed_input.shape)
    prediction = model.predict(preprocessed_input)
    #print("prediction1", prediction)
    churn_prediction = "Yes" if prediction[0] >= 0.5 else "No"
    st.subheader(f"Churn Prediction: {churn_prediction}")

# Footer
st.markdown("---")
st.write("This app helps predict whether a customer is likely to churn based on various attributes. Simply fill in the details and hit 'Predict Churn'.")
