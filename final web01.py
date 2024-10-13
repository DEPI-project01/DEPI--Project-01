import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import joblib

# Page Configuration
st.set_page_config(
    page_title="Bank Customer Churn Dashboard",
    layout="wide",  # This makes the layout responsive on mobile devices
)

# Sidebar for Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Our Data", "Churn Dashboard", "Prediction"])

# Page 1: Our Data
if selection == "Our Data":
    # Set the title of the app
    st.title("🏦 Bank Customer Churn Prediction")
    
    # Display a team photo (make sure to replace 'team_photo.png' with your image file)
    st.image("team_photo.png", caption="Our Amazing Team", use_column_width=True)
    
    # Team members with emojis
    st.write("### Our Team: 👩‍💻👨‍💻")
    team_members = [
        "Donya Samir 🎉",
        "Esraa Mosaad 🎉",
        "Mostafa Omar 🎉",
        "Nour Ehab 🎉",
        "Talal Ibrahim 🎉"
    ]
    for member in team_members:
        st.write(f"- {member}")
    
    # Problem section
    st.subheader("🚩 Problem:")
    st.write("High churn rates lead to loss of customers for banks. 😟")
    
    # Objective section
    st.subheader("🎯 Objective:")
    st.write("Predict bank customer churn and build interactive dashboards. 📊")

# Page 2: Churn Dashboard
elif selection == "Churn Dashboard":
    st.title("Enhanced Customer Churn Dashboard")
    data = pd.read_csv('Bank Customer Churn Prediction.csv')

    # Sidebar filters
    st.sidebar.title("Filter Options")
    selected_country = st.sidebar.multiselect("Select Country", options=data['country'].unique(), default=data['country'].unique())
    selected_gender = st.sidebar.multiselect("Select Gender", options=data['gender'].unique(), default=data['gender'].unique())
    selected_churn = st.sidebar.radio("Churn Status", options=[0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
    min_age, max_age = int(data['age'].min()), int(data['age'].max())
    age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
    min_credit, max_credit = int(data['credit_score'].min()), int(data['credit_score'].max())
    credit_score_range = st.sidebar.slider("Select Credit Score Range", min_credit, max_credit, (min_credit, max_credit))
    min_balance, max_balance = float(data['balance'].min()), float(data['balance'].max())
    balance_range = st.sidebar.slider("Select Balance Range", min_balance, max_balance, (min_balance, max_balance))
    min_tenure, max_tenure = int(data['tenure'].min()), int(data['tenure'].max())
    tenure_range = st.sidebar.slider("Select Tenure Range", min_tenure, max_tenure, (min_tenure, max_tenure))

    # Filter data
    filtered_data = data[(data['country'].isin(selected_country)) &
                         (data['gender'].isin(selected_gender)) &
                         (data['churn'] == selected_churn) &
                         (data['age'].between(age_range[0], age_range[1])) &
                         (data['credit_score'].between(credit_score_range[0], credit_score_range[1])) &
                         (data['balance'].between(balance_range[0], balance_range[1])) &
                         (data['tenure'].between(tenure_range[0], tenure_range[1]))]

    # Overview Section
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    total_customers = filtered_data.shape[0]
    churn_rate = filtered_data['churn'].mean() * 100
    average_credit_score = filtered_data['credit_score'].mean()
    average_salary = filtered_data['estimated_salary'].mean()
    col1.metric(label="Total Filtered Customers", value=f"{total_customers}")
    col2.metric(label="Churn Rate", value=f"{churn_rate:.2f}%")
    col3.metric(label="Average Credit Score", value=f"{average_credit_score:.2f}")
    col4.metric(label="Average Estimated Salary", value=f"${average_salary:,.2f}")

    # Responsive Gender Distribution Pie Chart
    st.subheader("Gender Distribution")
    gender_counts = filtered_data['gender'].value_counts()
    col5, col6 = st.columns(2)
    with col5:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
        ax1.axis('equal')
        st.pyplot(fig1)

    # Churn by Country Bar Chart
    st.subheader("Churn by Country")
    country_churn = filtered_data.groupby('country')['churn'].mean() * 100
    st.bar_chart(country_churn)

    # Additional Charts (Churn by Age Group, Tenure, etc.)
    # ... (add more charts from your original code here)

# Page 3: Prediction
elif selection == "Prediction":
    st.title("Bank Customer Churn Prediction")
    model = load_model('best_models/NN_with_smote.h5')
    label_encoder = joblib.load('best_models/label_encoder.pkl')
    scaler = joblib.load('best_models/scaler.pkl')
    countries = ['France', 'Germany', 'Spain']

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox("Country", countries)
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    with col2:
        products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
        credit_card = st.selectbox("Has Credit Card?", [0, 1])
        active_member = st.selectbox("Active Member?", [0, 1])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    # Prepare input for the model
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

    # Preprocess and predict
    if st.button("Predict Churn"):
        def preprocess_input(input_data):
            input_data['gender'] = label_encoder.transform([input_data['gender']])[0]
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
            for country_name in countries:
                df_input[country_name] = 1 if input_data['country'] == country_name else 0
            scaled_values = scaler.transform(df_input[['credit_score', 'age', 'balance', 'estimated_salary']])
            df_input[['credit_score', 'age', 'balance', 'estimated_salary']] = scaled_values
            return df_input.astype(np.float32)

        preprocessed_input = preprocess_input(input_data)
        prediction = model.predict(preprocessed_input)
        churn_prediction = "Yes" if prediction[0] >= 0.5 else "No"
        st.subheader(f"Churn Prediction: {churn_prediction}")
