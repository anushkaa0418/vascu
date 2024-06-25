import streamlit as st
import pandas as pd
import json
import joblib
import os
import base64
from datetime import datetime, date, timedelta
import numpy as np
import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
import warnings
warnings.filterwarnings("ignore")

# Function to calculate BMI
def calculate_bmi(weight, height):
    height_m = height / 100  # Convert height to meters
    bmi = weight / (height_m ** 2)
    return bmi

def set_bg_hack(main_bg):
    file_extension = os.path.splitext(main_bg)[-1].lower().replace(".", "")
    with open(main_bg, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{file_extension};base64,{base64_image});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load the category options
with open('categorical_column_val.json', 'r') as f:
    category_options = json.load(f)
    
with open('tips.json', 'r') as f:
    health_tips = json.load(f)

# Load the trained model (Assuming you have a trained model saved as 'model.pkl')
model = joblib.load('model_pipeline.joblib')
max_birth_date = datetime.now().date() - timedelta(days=18*365)
min_birth_date = datetime.now().date() - timedelta(days=100*365)
# Define the user input form
def user_input_features():
    birth_date = st.date_input("Birth Date", value=datetime(1990, 1, 1), min_value=min_birth_date, max_value=max_birth_date)
    gender = st.selectbox("Gender", options=category_options["gender"])
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=150, value=80)
    cholesterol_value = st.number_input("Cholesterol (mmol/L)", min_value=0.0, max_value=20.0, value=4.5)
    gluc_value = st.number_input("Glucose (mmol/L)", min_value=0.0, max_value=20.0, value=5.0)
    smoke = st.selectbox("Smoking", options=category_options["smoke"])
    alco = st.selectbox("Alcohol Intake", options=category_options["alco"])
    active = st.selectbox("Physical Activity", options=category_options["active"])

    # Convert cholesterol to categories
    if cholesterol_value < 5.2:
        cholesterol = "Normal"
    elif 5.2 <= cholesterol_value <= 6.2:
        cholesterol = "Above Normal"
    else:
        cholesterol = "Well Above Normal"

    # Convert glucose to categories
    if gluc_value < 5.6:
        gluc = "Normal"
    elif 5.6 <= gluc_value <= 7.0:
        gluc = "Above Normal"
    else:
        gluc = "Well Above Normal"

    data = {
        'age': (datetime.now().date() - birth_date).days,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }

    features = pd.DataFrame(data, index=[0])
    return features, height, weight

__login__obj = __login__(auth_token = "dk_prod_XHG9DC6V4EMCB2J8X6GJA01AFJMS", 
                    company_name = "Shims",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:
    # set_bg_hack("heart-disease.jpg")
    st.title("Cardio Disease Prediction App")
    st.image("heart-disease.jpg", use_column_width=True)
    # Get user input
    input_df, height, weight = user_input_features()

    # Predict button
    if st.button("Predict"):
        bmi = calculate_bmi(weight, height)
        st.write(f"Your BMI is: {bmi:.2f}")
        # Preprocess input and make predictions
        prediction_proba = int(round(np.max(model.predict_proba(input_df)[0]),2) * 100)
        prediction = model.predict(input_df)[0]
        # print(prediction,prediction_proba)
        if prediction == 1:
            st.error(f"Prediction: Yes (High Risk)")
            st.error(f"Probability: {prediction_proba}%")
        else:
            if 50 <= prediction_proba <= 80:
                st.success(f"Prediction: No (Low Risk)")
            else:
                st.success(f"Prediction: No (No Risk)")
            st.success(f"Probability: {prediction_proba}%")
        for idx,tip in enumerate(health_tips[str(prediction)]):
            with st.expander(f"Health tip - {idx+1}"):
                st.write(tip)