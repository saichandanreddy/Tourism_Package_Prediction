import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
# Ensure the repo_id and filename match where your model was uploaded
model_path = hf_hub_download(repo_id="saichandanreddy/Tourism-Package-Prediction-Model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Download and load Xtrain_columns to ensure consistent feature order
xtrain_columns_path = hf_hub_download(repo_id="saichandanreddy/Tourism-Package-Prediction-Model", filename="xtrain_columns.joblib")
Xtrain_columns = joblib.load(xtrain_columns_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase the Wellness Tourism Package
based on their details and interaction data. Please enter the customer information below.
""")

# User Input Fields matching our dataset features
# Numerical features
age = st.slider("Age", min_value=18, max_value=80, value=35)
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=25000.0, step=100.0)
number_of_trips = st.slider("Number of Trips Annually", min_value=1, max_value=25, value=3)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
duration_of_pitch = st.slider("Duration of Pitch (minutes)", min_value=5, max_value=130, value=15)
number_of_person_visiting = st.slider("Number of Persons Visiting", min_value=1, max_value=5, value=2)
preferred_property_star = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
number_of_children_visiting = st.slider("Number of Children Visiting (<5 years old)", min_value=0, max_value=3, value=1)

# Binary features
passport = st.selectbox("Holds Valid Passport", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
own_car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')

# Categorical features
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Large Business'])
gender = st.selectbox("Gender", ['Male', 'Female'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
product_pitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'King', 'Standard', 'Super Deluxe'])
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
number_of_followups = st.slider("Number of Follow-ups", min_value=1, max_value=6, value=3)

# Assemble input into DataFrame for prediction
# Create a dictionary to hold raw input values
raw_input = {
    'Age': age,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'MonthlyIncome': monthly_income,
    'TypeofContact': type_of_contact,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'ProductPitched': product_pitched,
    'Designation': designation
}

# Convert raw input to a pandas DataFrame
input_df = pd.DataFrame([raw_input])

# Apply the same preprocessing as in train.py (gender correction and one-hot encoding)
input_df['Gender'] = input_df['Gender'].replace('Fe Male', 'Female') # Should handle if 'Fe Male' is ever entered

# Re-create dummy variables, ensuring all original categorical columns are covered
categorical_cols_input = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'Designation']
input_df = pd.get_dummies(input_df, columns=categorical_cols_input, drop_first=False)

# To ensure column alignment
final_input_df = input_df.reindex(columns=Xtrain_columns, fill_value=0)

if st.button("Predict Purchase"):
    prediction = model.predict(final_input_df)[0]
    probability = model.predict_proba(final_input_df)[:, 1][0]
    result = "Customer WILL purchase" if prediction == 1 else "Customer will NOT purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}** (Probability of Purchase: {probability:.2f})")
