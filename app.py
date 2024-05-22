import streamlit as st
import joblib
import pandas as pd

# Function to predict loan status
def predict_loan_status(model, scaler, gender_val, married_val, dependents, education_val, self_employed_val, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history_val, property_area_val):
    # Create a DataFrame with user input
    df = pd.DataFrame({
        'Gender': [gender_val],
        'Married': [married_val],
        'Dependents': [dependents],
        'Education': [education_val],
        'Self_Employed': [self_employed_val],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history_val],
        'Property_Area': [property_area_val]
    })

    # Scale the input data
    df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']] = scaler.transform(df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])

    # Predict loan status
    result = model.predict(df)

    if result == 1:
        return "Loan Approved"
    else:
        return "Loan Not Approved"

# Load the saved model and scaler
model = joblib.load('loan_status_classification_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app layout
st.set_page_config(page_title="Loan Status Classification", layout="centered")

# Main title and header
st.title("Loan Status Classification App")
st.markdown("Classify whether your loan application will be approved or not!")

# Display Model Comparison Plot
st.markdown(" #### How the app works?📊")
st.image("flowchart.png",use_column_width=True)

# Display Model Comparison Plot
st.markdown("#### Model Performance Comparison")
st.image("performance_comparison.png",use_column_width=True)


# Sidebar for user input
st.sidebar.title("Enter Loan Details💡")

st.sidebar.markdown("### Enter your details below and click on 'Classify' to see the result at the bottom 👇.")

# User input fields
gender = st.sidebar.radio("Gender", ["Male", "Female"], help="Enter your gender here!")
married = st.sidebar.radio("Married", ["Yes", "No"], help="Enter you're married or not!")
dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3], help="Select the number of people you live with")
education = st.sidebar.radio("Education", ["Graduate", "Not Graduate"], help="Select your education!")
self_employed = st.sidebar.radio("Self Employed", ["Yes", "No"], help="Select yes if you're self-employed! Otherwise select no!")
applicant_income = st.sidebar.number_input("Applicant Income", help="Enter your annual income")
coapplicant_income = st.sidebar.number_input("Coapplicant Income", help="Enter your coapplicant income her")
loan_amount = st.sidebar.number_input("Loan Amount", help="Enter the loan amount you want!")
loan_amount_term = st.sidebar.number_input("Loan Amount Term", help="Enter the loan term amount you want!")
credit_history = st.sidebar.radio("Credit History", ["Good", "Bad"], help="Enter your credit history you have!")
property_area = st.sidebar.radio("Property Area", ["Rural", "Semiurban", "Urban"], help="Enter the area you live in!")

# Convert user input to numerical values
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}
property_area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
credit_history_map = {"Good": 1, "Bad": 0}

st.markdown(" ### Your Loan Status Result📄")

# Predict loan status on button click
if st.sidebar.button("Classify🔎", help="Hit Classify to know your loan status"):
    gender_val = gender_map[gender]
    married_val = married_map[married]
    education_val = education_map[education]
    self_employed_val = self_employed_map[self_employed]
    property_area_val = property_area_map[property_area]
    credit_history_val = credit_history_map[credit_history]

    result = predict_loan_status(model, scaler, gender_val, married_val, dependents, education_val, self_employed_val, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history_val, property_area_val)
    
    
    st.write(result)