import streamlit as st
import joblib
import pandas as pd
import os
st.set_page_config(page_title="Credit Risk Analyzer")
st.title("Loan Credit Risk Analyzer")
st.write("Determine loan eligibility based on Income and Loan Amount.")

model_path = os.path.join('models','credit_model.pkl')

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found! Please run train.py first.")
    st.stop()

st.sidebar.header("Applicant Details")
income = st.sidebar.number_input("Annual Income (₹)", min_value=5000, value=50000, step=1000)
loan_amount = st.sidebar.number_input("Requested Loan Amount (₹)", min_value=1000,value=15000,step=500)

if st.button("Check Eligibility"):
    input_df = pd.DataFrame([[income,loan_amount]], columns=['income', 'loan_amount'])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction ==  0:
        st.success("Result: SAFE")
        st.write(f"The model suggests this applicant is likely to repay. (Risk Probability: {probability:.2%})")
    else :
        st.error("Result: RISKY")
        st.write(f"The model suggests a high risk of default. (Risk Probability: {probability:.2%})")
    with st.expander("How does this work?"):
        st.write(
            "This uses a **Logistic Regression** model. Unlike linear regression which predicts a price, this uses a Sigmoid function to calculate the probability of a 'Yes/No' outcome."
        )