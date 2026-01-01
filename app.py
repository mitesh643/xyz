import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_model.pkl')


st.title('fraud detection predection app')
st.write('enter the inforamtion to predict the fraud or not')

with st.form("fraud_form"):
    type_ = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0, step=100.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=10000.0, step=100.0)
    newbalanceOrig = st.number_input("New Balance (Origin)",  min_value=0.0, value=5000.0, step=100.0)
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=2000.0, step=100.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=7000.0, step=100.0)

    submitted = st.form_submit_button(" Predict Fraud")

if submitted:
    # Create DataFrame from user input
    input_data = pd.DataFrame({
        "type": [type_],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
    })

    # ✅ Create the same derived features used in training
    input_data['difforig'] = input_data['oldbalanceOrg'] - input_data['newbalanceOrig']
    input_data['diffdest'] = input_data['newbalanceDest'] - input_data['oldbalanceDest']
    input_data['errorbal'] = input_data['newbalanceDest'] + input_data['amount'] - input_data['oldbalanceDest']

    # Optional check (can remove later)
    st.write("Final input columns:", list(input_data.columns))

    # ✅ Make prediction
    prob = model.predict_proba(input_data)[:, 1][0]
    pred = model.predict(input_data)[0]

    st.subheader("🔎 Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Fraud Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob:.2f})")




    # Optional display
    st.markdown("Model Input Summary")
    st.dataframe(input_data.style.format(precision=2))

    # Add probability bar
    st.markdown("Fraud Probability Level")
    st.progress(float(prob))


