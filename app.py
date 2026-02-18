import streamlit as st
import requests

# Make the webpage wide and give it a title
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Telco Customer Churn Predictor")
st.write("Enter the customer's details below to ask the AI if they are at risk of canceling their service.")
st.markdown("---")

# Create 3 columns for a clean, dashboard-like layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.subheader("📝 Account Info")
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=2)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", value=70.7)
    total_charges = st.number_input("Total Charges ($)", value=151.65)

with col3:
    st.subheader("📶 Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("---")

# The Magic Button
if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    
    # 1. Package all the user's answers into a dictionary
    customer_data = {
        "Gender": gender, "Senior Citizen": senior, "Partner": partner,
        "Dependents": dependents, "Tenure Months": tenure, "Phone Service": phone,
        "Multiple Lines": multiple_lines, "Internet Service": internet,
        "Online Security": security, "Online Backup": backup, "Device Protection": protection,
        "Tech Support": support, "Streaming TV": tv, "Streaming Movies": movies,
        "Contract": contract, "Paperless Billing": paperless, "Payment Method": payment,
        "Monthly Charges": monthly_charges, "Total Charges": total_charges
    }

    # 2. Send the data to your FastAPI brain!
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=customer_data)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"] * 100
            
            # 3. Show the results on the screen!
            if result["churn_prediction"] == 1:
                st.error(f"🚨 **High Risk of Churn!** The AI calculates a **{prob:.1f}%** chance this customer will leave.")
            else:
                st.success(f"✅ **Customer is Safe.** The AI calculates only a **{prob:.1f}%** chance of them leaving.")
        else:
            st.error("Error from API. Check your FastAPI terminal!")
            
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to the API. Is your first terminal still running `uvicorn`?")