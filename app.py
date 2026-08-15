import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessing = joblib.load("preprocessing.pkl")
    model_artifact = joblib.load("customer_churn_model.pkl")

    # Recommended artifact format:
    # {
    #     "model": best_model,
    #     "total_charges_median": median_value
    # }
    if isinstance(model_artifact, dict):
        model = model_artifact["model"]
        total_charges_median = model_artifact.get(
            "total_charges_median",
            None
        )
    else:
        # Backward-compatible with saving best_model directly.
        model = model_artifact
        total_charges_median = None

    return preprocessing, model, total_charges_median


preprocessing, model, total_charges_median = load_artifacts()


# ---------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------
def predict_customer(input_data):
    input_df = pd.DataFrame([input_data])

    # Match the training-time treatment:
    # total_charges == 0 was replaced by the training median.
    if (
        total_charges_median is not None
        and input_df.loc[0, "total_charges"] == 0
    ):
        input_df.loc[0, "total_charges"] = total_charges_median

    # Apply exactly the same fitted preprocessing used in training.
    X_processed = preprocessing.transform(input_df)

    churn_probability = model.predict_proba(
        X_processed
    )[0, 1]

    prediction = int(churn_probability >= 0.50)

    return prediction, churn_probability


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Predict whether a telecom customer is likely to **churn**
    based on demographic, service, tenure and billing information.
    """
)

st.info(
    "Model: Gradient Boosting Classifier | "
    "Preprocessing: StandardScaler + OneHotEncoder"
)


# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------
st.sidebar.header("Customer Information")

# Demographics
st.sidebar.subheader("Demographics")

latitude = st.sidebar.number_input(
    "Latitude",
    value=36.000000,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=-118.000000,
    format="%.6f"
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

# senior_citizen is numeric (0/1) in the training data.
senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No", "Yes"]
)

# Account
st.sidebar.subheader("Account")

tenure_months = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12,
    step=1
)

# Services
st.sidebar.subheader("Services")

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No Phone Service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber Optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes"]
)

# Billing
st.sidebar.subheader("Billing")

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Mailed_Check",
        "Electronic_Check",
        "Bank_Transfer",
        "Credit_Card"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0,
    step=1.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0,
    step=10.0
)

# Model features
st.sidebar.subheader("Customer Value")

churn_score = st.sidebar.number_input(
    "Churn Score",
    min_value=0.0,
    value=50.0,
    step=1.0
)

cltv = st.sidebar.number_input(
    "CLTV",
    min_value=0.0,
    value=4500.0,
    step=100.0
)


# ---------------------------------------------------------
# Build input record
# ---------------------------------------------------------
input_data = {
    "latitude": latitude,
    "longitude": longitude,
    "gender": gender,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "dependents": dependents,
    "tenure_months": tenure_months,
    "phone_service": phone_service,
    "multiple_lines": multiple_lines,
    "internet_service": internet_service,
    "online_security": online_security,
    "online_backup": online_backup,
    "device_protection": device_protection,
    "tech_support": tech_support,
    "streaming_tv": streaming_tv,
    "streaming_movies": streaming_movies,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "churn_score": churn_score,
    "cltv": cltv
}


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
st.subheader("Prediction")

if st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    try:
        prediction, probability = predict_customer(
            input_data
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )

        with col2:
            if prediction == 1:
                st.error(
                    "⚠️ High Risk: Customer is likely to churn"
                )
            else:
                st.success(
                    "✅ Low Risk: Customer is likely to stay"
                )

        st.progress(float(probability))

        with st.expander("View Customer Input"):
            st.dataframe(
                pd.DataFrame([input_data]).T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Customer Churn Prediction | Machine Learning Project | "
    "Gradient Boosting Classifier"
)
