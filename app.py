import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

PREPROCESSOR_FILE = "preprocessing.pkl"
MODEL_FILE = "customer_churn_model.pkl"


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    preprocessing = joblib.load(
        PREPROCESSOR_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    return preprocessing, model


try:

    preprocessing, model = load_artifacts()

except Exception as e:

    st.error(
        "Unable to load the model files."
    )

    st.stop()


# ============================================================
# EXPECTED FEATURES
# ============================================================

FEATURE_ORDER = [
    "latitude",
    "longitude",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "churn_score",
    "cltv"
]


NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "churn_score",
    "cltv"
]


CATEGORICAL_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
    "payment_method"
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_customer(input_data):

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    test_customer = pd.DataFrame(
        [input_data],
        columns=FEATURE_ORDER
    )

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        test_customer[column] = pd.to_numeric(
            test_customer[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Convert categorical columns
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        test_customer[column] = (
            test_customer[column]
            .astype(str)
        )

    # --------------------------------------------------------
    # Check for invalid numerical values
    # --------------------------------------------------------

    if test_customer[NUMERIC_COLUMNS].isna().any().any():

        invalid_columns = (
            test_customer[NUMERIC_COLUMNS]
            .columns[
                test_customer[NUMERIC_COLUMNS]
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            f"Invalid numerical values in: {invalid_columns}"
        )

    # --------------------------------------------------------
    # IMPORTANT
    # Apply the SAME preprocessing used during training
    # --------------------------------------------------------

    X_processed = preprocessing.transform(
        test_customer
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X_processed
    )[0]

    probability = model.predict_proba(
        X_processed
    )[0, 1]

    return int(prediction), float(probability)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "📊 Customer Churn Prediction"
)

st.markdown(
    """
    Predict whether a telecom customer is likely to churn
    based on demographic, service, tenure and billing information.
    """
)

st.info(
    "Machine Learning Model: Gradient Boosting Classifier"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Customer Information"
)


# ============================================================
# DEMOGRAPHICS
# ============================================================

st.sidebar.subheader(
    "Demographics"
)


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
    [
        "Male",
        "Female"
    ]
)


senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [
        "No",
        "Yes"
    ]
)


partner = st.sidebar.selectbox(
    "Partner",
    [
        "No",
        "Yes"
    ]
)


dependents = st.sidebar.selectbox(
    "Dependents",
    [
        "No",
        "Yes"
    ]
)


# ============================================================
# ACCOUNT
# ============================================================

st.sidebar.subheader(
    "Account"
)


tenure_months = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12,
    step=1
)


# ============================================================
# SERVICES
# ============================================================

st.sidebar.subheader(
    "Services"
)


phone_service = st.sidebar.selectbox(
    "Phone Service",
    [
        "No",
        "Yes"
    ]
)


multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "No",
        "Yes"
    ]
)


internet_service = st.sidebar.selectbox(
    "Internet Service",
    [
        "Dsl",
        "Fiber Optic",
        "No"
    ]
)


online_security = st.sidebar.selectbox(
    "Online Security",
    [
        "No",
        "Yes"
    ]
)


online_backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "No",
        "Yes"
    ]
)


device_protection = st.sidebar.selectbox(
    "Device Protection",
    [
        "No",
        "Yes"
    ]
)


tech_support = st.sidebar.selectbox(
    "Tech Support",
    [
        "No",
        "Yes"
    ]
)


streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    [
        "No",
        "Yes"
    ]
)


streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "No",
        "Yes"
    ]
)


# ============================================================
# BILLING
# ============================================================

st.sidebar.subheader(
    "Billing"
)


paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "No",
        "Yes"
    ]
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


# ============================================================
# CUSTOMER VALUE
# ============================================================

st.sidebar.subheader(
    "Customer Value"
)


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


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = {

    "latitude": float(latitude),

    "longitude": float(longitude),

    "gender": str(gender),

    "senior_citizen": str(senior_citizen),

    "partner": str(partner),

    "dependents": str(dependents),

    "tenure_months": int(tenure_months),

    "phone_service": str(phone_service),

    "multiple_lines": str(multiple_lines),

    "internet_service": str(internet_service),

    "online_security": str(online_security),

    "online_backup": str(online_backup),

    "device_protection": str(device_protection),

    "tech_support": str(tech_support),

    "streaming_tv": str(streaming_tv),

    "streaming_movies": str(streaming_movies),

    "paperless_billing": str(paperless_billing),

    "payment_method": str(payment_method),

    "monthly_charges": float(monthly_charges),

    "total_charges": float(total_charges),

    "churn_score": float(churn_score),

    "cltv": float(cltv)
}


# ============================================================
# PREDICTION
# ============================================================

st.subheader(
    "Prediction"
)


if st.button(
    "🔮 Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    try:

        prediction, probability = predict_customer(
            input_data
        )

        # ----------------------------------------------------
        # Result columns
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        with col1:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with col2:

            if prediction == 1:

                st.error(
                    "⚠️ High Risk: Customer is likely to churn"
                )

            else:

                st.success(
                    "✅ Low Risk: Customer is likely to stay"
                )


        # ----------------------------------------------------
        # Probability bar
        # ----------------------------------------------------

        st.progress(
            probability
        )


        # ----------------------------------------------------
        # Customer information
        # ----------------------------------------------------

        with st.expander(
            "View Customer Information"
        ):

            display_df = pd.DataFrame(
                [input_data]
            ).T

            display_df.columns = [
                "Value"
            ]

            st.dataframe(
                display_df,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Prediction failed. Please verify the input values."
        )

        # Show error while debugging/developing.
        # Remove this line before final public deployment
        # if you don't want technical details exposed.
        st.error(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction | "
    "Gradient Boosting Classifier | "
    "Machine Learning Project"
)
