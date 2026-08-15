import joblib
import pandas as pd


PREPROCESSOR_FILE = "preprocessing.pkl"
MODEL_FILE = "customer_churn_model.pkl"


print("Checking model files...")

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------

preprocessing = joblib.load(
    PREPROCESSOR_FILE
)

model = joblib.load(
    MODEL_FILE
)

print("Preprocessor:", type(preprocessing))
print("Model:", type(model))


# --------------------------------------------------
# Create test customer
# --------------------------------------------------

test_customer = pd.DataFrame([{
    "latitude": 36.0,
    "longitude": -118.0,
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Dsl",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "paperless_billing": "Yes",
    "payment_method": "Electronic_Check",
    "monthly_charges": 70.0,
    "total_charges": 1000.0,
    "churn_score": 50.0,
    "cltv": 4500.0
}])


# --------------------------------------------------
# Display input
# --------------------------------------------------

print("\nInput dtypes:")
print(test_customer.dtypes)


# --------------------------------------------------
# Transform
# --------------------------------------------------

X_processed = preprocessing.transform(
    test_customer
)

print("\nTransformation successful!")
print("Processed type:", type(X_processed))
print("Processed shape:", X_processed.shape)

if hasattr(X_processed, "dtypes"):
    print("Processed dtypes:")
    print(X_processed.dtypes)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = model.predict(
    X_processed
)

probability = model.predict_proba(
    X_processed
)[:, 1]


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nPrediction:", prediction)
print("Churn probability:", probability)

print("\nLOCAL MODEL TEST PASSED.")