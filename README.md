# 📊 Customer Churn Prediction

A Machine Learning web application that predicts whether a telecom customer is likely to churn.

The application is built using **Python, Scikit-learn and Streamlit** and is designed for deployment through **Streamlit Community Cloud**.

## Project Overview

Customer churn prediction is a binary classification problem.

- `0` → Customer is retained
- `1` → Customer is likely to churn

Several classification algorithms were evaluated during model development:

- Logistic Regression
- SGD Classifier
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM
- MLP Classifier

The **GradientBoostingClassifier** was selected as the final candidate based on the model comparison, particularly its performance on the churn class.

## Final Model

Model:

```text
GradientBoostingClassifier
```

Preprocessing:

- `StandardScaler` for numerical features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features
- `ColumnTransformer`

Model selection / tuning:

- 5-fold `StratifiedKFold`
- `GridSearchCV`
- F1-score used as the tuning metric

The fitted preprocessing object is saved separately so that the Streamlit application applies exactly the same transformations used during training.

## Features Used

The application accepts:

### Demographics

- latitude
- longitude
- gender
- senior_citizen
- partner
- dependents

### Account

- tenure_months

### Services

- phone_service
- multiple_lines
- internet_service
- online_security
- online_backup
- device_protection
- tech_support
- streaming_tv
- streaming_movies

### Billing

- paperless_billing
- payment_method
- monthly_charges
- total_charges

### Customer value / score

- churn_score
- cltv

## Repository Structure

```text
customer-churn-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── preprocessing.pkl
└── customer_churn_model.pkl
```

## 1. Save the Trained Artifacts

Run this code in the same notebook where these variables exist:

```python
import joblib

# Save the fitted preprocessing object
joblib.dump(
    preprocessing,
    "preprocessing.pkl"
)

# Save the final model together with the training-time
# total_charges median used to replace zero values.
model_artifact = {
    "model": best_model,
    "total_charges_median": median_value
}

joblib.dump(
    model_artifact,
    "customer_churn_model.pkl"
)
```

Verify:

```python
import os

print(os.path.exists("preprocessing.pkl"))
print(os.path.exists("customer_churn_model.pkl"))
```

Expected:

```text
True
True
```

### Check the files

```python
import os

print(os.path.getsize("preprocessing.pkl"))
print(os.path.getsize("customer_churn_model.pkl"))
```

The files should have a non-zero size.

## 2. Check the Scikit-learn Version

Because serialized Scikit-learn models should ideally be loaded with the same Scikit-learn version used for training, check your training environment:

```python
import sklearn
print(sklearn.__version__)
```

If necessary, pin that exact version in `requirements.txt`.

## 3. Test Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## 4. Create the GitHub Repository

Create a repository such as:

```text
customer-churn-prediction
```

Upload these files:

```text
app.py
requirements.txt
README.md
preprocessing.pkl
customer_churn_model.pkl
```

Do not upload the original raw dataset unless you have permission to redistribute it.

## 5. Deploy on Streamlit Community Cloud

1. Go to Streamlit Community Cloud.
2. Sign in with GitHub.
3. Connect/authorize your GitHub account if requested.
4. Select **Create app**.
5. Select **Yup, I have an app**.
6. Select your GitHub repository.
7. Select the branch, normally `main`.
8. Set the main file to:

```text
app.py
```

9. Choose an app subdomain if desired.
10. Click **Deploy**.

## 6. After Deployment

Streamlit Community Cloud installs the packages listed in `requirements.txt`.

If the deployment succeeds, you will receive a URL similar to:

```text
https://your-app-name.streamlit.app
```

Open the URL and test the prediction form.

## 7. Common Deployment Errors

### Error: FileNotFoundError

Example:

```text
FileNotFoundError: preprocessing.pkl
```

Make sure both model files are in the repository root:

```text
app.py
preprocessing.pkl
customer_churn_model.pkl
```

### Error: Feature mismatch

Example:

```text
X has ... features, but the model is expecting ...
```

This usually means that the deployed input does not match the training preprocessing pipeline.

Do not manually recreate the encoder in `app.py`.

The application loads the original fitted:

```text
preprocessing.pkl
```

and calls:

```python
preprocessing.transform(input_df)
```

### Error: Scikit-learn version mismatch

If you see a warning/error about the serialized model being created with a different Scikit-learn version, use the same Scikit-learn version in the deployment environment as the training environment.

Check:

```python
import sklearn
print(sklearn.__version__)
```

Then pin that version in:

```text
requirements.txt
```

### Error: App cannot start

Open the Streamlit app logs and inspect the first Python exception. Usually the root cause is a missing dependency, missing model artifact, or package-version incompatibility.

## Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Preprocessing
     ├── StandardScaler
     └── OneHotEncoder
     ↓
Baseline Model Comparison
     ↓
Gradient Boosting Selection
     ↓
5-Fold Stratified Cross Validation
     ↓
GridSearchCV
     ↓
Best Model
     ↓
Final Test Evaluation
     ↓
Model Serialization
     ↓
Streamlit Application
     ↓
GitHub
     ↓
Streamlit Community Cloud
```

## Deployment Architecture

```text
User
  │
  ▼
Streamlit Web App
  │
  ├── Customer Input
  │
  ▼
preprocessing.pkl
  │
  ▼
GradientBoostingClassifier
  │
  ▼
Churn Probability
  │
  ▼
Churn / No Churn
```

## Important Modeling Note

Before presenting this project as a production-quality churn system, verify that `churn_score` is not derived from the target or from information unavailable at prediction time.

If `churn_score` was calculated using post-churn information or the target label, it can introduce target leakage and should be removed from the model.

## Disclaimer

This project is an educational Machine Learning application and should not be used as the sole basis for customer-related business decisions.

## Author

**Gnanaprakash**

Customer Churn Prediction — Machine Learning Project
