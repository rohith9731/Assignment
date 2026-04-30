import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="🩺 Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    file_id = "1b0D1qkVrU8N_EokCZDPFVDNJyPt50Jxq"
    url = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(url)

    # Handling missing values
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_invalid] = df[zero_invalid].replace(0, np.nan)

    for col in zero_invalid:
        df[col].fillna(df[col].median(), inplace=True)

    return df


# -----------------------------------
# TRAIN MODEL
# -----------------------------------
@st.cache_resource
def train_model(df):
    FEATURES = ["Pregnancies", "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigreeFunction", "Age"]

    X = df[FEATURES]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, FEATURES


# -----------------------------------
# APP TITLE
# -----------------------------------
st.title("🩺 Diabetes Risk Prediction App")

df = load_data()
model, scaler, FEATURES = train_model(df)

# -----------------------------------
# DATA OVERVIEW
# -----------------------------------
st.header("📊 Dataset Overview")

if st.checkbox("Show Raw Data"):
    st.write(df.head())

if st.checkbox("Show Summary Statistics"):
    st.write(df.describe())

if st.checkbox("Show Missing Values"):
    st.write(df.isnull().sum())

# -----------------------------------
# VISUALIZATIONS
# -----------------------------------
st.header("📈 Data Visualization")

if st.checkbox("Show Histograms"):
    fig, ax = plt.subplots()
    df.hist(figsize=(10, 8), ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Boxplot"):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

if st.checkbox("Show Pairplot"):
    fig = sns.pairplot(df, hue='Outcome')
    st.pyplot(fig)

# -----------------------------------
# CORRELATION HEATMAP
# -----------------------------------
st.header("🔍 Correlation Analysis")

if st.checkbox("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("**Insights:**")
    st.write("- Glucose has strong correlation with diabetes outcome")
    st.write("- BMI shows moderate influence")
    st.write("- Age and Pregnancies also contribute")

# -----------------------------------
# MODEL INTERPRETATION
# -----------------------------------
st.header("🤖 Model Interpretation")

if st.checkbox("Show Model Coefficients"):
    coeff_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_[0]
    })
    st.write(coeff_df)

    st.write("""
    **Interpretation:**
    - Positive coefficient → increases diabetes risk
    - Negative coefficient → decreases risk
    - Glucose usually has highest impact
    """)

# -----------------------------------
# USER INPUT
# -----------------------------------
st.header("🧾 Enter Patient Details")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("Predict Diabetes Risk"):

    input_data = np.array([[pregnancies, glucose, bp, skin,
                            insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Diabetes (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability: {probability:.2f})")

# -----------------------------------
# INTERVIEW QUESTIONS
# -----------------------------------
st.header("🎯 Interview Questions")

st.write("""
### 1. What is the difference between Precision and Recall?

**Precision** measures how many of the predicted positive cases are actually positive.

Precision = TP / (TP + FP)

**Recall** measures how many of the actual positive cases are correctly identified.

Recall = TP / (TP + FN)

**Key Difference:**
- Precision focuses on correctness of positive predictions
- Recall focuses on capturing all actual positive cases

**Example (Diabetes Prediction):**
- High Precision → If model predicts diabetes, it is likely correct  
- High Recall → Model detects most diabetes patients  

In medical applications, **Recall is more important** because missing a patient can be dangerous.

---

### 2. What is Cross-Validation and why is it important?

Cross-validation is a technique used to evaluate model performance by splitting data into multiple parts.

**K-Fold Cross Validation:**
- Data is divided into K parts
- Train on K-1 parts, test on remaining part
- Repeat K times
- Take average performance

**Importance:**
- Prevents overfitting  
- Gives reliable performance estimate  
- Uses full dataset efficiently  
- Works well for binary classification problems  

In diabetes prediction, it ensures the model performs well on both classes (diabetic and non-diabetic).
""")