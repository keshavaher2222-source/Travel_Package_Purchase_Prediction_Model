import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Product Purchase Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Product Taken Prediction App")

# ---------------- Load Model & Preprocessor ----------------
model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

# ---------------- Form ----------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧾 Customer Information")

        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        MaritalStatus = st.selectbox("Marital Status", ["Unmarried", "Married", "Divorced"])
        DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, value=10)
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, value=2)
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

    with col2:
        st.subheader("🏠 Additional Details")

        PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
        NumberOfTrips = st.number_input("Number Of Trips", min_value=0, value=1)
        Passport = st.selectbox("Passport", [0, 1])
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        OwnCar = st.selectbox("Own Car", [0, 1])
        Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=30000)
        TotalVisiting = st.number_input("Total Visiting", min_value=0, value=1)

    submit = st.form_submit_button("🚀 Submit")

# ---------------- Prediction Section ----------------
if submit:

    input_data = pd.DataFrame([{
        'Age': Age,
        'TypeofContact': TypeofContact,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': Occupation,
        'Gender': Gender,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': ProductPitched,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': Passport,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': OwnCar,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome,
        'TotalVisiting': TotalVisiting
    }])

    # Preprocess
    processed_data = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    # ---------------- Result Section ----------------
    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.success("✅ Customer is likely to take the product!")
    else:
        st.error("❌ Customer is NOT likely to take the product.")

    st.write(f"**Probability of Taking Product:** {round(probability*100, 2)} %")

    # ---------------- Pie Chart ----------------
    st.subheader("📊 Prediction Probability Chart")

    fig = plt.figure()
    plt.pie(
        [probability, 1 - probability],
        labels=["Will Take", "Will Not Take"],
        autopct="%1.1f%%"
    )
    plt.title("Prediction Probability Distribution")
    st.pyplot(fig)

    # ---------------- Feature Summary ----------------
    st.subheader("📌 Customer Overview")

    numeric_features = {
        "Age": Age,
        "MonthlyIncome": MonthlyIncome,
        "Trips": NumberOfTrips,
        "Followups": NumberOfFollowups,
        "Visiting": TotalVisiting
    }

    fig2 = plt.figure()
    plt.bar(numeric_features.keys(), numeric_features.values())
    plt.xticks(rotation=45)
    plt.title("Key Numeric Features")
    st.pyplot(fig2)