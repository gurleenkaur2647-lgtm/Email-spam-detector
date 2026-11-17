
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Load trained model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Heading
st.title("📧 Email Spam Detector")
st.write("Check whether an email is Spam or Non-Spam.")

# Email input
email = st.text_area(
    "Enter Email:",
    height=200,
    placeholder="Paste your email message here..."
)

# Check button
if st.button("🔍 Check Email", use_container_width=True):

    if email.strip():

        # Convert email text into numerical features
        message_vectorized = vectorizer.transform([email])

        # Predict email category
        prediction = model.predict(message_vectorized)[0]

        # Classify result
        if str(prediction).lower() in ["1", "spam"]:
            st.error("🚨 SPAM DETECTED!")
            st.write("This email is classified as Spam.")
        else:
            st.success("✅ NON-SPAM EMAIL")
            st.write("This email is classified as Non-Spam.")

        # Optional confidence score
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(message_vectorized)[0]
            confidence = max(probabilities) * 100

            st.write(f"Prediction confidence: {confidence:.2f}%")

    else:
        st.warning("⚠️ Please enter an email before checking.")
