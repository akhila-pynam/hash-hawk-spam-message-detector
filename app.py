import streamlit as st
from spam_detector import predict

st.title("📩 Hash Hawk - Spam Message Detector")

message = st.text_input("Enter your message:")

if st.button("Check"):
    if message:
        result = predict(message)
        st.success(f"Result: {result}")
    else:
        st.warning("Please enter a message")