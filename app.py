import streamlit as st
import joblib
import numpy as np

st.title("Salary Prediction App")
st.divider()

st.write("With this app, you can get estimations for the salaries of employees.")

# Input for number of years
years = st.number_input("Enter years of experience", value=1, step=1, min_value=0)

# Input for job rate
jobrate = st.number_input("Enter the job rate", value=3.5, step=0.5, min_value=0.0)

# Feature vector
X = [years, jobrate]

# Load trained model
model = joblib.load("linearmodel.pkl")

st.divider()

# Button to predict salary
predict = st.button("Press the button for salary prediction")

st.divider()

if predict:
    st.balloons()

    # Convert input to NumPy array
    X1 = np.array([X])

    # Make prediction
    prediction = model.predict(X1)

    # Show the prediction result
    st.write(f"Salary prediction is {prediction[0]:.2f}")

else:
    st.write("Please press the button for app to make the prediction")
