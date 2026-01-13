import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Title
st.title("Hourly Public Bus Passenger Demand Prediction")

st.write("This system predicts passenger demand for public buses based on time.")

# Sample training data (Hour vs Passenger Count)
hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]).reshape(-1, 1)
passengers = np.array([20, 15, 25, 40, 80, 120, 150, 140, 130, 160, 100, 50])

# Train model
model = LinearRegression()
model.fit(hours, passengers)

# User input
hour = st.slider("Select Hour of the Day", 0, 23)

# Prediction
prediction = model.predict([[hour]])

st.subheader("Predicted Passenger Count:")
st.success(f"{int(prediction[0])} passengers")

# Plot
st.subheader("Passenger Demand Trend")
plt.plot(hours, passengers, marker='o')
plt.scatter(hour, prediction, s=100)
plt.xlabel("Hour")
plt.ylabel("Passenger Count")
st.pyplot(plt)
