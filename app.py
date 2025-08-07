import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Student Performance Prediction & Placement Eligibility")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Student_Performance.csv')
    return df

df = load_data()

# Show dataset info if checkbox enabled
if st.checkbox("Show Dataset Preview"):
    st.write(df.head())
    st.write(df.info())
    st.write(df.describe())
    st.write(df.tail())

# Encode categorical column
lb = LabelEncoder()
df['Extracurricular Activities'] = lb.fit_transform(df['Extracurricular Activities'])

# Prepare features and target
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
regression = LinearRegression()
regression.fit(X_train, y_train)

# Model evaluation
y_predict = regression.predict(X_test)
r2 = r2_score(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

st.subheader("Model Performance on Test Set")
st.write(f"R2 Score: {r2 * 100:.2f}%")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Input form for prediction
st.subheader("Make a Prediction")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=100, value=7)
    attendance = st.number_input("Attendance Percentage", min_value=0, max_value=100, value=99)
    extracurricular = st.selectbox("Extracurricular Activities", options=lb.classes_)

with col2:
    assignments_completed = st.number_input("Assignments Completed", min_value=0, max_value=20, value=9)
    previous_scores = st.number_input("Previous Semester Score (out of 10)", min_value=0, max_value=10, value=1)

# Convert extracurricular to encoded value
extracurricular_encoded = lb.transform([extracurricular])[0]

if st.button("Predict Performance Index"):
    input_features = np.array([[study_hours, attendance, extracurricular_encoded, assignments_completed, previous_scores]])
    pred_result = regression.predict(input_features)[0]
    st.write(f"Predicted Performance Index: {pred_result:.2f}")
    if pred_result >= 60:
        st.success("You are eligible for placements 🎉")
    else:
        st.error("Sorry, you are not eligible for placements.")