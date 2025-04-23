import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Stress Level Detection App")
st.markdown("Predict your stress level based on lifestyle and sleep habits.")

# Mappings
gender_map = {"Female": 0, "Male": 1}
bmi_map = {"Normal": 0, "Overweight": 1, "Obese": 2, "Underweight": 3}
sleep_disorder_map = {"None": 0, "Insomnia": 1, "Sleep Apnea": 2}
occupation_map = {
    "Software Engineer": 0, "Doctor": 1, "Teacher": 2, "Nurse": 3,
    "Lawyer": 4, "Accountant": 5, "Salesperson": 6, "Scientist": 7,
    "Manager": 8, "Student": 9, "Other": 10
}
sleep_quality_map = {
    "1–3: Poor sleep (frequent waking, restlessness, low energy)": 2,
    "4–6: Average or disturbed sleep (some issues, not fully rested)": 5,
    "7–8: Good sleep (mostly undisturbed, feel okay)": 7,
    "9–10: Excellent sleep (deep, restful, uninterrupted)": 9
}

# Default values
defaults = {
    "gender": "Select Gender",
    "age": "",
    "occupation": "Select Occupation",
    "sleep_duration": "",
    "activity": "",
    "hr": "",
    "steps": "",
    "quality": "Select Quality of Sleep",
    "bmi": list(bmi_map.keys())[0],
    "disorder": list(sleep_disorder_map.keys())[0]
}

# ✅ RESET (BEFORE WIDGETS)
if "reset_flag" in st.session_state and st.session_state.reset_flag:
    st.session_state.update(defaults)
    st.session_state.reset_flag = False
    st.experimental_rerun()

# 🌟 Input Form
with st.form("stress_form"):
    gender = st.selectbox("Select Gender", ["Select Gender"] + list(gender_map.keys()), key="gender")
    age = st.text_input("Enter Age", key="age")
    occupation = st.selectbox("Select Occupation", ["Select Occupation"] + list(occupation_map.keys()), key="occupation")
    sleep_duration = st.text_input("Sleep Duration (in hours)", key="sleep_duration")
    physical_activity = st.text_input("Physical Activity Level (minutes per day)", key="activity")
    heart_rate = st.text_input("Heart Rate (bpm)", key="hr")
    daily_steps = st.text_input("Daily Steps (steps per day)", key="steps")
    quality_sleep_label = st.selectbox("Select Quality of Sleep", ["Select Quality of Sleep"] + list(sleep_quality_map.keys()), key="quality")
    bmi = st.selectbox("BMI Category", list(bmi_map.keys()), key="bmi")
    sleep_disorder = st.selectbox("Sleep Disorder", list(sleep_disorder_map.keys()), key="disorder")

    predict = st.form_submit_button("Predict Stress Level")
    reset = st.form_submit_button("Reset")

# ✅ Trigger Reset
if reset:
    st.session_state.reset_flag = True
    st.experimental_rerun()

# 🤖 Prediction logic
if predict:
    try:
        if (
            gender == "Select Gender"
            or occupation == "Select Occupation"
            or quality_sleep_label == "Select Quality of Sleep"
        ):
            st.warning("Please select Gender, Occupation, and Quality of Sleep.")
        elif not age or not sleep_duration or not physical_activity or not heart_rate or not daily_steps:
            st.warning("Please fill in all numeric fields.")
        else:
            input_data = np.array([[
                gender_map[gender],
                int(age),
                occupation_map[occupation],
                float(sleep_duration),
                sleep_quality_map[quality_sleep_label],
                int(physical_activity),
                bmi_map[bmi],
                int(heart_rate),
                int(daily_steps),
                sleep_disorder_map[sleep_disorder]
            ]])

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)

            stress_labels = {0: "High", 1: "Low", 2: "Medium"}
            st.success(f"🧘 Predicted Stress Level: **{stress_labels[prediction[0]]}**")
    except ValueError:
        st.error("❗ Please enter valid numeric values.")
