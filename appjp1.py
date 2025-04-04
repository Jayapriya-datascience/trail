import streamlit as st
import pickle
import numpy as np
import os

# CSS for Multi-Color Blinking Effect
multi_color_blink_css = """
<style>
@keyframes colorBlink {
    0% {color: red;}
    25% {color: green;}
    50% {color: blue;}
    75% {color: orange;}

}
.color-blink {
    animation: colorBlink 1s infinite;
    text-align: center;
    font-weight: bold;
}
</style>
"""

# Inject CSS into Streamlit App
st.markdown(multi_color_blink_css, unsafe_allow_html=True)
##########################

# Load Model and Scaler with Path Handling
model_path = os.path.join("modeljp", "trained_modeljp2.pkl")
scaler_path = os.path.join("modeljp", "scalerjp2.pkl")

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found! Expected at: {model_path}")
    st.stop()

if not os.path.exists(scaler_path):
    st.error(f"❌ Scaler file not found! Expected at: {scaler_path}")
    st.stop()

with open("modeljp/trained_modeljp2.pkl", "rb") as f:
    model = pickle.load(f)

import joblib  # Use joblib instead of pickle for loading scaler

scaler_path = os.path.join("modeljp", "scalerjp2.pkl")

try:
    scaler = joblib.load(scaler_path)  # Proper way to load the scaler
    if not hasattr(scaler, "transform"):  # Check if the object is a valid scaler
        raise TypeError("Loaded scaler is not a valid scaler object.")
    print("✅ Scaler loaded successfully and ready to transform data!")
except Exception as e:
    st.error(f"❌ Error loading scaler: {e}")
    st.stop()

# Title
st.markdown(
    """
    <div style="background-color:#e6f2ff; padding:10px; border-radius:10px; margin-bottom: 10px;">
    <h2 class="color-blink">💤Sleep Disorder Prediction App</h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("🌙 Enter your details to check for sleep disorder risk.")

# User Inputs
st.title("Personal information")
age = st.number_input("Age", min_value=10, max_value=100,value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
gender = gender_mapping[gender]

# Occupation
occupation = st.selectbox("Occupation", ["Nurse", "Doctor", "Engineer", "Lawyer", "Teacher", "Accountant", "Salesperson", "Student","Others"])
occupation_mapping = {"Nurse": 0, "Doctor": 1, "Engineer": 2, "Lawyer": 3, "Teacher": 4, "Accountant": 5, "Salesperson": 6, "Student":7,"Others": 8}
occupation = occupation_mapping[occupation]

# Get Height and Weight Input
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)


if height > 0 and weight > 0:
    bmi = weight / ((height / 100) ** 2)

    if bmi < 18.5:
        bmi_category = 0
    elif 18.5 <= bmi < 24.9:
        bmi_category = 1
    elif 25 <= bmi < 29.9:
        bmi_category = 2
    else:
        bmi_category = 3

    st.text_input("BMI Category", value=["Underweight", "Normal", "Overweight", "Obese"][bmi_category], disabled=True)
    
# Sleep Details
st.title("Sleep Details")
sleep_duration = st.slider("Sleep Duration (hours)", 1.0, 12.0, 7.0)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 5)
st.session_state.sleep_details = {'quality_of_sleep': quality_of_sleep}  # Store it in session state
st.write()
st.markdown("""
<div style="color:blue;">
Quality of Sleep (1-10)<br>
- 1-3: Poor Sleep Quality (Frequent disturbances)  <br>
- 4-6: Fair Sleep Quality (Light sleep, not refreshing)  <br>
- 7-8: Good Sleep Quality (Mostly uninterrupted, refreshing)  <br>
- 9-10: Excellent Sleep Quality (Deep, restorative sleep)  
</div>
""", unsafe_allow_html=True)

physical_activity = st.number_input("Physical Activity Level", 0, 100, value=30,step=10)
st.write()
st.write("""
<div style="color:blue;">
Physical Activity Level (Rating 0-100)<br>
- 0: No Physical Activity  <br>
- 10-30: Low Activity (Sedentary)<br>  
- 31-60: Moderate Activity (Light exercise)<br>  
- 61-80: High Activity (Regular exercise)  <br>
- 81-100: Very High Activity (Intense daily exercise) 
</div>
""", unsafe_allow_html=True)

# Health Details
st.title("Health Details")
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
st.write()
st.write("""
<div style="color:blue;">
Stress Level (0-10)<br>
- 0: No Stress <br> 
- 1-3: Low Stress  <br>
- 4-6: Moderate Stress<br>  
- 7-8: High Stress  <br>
- 9-10: Extreme Stress  
</div>
""", unsafe_allow_html=True)

heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, value=70)
# Daily Steps with Increase Button
daily_steps = st.number_input("Daily Steps (0-10000)", 0, 10000, value=5000,step=100)
st.write()
st.write("""
<div style="color:blue;">
Daily Steps : The average number of steps the individual takes per day
</div>
""", unsafe_allow_html=True)

systolic = st.number_input("Systolic Blood Pressure", 80, 200, value=120)
st.write()
st.write("""
<div style="color:blue;">
Systolic : The systolic blood pressure of the individual in mmHg.
</div>
""", unsafe_allow_html=True)


diastolic = st.number_input("Diastolic Blood Pressure", 50, 130, value=80)
st.write()
st.write("""
<div style="color:blue;">
Diastolic:The diastolic blood pressure of the individual in mmHg.
</div>
""", unsafe_allow_html=True)



# Prepare Data
input_features = np.array([[age, gender, occupation, sleep_duration, quality_of_sleep, 
                            physical_activity, stress_level, bmi_category, heart_rate, 
                            daily_steps, systolic, diastolic]])

# 🚀 Scale the input
try:
    input_features_scaled = scaler.transform(input_features)
except Exception as e:
    st.error(f"Error scaling input: {e}")
    st.stop()

# Sleep Disorders Information
disorder_info = {
    "Insomnia": ("Difficulty falling or staying asleep.", "Reduce caffeine, maintain a sleep schedule, try relaxation techniques."),
    "Sleep Anxiety": ("Anxiety-related sleep disturbances.", "Practice meditation, avoid screens(phones, TV) before bed, deep breathing exercises."),
    "Obstructive Sleep Apnea": ("Breathing stops during sleep due to airway blockage.", "Lose weight, avoid alcohol, consider CPAP therapy."),
    "Hypertension-related Sleep Issues": ("Poor sleep linked to high blood pressure.", "Monitor BP, reduce salt, maintain a balanced diet."),
    "Restless Leg Syndrome": ("Uncontrollable urge to move legs, worse at night.", "Exercise, avoid caffeine, maintain a regular sleep schedule."),
    "Narcolepsy": ("Excessive daytime sleepiness, sudden sleep attacks due to high stress levels.", "Maintain a consistent schedule, avoid heavy meals before bed."),
    "General Sleep Disorder": ("Mild sleep disturbances affecting sleep quality.", "Improve sleep hygiene, avoid blue light before bed, maintain a dark and quiet sleep environment.")
}

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_features_scaled)
    st.session_state.prediction = prediction  # Store prediction in session state
    possible_disorders = []



    
    if prediction[0] == 1:
        st.error("⚠️ High risk of sleep disorder detected! Consult a doctor.")
        
        if sleep_duration < 5 or quality_of_sleep < 3:
            possible_disorders.append("Insomnia")

        if stress_level > 5:
            possible_disorders.append("Sleep Anxiety")
        
        if bmi_category == 3 and heart_rate > 90:
            possible_disorders.append("Obstructive Sleep Apnea")
        
        if systolic > 140 or diastolic > 90:
            possible_disorders.append("Hypertension-related Sleep Issues")
        
        if daily_steps < 3000 and physical_activity < 20:
            possible_disorders.append("Restless Leg Syndrome")
        
        if sleep_duration > 9 or stress_level > 3:
            possible_disorders.append("Narcolepsy")

        if not possible_disorders:
            possible_disorders.append("General Sleep Disorder")

                
            
        st.session_state.possible_disorders = possible_disorders  # Store possible disorders in session state
        st.warning(f"🛏️ **Possible Sleep Disorders:** {', '.join(possible_disorders)}")

        for disorder in possible_disorders:
            if disorder in disorder_info:
                st.subheader(f"🩺 {disorder}")
                st.write(f"🔹 **Definition:** {disorder_info[disorder][0]}")
                st.write(f"💡 **Tips:** {disorder_info[disorder][1]}")

    else:
        st.success("✅ No disease detected! Low risk of sleep disorder. Keep maintaining good habits!")
        st.session_state.possible_disorders = []
        st.subheader("🛌 Tips for Healthy Sleep")
        st.write("1. **Maintain a Consistent Sleep Schedule** – Go to bed and wake up at the same time every day, even on weekends.")
        st.write("2. **Create a Relaxing Bedtime Routine** – Avoid screens, heavy meals, and caffeine before bed. Try reading or meditation.")
        st.write("3. **Stay Physically Active** – Engage in regular exercise, but avoid intense workouts close to bedtime.")
        st.write("4. **Optimize Your Sleep Environment** – Keep your room dark, quiet, and cool for better sleep quality.")
        st.write("5. **Manage Stress and Anxiety** – Practice relaxation techniques like deep breathing, yoga, or journaling to reduce stress before sleep.")

st.write("📌 **Note:** This AI prediction should not replace professional medical advice.")


import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf(disorders):
    pdf_path = "Sleep_Disorder_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Sleep Disorder Prediction Report")
    
    y_position = height - 100
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, "Based on the provided health parameters, the following sleep disorders have been detected:")
    
    disorder_info = {
        "Insomnia": ("A sleep disorder characterized by difficulty falling or staying asleep.",
                     "Maintain a regular sleep schedule, avoid caffeine before bed, and practice relaxation techniques."),
        "Sleep Anxiety": ("Anxiety-related sleep disturbances.",
                          "Practice meditation, avoid screens before bed, and do deep breathing exercises."),
        "Obstructive Sleep Apnea": ("Breathing stops during sleep due to airway blockage.",
                                     "Lose weight, avoid alcohol, and consider CPAP therapy."),
        "Hypertension-related Sleep Issues": ("Poor sleep linked to high blood pressure.",
                                              "Monitor BP, reduce salt intake, and maintain a balanced diet."),
        "Restless Leg Syndrome": ("Uncontrollable urge to move legs, worse at night.",
                                   "Exercise, avoid caffeine, and maintain a regular sleep schedule."),
        "Narcolepsy": ("Excessive daytime sleepiness, sudden sleep attacks due to high stress levels.",
                        "Maintain a consistent schedule and avoid heavy meals before bed."),
        "General Sleep Disorder": ("Mild sleep disturbances affecting sleep quality.",
                                    "Improve sleep hygiene, avoid blue light before bed, and maintain a dark sleep environment.")
    }
    
    y_position -= 30
    for disorder in disorders:
        if disorder in disorder_info:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_position, disorder)
            y_position -= 20
            c.setFont("Helvetica", 12)
            c.drawString(50, y_position, f"Definition: {disorder_info[disorder][0]}")
            y_position -= 20
            c.drawString(50, y_position, f"Tips: {disorder_info[disorder][1]}")
            y_position -= 40

    # ✅ Add a section for general sleep tips at the end
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "🛌 Tips for Healthy Sleep")
    y_position -= 20
    c.setFont("Helvetica", 12)

    sleep_tips = [
        "✔ Maintain a consistent sleep schedule.",
        "✔ Create a relaxing bedtime routine.",
        "✔ Stay physically active but avoid intense workouts before bed.",
        "✔ Optimize your sleep environment (dark, quiet, cool).",
        "✔ Manage stress with meditation, deep breathing, or journaling."
    ]

    for tip in sleep_tips:
        c.drawString(50, y_position, tip)
        y_position -= 20  # Move down for next tip

    c.save()
    return pdf_path

if "possible_disorders" in st.session_state and st.session_state.possible_disorders:
    pdf_file = generate_pdf(st.session_state.possible_disorders)
    
    with open(pdf_file, "rb") as file:
        st.download_button(
            label="📄 Download Sleep Disorder Report", 
            data=file, 
            file_name="Sleep_Disorder_Report.pdf", 
            mime="application/pdf"
        )

        
