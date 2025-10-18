import streamlit as st
import numpy as np
import pickle

# === Load model ===
loaded_model = pickle.load(open('accident_predict.pkl', 'rb'))

# === Label maps (replace with your real mappings) ===
label_maps = {
    'Weather': {'Sunny': 0, 'Rainy': 1, 'Foggy': 2},
    'Road_Type': {'Highway': 0, 'Urban': 1, 'Rural': 2},
    'Time_of_Day': {'Day': 0, 'Night': 1},
    'Road_Condition': {'Dry': 0, 'Wet': 1},
    'Vehicle_Type': {'Car': 0, 'Truck': 1, 'Motorbike': 2},
    'Road_Light_Condition': {'Good': 0, 'Poor': 1},
    'Day_of_Week': {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
}

# === Reverse maps (for decoding prediction) ===
reverse_maps = {col: {v: k for k, v in mapping.items()} for col, mapping in label_maps.items()}

# === Streamlit App Title ===
st.title("🚨 Accident Severity Prediction App")
st.write("Provide the following details to predict the accident severity:")

# === Input fields ===
weather = st.selectbox("🌤️ Weather", list(label_maps['Weather'].keys()))
road_type = st.selectbox("🛣️ Road Type", list(label_maps['Road_Type'].keys()))
time_of_day = st.selectbox("🕐 Time of Day", list(label_maps['Time_of_Day'].keys()))
day_of_week = st.selectbox("📅 Day of Week", list(label_maps['Day_of_Week'].keys()))
traffic_density = st.slider("🚗 Traffic Density", 0, 2, 1)
speed_limit = st.slider("🏎️ Speed Limit (km/h)", 30, 150, 60)
num_vehicles = st.slider("🚘 Number of Vehicles", 1, 20, 2)
driver_alcohol = st.selectbox("🍺 Driver Alcohol Influence", [0, 1])
road_condition = st.selectbox("🛤️ Road Condition", list(label_maps['Road_Condition'].keys()))
vehicle_type = st.selectbox("🚗 Vehicle Type", list(label_maps['Vehicle_Type'].keys()))
driver_age = st.slider("👤 Driver Age", 16, 90, 30)
driver_experience = st.slider("🎓 Driver Experience (years)", 0, 60, 5)
road_light = st.selectbox("💡 Road Light Condition", list(label_maps['Road_Light_Condition'].keys()))

# === Prepare input for prediction ===
inputs = np.array([[
    label_maps['Weather'][weather],
    label_maps['Road_Type'][road_type],
    label_maps['Time_of_Day'][time_of_day],
    traffic_density,
    speed_limit,
    num_vehicles,
    driver_alcohol,
    label_maps['Road_Condition'][road_condition],
    label_maps['Vehicle_Type'][vehicle_type],
    driver_age,
    driver_experience,
    label_maps['Road_Light_Condition'][road_light],
    label_maps['Day_of_Week'][day_of_week]
]])

# === Prediction ===
if st.button("🔮 Predict Accident Severity"):
    prediction = loaded_model.predict(inputs)[0]

    # if your model predicts encoded labels (e.g. 0=Low, 1=Medium, 2=High)
    severity_map = {0: "Low", 1: "Medium", 2: "High"}

    predicted_label = severity_map.get(prediction, prediction)
    st.success(f"🚧 **Predicted Accident Severity:** {predicted_label}")
