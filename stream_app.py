import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('delivery_delay_sav.pkl')

st.title('Delivery Delay Prediction')
st.write('Enter the features below to predict if there will be a delivery delay.')

# Define the input features based on your X_train columns
delivery_distance = st.slider('Delivery Distance', 0.0, 100.0, 25.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 30, 5)
num_stops = st.slider('Number of Stops', 0, 10, 3)
vehicle_age = st.slider('Vehicle Age (years)', 0, 15, 5)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.slider('Package Weight (kg)', 0.0, 50.0, 10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/L)', 5.0, 25.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 120, 60)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([[delivery_distance, traffic_congestion, weather_condition,
                            delivery_slot, driver_experience, num_stops, vehicle_age,
                            road_condition_score, package_weight, fuel_efficiency,
                            warehouse_processing_time]],
                            columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                                     'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                                     'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                                     'Warehouse_Processing_Time'])

if st.button('Predict Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction[0] == 1:
        st.error(f'Prediction: Delivery is LIKELY to be Delayed (Probability: {prediction_proba[1]:.2f})')
    else:
        st.success(f'Prediction: Delivery is UNLIKELY to be Delayed (Probability: {prediction_proba[0]:.2f})')

    st.write('---')
    st.subheader('Feature Values Inputted:')
    st.write(input_data)
