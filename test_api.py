import requests
import json

url = 'http://127.0.0.1:5000/predict'

data = {
    'Year_of_Manufacture': 2018,
    'Failure_History': 0,
    'Load_Capacity_final': 1000,
    'Actual_Load': 500,
    'Usage_Hours': 5000,
    'Last_Maintenance_Date': '2023-01-01',
    'Vibration_Levels_cat': 'Low',
    'Tire_Pressure_cat': 'Normal',
    'Oil_Quality_cat': 'Good',
    'Impact_on_Efficiency_cat': 'Low',
    'Make_and_Model': 'Truck A',
    'Route_Info': 'Long Haul',
    'Weather_Conditions': 'Clear',
    'Road_Conditions': 'Good',
    'Maintenance_Type': 'Preventive',
    'Downtime_Maintenance_cat': 'Low',
    'Vehicle_Type': 'Truck',
    'Fuel_Consumption_cat': 'Low',
    'Brake_Condition': 'Good'
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())
except requests.exceptions.ConnectionError:
    print("Failed to connect. Make sure the server is running.")
