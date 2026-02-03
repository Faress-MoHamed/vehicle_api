
from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

model = joblib.load('vehicle_maintenance_model_lgbm.pkl')
oe = joblib.load('ordinal_encoder.pkl')
ohe = joblib.load('onehot_encoder.pkl')
std_scaler = joblib.load('standard_scaler.pkl')
rob_scaler = joblib.load('robust_scaler.pkl')

app = Flask(__name__)

ord_cols = [
    'Brake_Condition',
    'Impact_on_Efficiency_cat',
    'Vibration_Levels_cat',
    'Fuel_Consumption_cat'
]

onehot_cols = ['Make_and_Model', 'Route_Info', 'Weather_Conditions', 'Road_Conditions', 'Maintenance_Type','Tire_Pressure_cat','Oil_Quality_cat','Downtime_Maintenance_cat','Vehicle_Type']

numeric_cols = ['Year_of_Manufacture', 'Failure_History', 'Load_Capacity_final',
                'Actual_Load', 'Usage_Hours']


# ===== Bins for mid value calculation =====
OIL_QUALITY_MID = {
    'Poor': 55.75,     # (38.3 + 73.2)/2
    'Fair': 79.95,     # (73.2 + 86.7)/2
    'Good': 93.35      # (86.7 + 100)/2
}

VIBRATION_MID = {
    'Low': 0.565,      # (0 + 1.13)/2
    'Medium': 3.3,     # (1.13 + 5.47)/2
    'High': 24.765     # (5.47 + 44.06)/2
}



@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None

   
    if request.method == 'POST' and 'predict' in request.form:

        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # ===== Numeric =====
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # ===== Date =====
        df['Last_Maintenance_Date'] = pd.to_datetime(df['Last_Maintenance_Date'])
        today = pd.Timestamp.today()

        df['days_since_last_maintenance'] = (today - df['Last_Maintenance_Date']).dt.days
        df['last_maintenance_month'] = df['Last_Maintenance_Date'].dt.month
        df['last_maintenance_dayofweek'] = df['Last_Maintenance_Date'].dt.dayofweek
        df.drop('Last_Maintenance_Date', axis=1, inplace=True)

        # ===== Anomalies =====
        df['Anomalies_Detected'] = 0.0
        df.loc[df['Vibration_Levels_cat'] == 'Medium', 'Anomalies_Detected'] += 0.5
        df.loc[df['Vibration_Levels_cat'] == 'High', 'Anomalies_Detected'] += 1
        df.loc[df['Tire_Pressure_cat'] == 'Low', 'Anomalies_Detected'] += 1
        df.loc[df['Oil_Quality_cat'] == 'Fair', 'Anomalies_Detected'] += 0.5
        df.loc[df['Oil_Quality_cat'] == 'Poor', 'Anomalies_Detected'] += 1
        df.loc[df['Impact_on_Efficiency_cat'] == 'Medium', 'Anomalies_Detected'] += 0.5
        df.loc[df['Impact_on_Efficiency_cat'] == 'High', 'Anomalies_Detected'] += 1

        # ===== Feature Engineering =====
        df['vehicle_age'] = pd.Timestamp.today().year - df['Year_of_Manufacture']
        df['usage_per_year'] = df['Usage_Hours'] / (df['vehicle_age'] + 1)
        df['load_ratio'] = df['Actual_Load'] / (df['Load_Capacity_final'] + 1e-5)
        
        #Log Transformation
        df['Load_Capacity_final'] = np.log1p(df['Load_Capacity_final'])
        df['usage_per_year_final'] = np.log1p(df['usage_per_year'])
        
        # ===== Mechanical Stress =====
        ENGINE_TEMP = 120
        df['mechanical_stress'] = (
        df['Oil_Quality_cat'].map(OIL_QUALITY_MID) +
        df['Vibration_Levels_cat'].map(VIBRATION_MID) +
        ENGINE_TEMP
        ) / 3
        
        # ===== Encoding =====
        df[ord_cols] = oe.transform(df[ord_cols])

        ohe_df = pd.DataFrame(
            ohe.transform(df[onehot_cols]),
            columns=ohe.get_feature_names_out(onehot_cols)
        )

        df = pd.concat([df.drop(columns=onehot_cols), ohe_df], axis=1)

        # ===== Align Columns =====
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        # ===== Scaling =====
        df[std_scaler.feature_names_in_] = std_scaler.transform(
            df[std_scaler.feature_names_in_]
        )
        df[rob_scaler.feature_names_in_] = rob_scaler.transform(
            df[rob_scaler.feature_names_in_]
        )
        # ===== Best threshold from validation =====
        BEST_THRESHOLD = 0.4 
        # ===== Predict =====
        probability = float(model.predict_proba(df)[0, 1])
        prediction = int(probability > BEST_THRESHOLD)


    return render_template(
        'index.html',
        prediction=prediction,
        probability=round(probability, 3) if probability is not None else None
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




