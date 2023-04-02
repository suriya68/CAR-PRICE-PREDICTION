from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('car.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    reviews_count = int(request.form['reviews_count'])
    fuel_type = request.form['fuel_type']
    engine_displacement = float(request.form['engine_displacement'])
    no_cylinder = int(request.form['no_cylinder'])
    seating_capacity = int(request.form['seating_capacity'])
    transmission_type = request.form['transmission_type']
    fuel_tank_capacity = float(request.form['fuel_tank_capacity'])
    
    rating = float(request.form['rating'])
    max_torque_nm = float(request.form['max_torque_nm'])
    max_torque_rpm = int(request.form['max_torque_rpm'])
    max_power_bhp = float(request.form['max_power_bhp'])
    max_power_rpm = int(request.form['max_power_rpm'])
    
    # Create a DataFrame with the input features
    data = pd.DataFrame({
                         'reviews_count': reviews_count,
                         'fuel_type': fuel_type,
                         'engine_displacement': engine_displacement,
                         'no_cylinder': no_cylinder,
                         'seating_capacity': seating_capacity,
                         'transmission_type': transmission_type,
                         'fuel_tank_capacity': fuel_tank_capacity,
                         
                         'rating': rating,
                         'max_torque_nm': max_torque_nm,
                         'max_torque_rpm': max_torque_rpm,
                         'max_power_bhp': max_power_bhp,
                         'max_power_rpm': max_power_rpm}, index=[0])
    
    # Make a prediction using the trained model
    prediction = model.predict(data)[0]
    
    # Return the predicted price to the user
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
