from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder='ui_elements')

app = application

# route for home page
@app.route('/') 
def index():
    return render_template('home.html')

# route for predicting data
@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # creating CustomData object from form input
        data = CustomData(
            age = request.form.get('age'),
            children = request.form.get('children'),
            bmi = request.form.get('bmi'),
            sex = request.form.get('sex'),
            smoker = request.form.get('smoker'),
            region = request.form.get('region'),
        )
        
        
        pred_df = data.get_data_as_dataFrame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=int(results[0]))
    
    
if __name__ == "__main__":
    app.run(host="localhost", debug=True, port=5000)