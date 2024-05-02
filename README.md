# Medical Cost Prediction

Predicting medical costs of individuals based on different features using several ML (Regression) algorithms.

## Dataset
The [Medical Cost Prediction](https://www.kaggle.com/datasets/mirichoi0218/insurance) consists of around 1300 records and six independent variables along with ``charges`` target  variable:

1) ``age``: Age of the individual
   
3) ``children``: Number of children the individual has
   
5) ``bmi``: Body Mass Index of the individual - where bmi <18.5 falls under underweight range, 18.5 - 24.9 falls under normal range, 25.0 - 29.9 falls under overweight range, and >30.0 falls under obese range
   
7) ``sex``: Sex of the individual - Male or Female
   
9) ``smoking``: Whether the individual is a smoker or not
    
11) ``region``: What region the individual belongs to - Northeast, Northwest, Southeast, Southwest
<br></br>
## Getting Started (Cloning)
Clone the repo using 
```
git clone https://github.com/rohitmtak/medical-cost-prediction.git
```
## Installation Steps
Install the required packages from ``requirements.txt`` after commenting out ``-e .`` which runs ``setup.py`` automatically.
```
pip install -r requirements.txt
```
## Usage
1) Once cloned, run the ``data_ingestion.py`` script to load, transform, and train different ML algorithms (Regression) on loaded data. This script creates all the required artifacts (train, test, validation data, model, and preprocessor pickle files).

Model.pkl will have the best model with the best parameters from different models used.
```
python src/components/data_ingestion.py
``` 
2) Run the ``app.py`` which is a Flask application to get the required UI locally.
```
python app.py
```
And, that's it, the application should run perfectly on local machine, and you can test the UI out and play with it.
<br></br>
## Github: 
https://github.com/rohitmtak

## Contact
Rohit Tak - [rohitmtak@gmail.com](rohitmtak@gmail.com)