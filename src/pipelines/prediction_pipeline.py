import sys
import os
from src.logger import logging
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            logging.info("Loading model and preprocessor objects..")
            print("Before Loading")
            
            # load model and preprocessor object
            model= load_object(file_path = model_path)
            preprocessor = load_object(file_path= preprocessor_path)
            logging.info("Model and preprocessor objects loaded successfully")
            print("After Loading")
            
            # transform features using preprocessor and make predictions
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            logging.info("Prediction made successfully")
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
        
# numerical_features = ['children']
# categorical_features = ['age_range', 'bmi_range', 'sex', 'smoker', 'region']

class CustomData:
    def __init__(self,
        age: int,
        children: int,
        bmi: float,
        sex: str,
        smoker: str,
        region: str):
        
        # # Validate input data before assigning to attributes
        # if not isinstance(age, int):
        #     raise ValueError("Age must be an integer.")
        # if not isinstance(children, int):
        #     raise ValueError("Children must be an integer.")
        # if not isinstance(bmi, float):
        #     raise ValueError("BMI must be a float.")
        # if sex not in ["male", "female"]:
        #     raise ValueError("Sex must be 'male' or 'female'.")
        # if smoker not in ["yes", "no"]:
        #     raise ValueError("Smoker must be 'yes' or 'no'.")
        # if region not in ["northeast", "northwest", "southeast", "southwest"]:
        #     raise ValueError("Region must be one of 'northeast', 'northwest', 'southeast', 'southwest'.")
        
        self.age = age
        self.children = children
        self.bmi = bmi if bmi else None
        self.sex = sex
        self.smoker = smoker
        self.region = region
        
    def get_data_as_dataFrame(self):
        try:
            custom_data_input_dict = {
                "age" : [self.age],
                "children" : [self.children],
                "bmi" : [self.bmi],
                "sex" : [self.sex],
                "smoker" : [self.smoker],
                "region" : [self.region]
            } 
            
            convert_dict = {
                "age": int,
                "children": int,
                "bmi": float,
                "sex": str,
                "smoker": str,
                "region": str
            }            
            
            df = pd.DataFrame(custom_data_input_dict)
            
            df = df.astype(convert_dict)
            logging.info("Data loaded successfully into dataframe")
            print(df.info())
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)