import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def initiate_model_trainer(self, train_array, valid_array):
        try:
            logging.info("Split train and test input data")
            X_train, y_train, X_valid, y_valid = (
                train_array[:,:-1],
                train_array[:,-1],
                valid_array[:,:-1],
                valid_array[:,-1]
            )
            
            # define models and their hyperparameters
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost Regressor": XGBRFRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                            
                "Random Forest": {
                    'n_estimators':[8,16,32,64,128,256]
                },
                
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "Linear Regression":{},
                
                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            # evaluate models and get a report of their performance
            model_report: dict = evaluate_model(X_train, y_train, X_valid, y_valid, models, params)
            
            # get the best model score and name from the report dict
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            print(f"The best model we found is: {best_model_name}")
            
            model_names = list(params.keys())
            
            actual_model = ""
            
            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model +  model
                    
            best_params = params[actual_model]

            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model on both train and test dataset")
            
            # save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            
            # predict using the best model and compute R^2 score
            predicted = best_model.predict(X_valid)
            r2_square = r2_score(y_valid, predicted)
            return r2_square
        
            
        except Exception as e:
            raise CustomException(e,sys)