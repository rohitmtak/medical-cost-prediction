
#import necessary modules
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    # define default file path for preprocessor object and create pickle file
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
        
class DataTransformation:
    def __init__(self):
        # initialize with default configuration
        self.data_transformation_config = DataTransformationConfig()
        
    # method to create and return data transformer object
    def get_preprocessor_object(self):
        try:
            # define numerical and categorical columns
            numerical_columns = ['age', 'bmi', 'children']
            categorical_columns = ['sex', 'smoker', 'region']
            
            # define pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler(with_mean=False)),
                
            ])
            
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder())
            ])
            
            # log column information
            logging.info("Handled missing values for both numerical and categorical features")
            logging.info("Numerical features are scaled")
            logging.info("Categorical features are encoded using OneHotEncoder")
            
            # create ColumnTransformer to apply different preprocessing steps to different columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
            
    #method to initiate data transformation process
    def initiate_data_transformation(self, train_path, valid_path):
        try:
            # read train and test data from CSV files
            train_df = pd.read_csv(train_path)
            valid_df = pd.read_csv(valid_path)
            
            logging.info("Reading the train and test file")
            
            # obtain the data transformer object
            preprocessing_obj = self.get_preprocessor_object()
            
            # define target column name and numerical columns
            target_column_name = 'charges'
            
            # divide the train dataset into independent and dependent features
            input_feature_train = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train = train_df[target_column_name]
            
            # divide the test dataset into independent and dependent features
            input_feature_test = valid_df.drop(columns=[target_column_name], axis=1)
            target_feature_test = valid_df[target_column_name]
            
            logging.info("Applying Preprocessing on train and test dataframe")
            
            # apply preprocessing to training and test dataframes
            train_features = preprocessing_obj.fit_transform(input_feature_train)
            test_features = preprocessing_obj.transform(input_feature_test)
            
            # combine input features with target features for training and test data
            train_data = np.c_[train_features, np.array(target_feature_train)]
            test_data = np.c_[test_features, np.array(target_feature_test)]
            
            logging.info("Saved preprocessing object successfully")
            
            # save preprocessing object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return(
                train_data,
                test_data,
                self.data_transformation_config.preprocessor_obj_file_path
            )
             
        except Exception as e:
            raise CustomException(e, sys)