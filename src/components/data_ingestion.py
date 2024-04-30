# import necessary modules
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    validation_data_path: str = os.path.join('artifacts', 'validation_data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion")
        try:
            # read raw data from notebook
            df = pd.read_csv(os.path.join('notebook/data', 'insurance.csv'))
            logging.info("Successfully read the dataset as dataframe")            
            
            # drop the duplicates
            df.drop_duplicates(inplace=True)
            logging.info("Apply necessary data cleaning")
            
            # save raw data to a file
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # create directory if not exists
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # split data into train and test set
            train_set, remaining_set = train_test_split(df, test_size=0.7, random_state=42)
            valid_set, test_set = train_test_split(remaining_set, test_size=0.15, random_state=42)
            logging.info("Train test split done")
            
            # save train and test sets to a separate files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            valid_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__== '__main__':
    obj = DataIngestion()
    train_path, validation_path = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path, validation_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))