import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing score','reading score']
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline = Pipeline(steps=[
                ('SimpleImputer',SimpleImputer(strategy="median")),
                ('StandardScalar',StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("SimpleImputer",SimpleImputer(strategy='most_frequent')),
                ('OneHotEncoder',OneHotEncoder())
            ])

            logging.info(f'Categorical Columns: {categorical_columns}')
            logging.info(f'Numerical Features: {numerical_features}')

            preprocessor = ColumnTransformer([
                ('NumericalPipeline',num_pipeline,numerical_features),
                ('CategoricalPipeline',cat_pipeline,categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_transform(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("The train and test data set are read")
            logging.info('Obtaining preprocessing object')

            preprocessor = self.get_data_transformer_object()

            target_columns = 'math score'

            input_train_feature_df = train_df.drop(columns=target_columns,axis=1)
            train_target_feature_column = train_df[target_columns]

            input_test_feature_df = test_df.drop(columns=target_columns,axis=1)
            test_target_feature_column = test_df[target_columns]

            logging.info("Applying the preprocessing")

            input_feature_train_arr = preprocessor.fit_transform(input_train_feature_df)
            input_feature_test_arr = preprocessor.transform(input_test_feature_df)

            train_arr = np.c_[input_feature_train_arr, np.array(train_target_feature_column)]
            test_arr = np.c_[input_feature_test_arr, np.array(test_target_feature_column)]

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)