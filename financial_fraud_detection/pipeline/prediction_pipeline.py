import os
import sys

import numpy as np
import pandas as pd
from financial_fraud_detection.entity.config_entity import FraudPredictorConfig
from financial_fraud_detection.entity.s3_estimator import FraudEstimator
from financial_fraud_detection.exception import FFDMException
from financial_fraud_detection.logger import logging
from financial_fraud_detection.utils.main_utils import read_yaml_file
from pandas import DataFrame


class FraudData:
    def __init__(self,
                 months_as_customer,
                 age,
                 policy_deductable,
                 policy_annual_premium,
                 umbrella_limit,
                 capital_gains,
                 capital_loss,
                 incident_hour_of_the_day,
                 number_of_vehicles_involved,
                 bodily_injuries,
                 witnesses,
                 total_claim_amount,
                 injury_claim,
                 property_claim,
                 vehicle_claim,
                 policy_csl,
                 insured_sex,
                 insured_education_level,
                 insured_occupation,
                 insured_relationship,
                 incident_type,
                 collision_type,
                 incident_severity,
                 authorities_contacted,
                 property_damage,
                 police_report_available
                 ):
        """
        FraudData constructor: initializes all features of the dataset for fraud prediction.
        """
        try:
            # Numerical features
            self.months_as_customer = months_as_customer
            self.age = age
            self.policy_deductable = policy_deductable
            self.policy_annual_premium = policy_annual_premium
            self.umbrella_limit = umbrella_limit
            self.capital_gains = capital_gains
            self.capital_loss = capital_loss
            self.incident_hour_of_the_day = incident_hour_of_the_day
            self.number_of_vehicles_involved = number_of_vehicles_involved
            self.bodily_injuries = bodily_injuries
            self.witnesses = witnesses
            self.total_claim_amount = total_claim_amount
            self.injury_claim = injury_claim
            self.property_claim = property_claim
            self.vehicle_claim = vehicle_claim

            # Categorical features
            self.policy_csl = policy_csl
            self.insured_sex = insured_sex
            self.insured_education_level = insured_education_level
            self.insured_occupation = insured_occupation
            self.insured_relationship = insured_relationship
            self.incident_type = incident_type
            self.collision_type = collision_type
            self.incident_severity = incident_severity
            self.authorities_contacted = authorities_contacted
            self.property_damage = property_damage
            self.police_report_available = police_report_available

        except Exception as e:
            raise FFDMException(f"Error occurred in FraudData initialization: {e}")

    def get_fraud_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from FraudData class input.
        """
        try:
            fraud_input_dict = self.get_fraud_data_as_dict()
            return DataFrame(fraud_input_dict)
        
        except Exception as e:
            raise FFDMException(f"Error occurred while creating DataFrame: {e}")

    def get_fraud_data_as_dict(self):
        """
        This function returns a dictionary from FraudData class input.
        """
        try:
            input_data = {
                # Numerical features
                "months_as_customer": [self.months_as_customer],
                "age": [self.age],
                "policy_deductable": [self.policy_deductable],
                "policy_annual_premium": [self.policy_annual_premium],
                "umbrella_limit": [self.umbrella_limit],
                "capital_gains": [self.capital_gains],
                "capital_loss": [self.capital_loss],
                "incident_hour_of_the_day": [self.incident_hour_of_the_day],
                "number_of_vehicles_involved": [self.number_of_vehicles_involved],
                "bodily_injuries": [self.bodily_injuries],
                "witnesses": [self.witnesses],
                "total_claim_amount": [self.total_claim_amount],
                "injury_claim": [self.injury_claim],
                "property_claim": [self.property_claim],
                "vehicle_claim": [self.vehicle_claim],

                # Categorical features
                "policy_csl": [self.policy_csl],
                "insured_sex": [self.insured_sex],
                "insured_education_level": [self.insured_education_level],
                "insured_occupation": [self.insured_occupation],
                "insured_relationship": [self.insured_relationship],
                "incident_type": [self.incident_type],
                "collision_type": [self.collision_type],
                "incident_severity": [self.incident_severity],
                "authorities_contacted": [self.authorities_contacted],
                "property_damage": [self.property_damage],
                "police_report_available": [self.police_report_available]
            }

            return input_data

        except Exception as e:
            raise FFDMException(f"Error occurred while creating dictionary: {e}")

class FraudClassifier:
    def __init__(self,prediction_pipeline_config: FraudPredictorConfig = FraudPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise FFDMException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of USvisaClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            model = FraudEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise FFDMException(e, sys)