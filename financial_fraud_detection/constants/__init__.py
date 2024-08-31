import os
from datetime import date

DATABASE_NAME='financial_fraud_db'
COLLECTION_NAME='financial_fraud_collection'
COLLECTION_URL='mongodb+srv://jashsuke:jashsuke@cluster0.7jivr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

PIPELINE_NAME: str = "financial_fraud"
ARTIFACT_DIR: str = "artifact"
MONGODB_URL_KEY = "MONGODB_URL"

TARGET_COLUMN = "fraud_reported"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "fraud.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

MODEL_FILE_NAME = "model.pkl"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "financial_fraud_collection"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


