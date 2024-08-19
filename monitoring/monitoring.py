import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset,
)
import psycopg
import os
import mlflow
from model_api.E2E_test import preprocess


def get_model():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:4566"
    os.environ["AWS_ACCESS_KEY_ID"] = "localstack"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"

    model_uri = "s3://game-behaviour-model/v1/"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=password", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=password"
        ) as conn:
            create_table_statement = """
            CREATE TABLE IF NOT EXISTS evidently_reports (
                report_type VARCHAR(50),
                report_data JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
            conn.execute(create_table_statement)


# Ensure the database is prepared
prep_db()
model = get_model()

# Load your CSV data
data = pd.read_csv("../data/online_gaming_behavior_dataset.csv")

data = preprocess(data)

data["Predictions"] = model.predict(data.drop(columns=["EngagementLevel"]))

limit = int(len(data) * 0.75)
data = data[:limit]
reference = data[limit:]

# Define column mapping if necessary
column_mapping = ColumnMapping(
    target="EngagementLevel",  # Assuming 'EngagementLevel' is the target variable
    prediction="Predictions",  # If you don't have predictions yet
)


# Create reports
data_quality_report = Report(metrics=[DataQualityPreset()])
data_drift_report = Report(metrics=[DataDriftPreset()])
# If you have model predictions:
model_performance_report = Report(metrics=[ClassificationPreset()])

# Run the reports
data_quality_report.run(
    current_data=data, column_mapping=column_mapping, reference_data=reference
)
data_drift_report.run(
    current_data=data, reference_data=reference, column_mapping=column_mapping
)
model_performance_report.run(
    current_data=data, column_mapping=column_mapping, reference_data=reference
)

# Get the JSON output
data_quality_json = data_quality_report.json()
data_drift_json = data_drift_report.json()
model_performance_json = model_performance_report.json()


# Function to save reports in the database
def save_report_to_db(report_type, report_json):
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=password",
        autocommit=True,
    ) as conn:
        conn.execute(
            "INSERT INTO evidently_reports (report_type, report_data) VALUES (%s, %s)",
            (report_type, report_json),
        )


# Save the reports to the database
save_report_to_db("Data Quality", data_quality_json)
save_report_to_db("Data Drift", data_drift_json)
save_report_to_db("Model Performance", model_performance_json)
