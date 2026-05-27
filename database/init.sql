-- Create a separate database for Prefect server (MLflow uses the 'mlflow' database, auto-created via POSTGRES_DB env var)
CREATE DATABASE prefect;

-- Create a separate database for Evidently monitoring metrics
CREATE DATABASE metrics;

