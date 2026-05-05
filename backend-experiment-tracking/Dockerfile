FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow>=2.12 psycopg2-binary boto3

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://mlflow:password@database:5432/mlflow \
    --default-artifact-root /mlflow/artifacts \
    --allowed-hosts "*"
