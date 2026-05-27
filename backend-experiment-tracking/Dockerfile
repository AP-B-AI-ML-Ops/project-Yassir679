FROM python:3.13-slim

RUN pip install -U pip

# We add uvicorn explicitly to ensure MLflow uses the modern stack
RUN pip install psycopg2-binary mlflow uvicorn

# 1. Use --host 0.0.0.0 to listen on the network
# 2. Use --allowed-hosts "*" to satisfy the new security middleware
# 3. We removed --gunicorn-opts to avoid the conflict
CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "postgresql+psycopg2://mlflow:password@database/mlflow", \
    "--allowed-hosts", "*"]

# --host omdat mlflow anders op 127.0.0.1 runt, en de port niet kan exposen naar buiten
# --port omdat we zeker willen zijn dat mlflow runt op port 5000 (belangrijk voor scripts)
# psycopg2 om mlflow de database te laten accessen
# mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri postgresql+psycopg2://postgres:postgres@database/mlflow_db