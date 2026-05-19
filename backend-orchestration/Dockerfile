FROM python:3.11-slim

RUN pip install --no-cache-dir "prefect>=3.0" "asyncpg>=0.29"

EXPOSE 4200

CMD prefect server start --host 0.0.0.0 --port 4200
