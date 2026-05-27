FROM python:3.11-slim

RUN pip install -U pip setuptools wheel

WORKDIR /prefect

RUN pwd

# COPY ["./.prefectignore", "./prefect.yaml", "./"]

RUN pip install pendulum prefect --upgrade

RUN prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]
