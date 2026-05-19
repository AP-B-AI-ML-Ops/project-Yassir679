FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flow.py .
COPY start_pool_worker.sh .
RUN chmod +x start_pool_worker.sh

CMD ["./start_pool_worker.sh"]
