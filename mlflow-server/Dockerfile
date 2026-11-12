FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==3.5.0

RUN pip install --no-cache-dir "SQLAlchemy<2.1"

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow-data/mlflow.db", "--allowed-hosts", "*"]
