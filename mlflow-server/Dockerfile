FROM python:3.11-slim

WORKDIR /app

# Same version as the clients (audio-engine, agent-orchestrator: mlflow-skinny==3.17.2)
RUN pip install --no-cache-dir mlflow==3.17.2

EXPOSE 5000

# Artifacts (models, figures) go in the volume too, otherwise they are lost when the container is rebuilt
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow-data/mlflow.db", "--artifacts-destination", "mlflow-data/artifacts", "--allowed-hosts", "*"]
