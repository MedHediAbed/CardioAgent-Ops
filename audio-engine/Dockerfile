FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch (the default wheel ships CUDA and weighs ~2 GB)
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt

COPY *.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
