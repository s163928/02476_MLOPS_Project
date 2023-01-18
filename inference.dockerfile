FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY models/ models/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install --upgrade google-cloud-storage

ENV PORT=8080

CMD exec uvicorn src.api.inference:app --port ${PORT} --host 0.0.0.0 --workers 1
