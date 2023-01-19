FROM python:3.8-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY configs/ configs/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# # Set environment variables for wandb API key and project name
# ENV WANDB_API_KEY=${WANDB_API_KEY}

ENTRYPOINT ["python", "-u", "src/models/train_LN_model.py"]