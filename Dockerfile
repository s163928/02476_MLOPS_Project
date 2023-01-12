FROM python:3.8-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Set environment variables for wandb API key and project name
ENV WANDB_API_KEY=23d639597d89bb9a9dabe3e76db7ae7cbc74641b

ENTRYPOINT ["python", "-u", "src/models/train_LN_model.py"]
