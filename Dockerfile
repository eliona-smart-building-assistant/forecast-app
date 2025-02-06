FROM eliona/base-python:latest-3.11-alpine-eliona

WORKDIR /app

RUN apk update && apk add --no-cache \
    git \
    postgresql-dev \
    gcc \
    musl-dev

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV HYPERPARAMETER_SEARCH_PATH=/tmp/hyperparameter_search
RUN mkdir -p /tmp/hyperparameter_search

COPY . .

EXPOSE 3000

CMD ["python", "main.py"]
