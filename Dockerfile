FROM python:3.11.9

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libpq-dev \
    gcc \
    && apt-get clean

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
ENV HYPERPARAMETER_SEARCH_PATH=/tmp/hyperparameter_search

RUN mkdir -p /tmp/hyperparameter_search


COPY . .

EXPOSE 3000

CMD ["python", "main.py"]
