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

COPY . .

EXPOSE 3000

CMD ["python", "main.py"]
