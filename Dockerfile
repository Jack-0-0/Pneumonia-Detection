FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

ADD server.py server.py
ADD pneu_model_v3 pneu_model_v3

RUN python server.py

EXPOSE 5000

CMD ["python", "server.py", "serve"]
