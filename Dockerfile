# Dockerfile for AskMyCompanyPro
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev     && pip install --no-cache-dir -r requirements.txt

CMD ["python", "askmycompany/src/app.py"]
