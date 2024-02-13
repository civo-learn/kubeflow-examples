FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

COPY app.py .

RUN apt-get update && \
    apt-get install -y wget && \
    pip install --no-cache-dir -r requirements.txt

RUN wget https://huggingface.co/TheBloke/Falcon-7B-Instruct-GGML/resolve/main/falcon-7b-instruct.ggccv1.q4_0.bin

# EXPOSE 8080

CMD ["python", "./app.py"]
