FROM python:3.8-slim
LABEL org.opencontainers.image.source="https://github.com/civo-learn/kubeflow-examples"

WORKDIR /app

COPY requirements.txt .

COPY app.py .

RUN apt-get update && \
    apt-get install -y wget build-essential cmake ninja-build libopenblas-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
RUN pip install llama-cpp-python

RUN wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q8_0.gguf -o llama3.gguf

# EXPOSE 8080

CMD ["python", "./app.py"]
