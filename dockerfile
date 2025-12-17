# 1st stage
FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# 2nd stage
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /install /usr/local

COPY server.py .
COPY frame_pb2.py .
COPY frame_pb2_grpc.py .

EXPOSE 8501
ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
