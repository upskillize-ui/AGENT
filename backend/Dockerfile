FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY rag_requirements.txt .
RUN pip install --no-cache-dir -r rag_requirements.txt

COPY Db.py .
COPY rag_main.py .
COPY rag_pipeline.py .
COPY questions_generator.py .
COPY ca.pem .

RUN mkdir -p /app/chroma_store

EXPOSE 7860

CMD ["uvicorn", "rag_main:app", "--host", "0.0.0.0", "--port", "7860"]