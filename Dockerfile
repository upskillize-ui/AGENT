FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY backend/rag_requirements.txt .
RUN pip install --no-cache-dir -r rag_requirements.txt

COPY backend/Db.py .
COPY backend/rag_main.py .
COPY backend/rag_pipeline.py .
COPY backend/questions_generator.py .
COPY backend/chroma_mysql.py .
COPY backend/ca.pem .

EXPOSE 7860

CMD ["uvicorn", "rag_main:app", "--host", "0.0.0.0", "--port", "7860"]