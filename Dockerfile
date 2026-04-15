FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# PyTorch CPU-only 먼저 설치 (CUDA 미포함, ~300MB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치 (ultralytics는 이미 설치된 torch 사용)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 절대 경로 환경변수
ENV MODEL_DIR=/app/model
ENV STATIC_DIR=/app/static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
