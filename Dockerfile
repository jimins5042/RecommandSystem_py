FROM python:3.11-slim

# 이미지 처리에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# PyTorch CPU-only 먼저 설치 (CUDA 미포함, ~300MB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치 (ultralytics는 이미 설치된 torch 사용)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 절대 경로 환경변수 (.env보다 우선됨)
ENV MODEL_DIR=/app/model
ENV STATIC_DIR=/app/static

EXPOSE 8000

# /app 이 sys.path 에 포함된 상태에서 web.main:app 로드
# (config, backbones 모듈을 최상위에서 import 하기 위함)
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
