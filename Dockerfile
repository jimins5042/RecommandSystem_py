FROM python:3.11-slim

# Install standard libraries for image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the absolute work directory as requested
WORKDIR /opt/vgg16/RecommandSystem_py

# Define an environment variable for absolute path reference if needed
ENV APP_HOME=/opt/vgg16/RecommandSystem_py

# Force CPU-only PyTorch (Save space)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
COPY requirements.txt .

# PyTorch CPU-only 먼저 설치 (CUDA 미포함, ~300MB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치 (ultralytics는 이미 설치된 torch 사용)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files to /opt/vgg16/RecommandSystem_py
COPY . .

# 절대 경로 환경변수
ENV MODEL_DIR=/app/model
ENV STATIC_DIR=/app/static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
