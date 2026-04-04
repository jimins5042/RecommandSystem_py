FROM python:3.11-slim

# Install minimal GLIB and OpenMP for OpenCV/PyTorch functionality
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Force CPU-only PyTorch installation (Significant space saver, 2GB+ reduction)
# This prevents installing massive CUDA libraries if GPU is not needed.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Exposure and start
EXPOSE 8000
CMD ["python", "main.py"]
