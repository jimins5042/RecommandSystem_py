FROM python:3.11-slim

# Install minimal GLIB and OpenMP for OpenCV/TensorFlow functionality
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install sorted dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Exposure and start
EXPOSE 8000
CMD ["python", "main.py"]
