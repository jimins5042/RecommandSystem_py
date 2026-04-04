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
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files to /opt/vgg16/RecommandSystem_py
COPY . .

# Launch application from the absolute path
EXPOSE 8000
CMD ["python", "main.py"]
