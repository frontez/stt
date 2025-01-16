# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    python3.9 \
    python3-distutils \
    ffmpeg \
    gcc \
    g++ \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies, explicit CUDA support for PyTorch
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install all other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python3", "app.py"]