# Use Python 3.9+ as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PIL and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the src package first
COPY src/ /app/src/

# Copy the main script
COPY main.py .

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the script
ENTRYPOINT ["python", "main.py"] 