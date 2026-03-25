# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies (Updated for modern Debian)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your NexScan code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# The default command to run
CMD ["python", "app.py"]