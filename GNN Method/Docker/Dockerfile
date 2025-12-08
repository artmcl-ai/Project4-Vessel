# ===============================
# Base image: PyTorch + CUDA + cuDNN
# ===============================
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ===============================
# Install system dependencies
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# ===============================
# Create directories (as root)
# ===============================
RUN mkdir -p /app /input /output \
 && chmod -R 755 /app /input /output

# ===============================
# Switch to working directory
# ===============================
WORKDIR /app

# ===============================
# Copy and install Python dependencies
# ===============================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy application code and models
# ===============================
COPY . /app

# ===============================
# Default runtime command
# (Container expects input & output to be mounted)
# ===============================
CMD ["python", "Full_Pipeline.py", "--input_dir", "/input", "--output_dir", "/output"]
