# Base image with PyTorch + CUDA + cuDNN
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# ---- Install system dependencies (as root) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Create non-root user (for runtime only) ----
RUN groupadd -r vesselfm && useradd -m --no-log-init -r -g vesselfm vesselfm

# ---- Copy application code ----
WORKDIR /app
COPY vesselFM-main/vesselFM-main /app
COPY av_ct_best_cldice.pt /app/checkpoints/av_ct_best_cldice.pt
COPY README.md /app

# ---- Install Python dependencies into global env ----
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Optional: adjust ownership so runtime user can write logs, etc.
RUN chown -R vesselfm:vesselfm /app

# ---- Switch to non-root user for runtime ----
USER vesselfm
ENV PATH="/home/vesselfm/.local/bin:${PATH}"

# ---- Default runtime command ----
CMD ["python", "-m", "vesselfm.seg.eval_inference", "/input", "/output"]