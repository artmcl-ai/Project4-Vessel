# Base image with PyTorch + CUDA + cuDNN----
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# ---- Install system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
ffmpeg libsm6 libxext6 \
&& rm -rf /var/lib/apt/lists/*
# ---- Create non-root user ----
RUN groupadd -r VesselFM && useradd -m --no-log-init -r -g VesselFM VesselFM
# ---- Setup directories ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*
USER VesselFM
# ---- Add local user bin to PATH ----
ENV PATH="/home/VesselFM/.local/bin:${PATH}"
# ---- Copy application code ----
COPY --chown=VesselFM:VesselFM vesselFM-main/vesselFM-main /app
# ---- Install Python dependencies ----
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -e .
# ---- Default runtime command ----
CMD ["python", "-m", "vesselfm.seg.inference", "/input/1.nii.gz", "/output/1.nii.gz"]