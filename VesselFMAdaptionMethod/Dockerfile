# Base image with PyTorch + CUDA + cuDNN
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# ---- Install system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    build-essential python3-dev \
 && rm -rf /var/lib/apt/lists/*

# ---- Create non-root user ----
RUN groupadd -r VesselFM && useradd -m --no-log-init -r -g VesselFM VesselFM

# ---- Switch to non-root user ----
USER VesselFM

# ---- Add local user bin to PATH ----
ENV PATH="/home/VesselFM/.local/bin:${PATH}"

# ---- Copy application code ----
# Assumes eval_inference.py and eval_preprocess_av_ct_nii.py are inside vesselFM-main/vesselFM-main
COPY --chown=VesselFM:VesselFM vesselFM-main/vesselFM-main /app

# ---- Install Python dependencies ----
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# ---- Default runtime command ----
# Expectation: /input contains raw CT NIfTI files; /output is where masks will be written.
CMD ["python", "eval_inference.py", "/input", "/output"]