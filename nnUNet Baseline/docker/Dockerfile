FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

LABEL maintainer="zmh1225@bu.edu"
LABEL description="nnUNet 5-seed ensemble prediction with VesselFM dual-channel input"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements_docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Install nnUNet
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git /tmp/nnUNet && \
    cd /tmp/nnUNet && \
    pip install --no-cache-dir . && \
    cd / && \
    rm -rf /tmp/nnUNet

# Create directories with proper permissions
RUN mkdir -p /workspace/data/nnUNet_raw \
    /workspace/data/nnUNet_preprocessed \
    /workspace/nnUNet_results \
    /input \
    /output && \
    chmod -R 755 /workspace

# Copy preprocessed data and model checkpoints
COPY data/nnUNet_preprocessed /workspace/data/nnUNet_preprocessed
COPY nnUNet_results /workspace/nnUNet_results
COPY vesselfm_finetuned_monai.pt /workspace/vesselfm_finetuned_monai.pt

# Copy inference script and tools
COPY process.py /workspace/process.py
COPY tools/ /workspace/tools/
RUN chmod 755 /workspace/process.py && \
    find /workspace/tools -type f -name "*.py" -exec chmod 755 {} \; && \
    # Verify tools/probablity_maker.py exists (critical for new architecture)
    test -f /workspace/tools/probablity_maker.py || \
        (echo "ERROR: tools/probablity_maker.py not found" && exit 1)

# Set environment variables
ENV nnUNet_raw="/workspace/data/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/data/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/nnUNet_results"
ENV PYTHONUNBUFFERED=1

# Verify model files exist
RUN test -f /workspace/vesselfm_finetuned_monai.pt || \
        (echo "ERROR: VesselFM model not found" && exit 1) && \
    for seed in $(seq 1 5); do \
        CHECKPOINT="/workspace/nnUNet_results/Dataset003_vesselfm_good/seed_${seed}/fold_0/checkpoint_best.pth"; \
        test -f "$CHECKPOINT" || \
            (echo "ERROR: Missing checkpoint: $CHECKPOINT" && exit 1); \
    done

WORKDIR /workspace
ENTRYPOINT ["python", "/workspace/process.py"]
CMD ["--help"]
