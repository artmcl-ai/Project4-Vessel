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
    find /workspace/tools -type f -name "*.py" -exec chmod 755 {} \;

ENV nnUNet_raw="/workspace/data/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/data/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/nnUNet_results"
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace
ENTRYPOINT ["python", "/workspace/process.py"]
CMD ["--help"]
