FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends build-essential linux-libc-dev python3.12-dev python3.12-venv libasound2t64 libglib2.0-0t64 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /venv

ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/opt/Isaac-GR00T HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 GROOT_PATCH_MISTRAL=1 AWS_CONFIG_FILE=/opt/adamo-model-runtime/runtime/aws_config

COPY Isaac-GR00T/adamo_model/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY adamo-model-runtime /opt/adamo-model-runtime
COPY Isaac-GR00T /opt/Isaac-GR00T
RUN pip install --no-cache-dir --no-deps /opt/adamo-model-runtime /opt/Isaac-GR00T
