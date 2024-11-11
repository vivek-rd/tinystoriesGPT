FROM --platform=linux/amd64 nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y awscli && \
    apt-get install -y python3 python3-pip && \
    apt install -y python3.10-venv && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY requirements.txt .

RUN python3 -m venv env
RUN source env/bin/activate
RUN pip install -r requirements.txt

COPY start_training.sh .
RUN chmod +x start_training.sh
ENV PATH="/root/env/bin:$PATH"

ENTRYPOINT ["bash", "start_training.sh"]