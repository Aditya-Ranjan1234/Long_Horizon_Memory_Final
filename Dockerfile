# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

# Set the working directory to /app
WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire project into the container
COPY . /app/

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Synchronize dependencies into a local virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# Set the virtual environment path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=false
ENV UV_LINK_MODE=copy

# Health check - use 127.0.0.1 to ensure it's internal
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:7860/health || exit 1

# Run the app directly from the root for Hugging Face Spaces compatibility
# Using the venv's python to ensure all dependencies are found
CMD ["/app/.venv/bin/python", "app.py"]
