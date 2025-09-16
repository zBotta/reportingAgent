# syntax=docker/dockerfile:1.7

########################
# Stage 1 — Fetch source
########################
FROM python:3.11-slim AS fetcher

ARG GIT_REPO=https://github.com/zBotta/reportingAgent.git

ARG GIT_REF=main   # branch or tag (for a specific commit, see notes below) 

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Fail early if repo arg not provided
RUN test -n "$GIT_REPO" || (echo "ERROR: GIT_REPO build-arg is required" && false)

# Pull only the specified ref (branch/tag) shallowly
RUN git clone --depth 1 --branch "${GIT_REF}" "${GIT_REPO}" /src

########################
# Stage 2 — Runtime
########################

# Use a small Python base
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/home/appuser/.cache/huggingface

# Optional: git can help resolve some Hugging Face repos
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and working dir
RUN useradd -m appuser
WORKDIR /reportAgent

# Install Python deps first for better layer caching
COPY --from=fetcher src/requirements.txt /reportAgent/requirements.txt
RUN pip install -r requirements.txt

COPY --from=fetcher src/projectSetup.py /reportAgent/projectSetup.py
COPY --from=fetcher src/entry_point.sh /reportAgent/entry_point.sh
# Copy your app code (root/app -> /app)
COPY --from=fetcher src/app/ /reportAgent/app/

# Create & own runtime dirs the app/entrypoint will use
RUN mkdir -p /home/appuser/.cache/huggingface /reportAgent/app/logs \
 && chown -R appuser:appuser /home/appuser /reportAgent/app/logs/

RUN chmod +x /reportAgent/entry_point.sh && chown -R appuser:appuser /reportAgent /reportAgent/entry_point.sh /home/appuser
RUN chown -R appuser:appuser /reportAgent/ /home/appuser

# Give rights to the app user in tmp dir (for downloading HF models)
ENV HF_HOME=/home/appuser/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/home/appuser/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers \
    PYTORCH_HUB_DIR=/home/appuser/.cache/torch \
    TMPDIR=/home/appuser/tmp

RUN mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PYTORCH_HUB_DIR" "$TMPDIR" \
 && chown -R appuser:appuser /home/appuser \
 && chmod 700 "$TMPDIR" \
 && chmod 1777 /tmp   # belt & suspenders: ensure /tmp is world-writable with sticky bit


# Drop root
USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/reportAgent/entry_point.sh"]