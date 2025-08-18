# syntax=docker/dockerfile:1.7

########################
# Stage 1 — Fetch source
########################
FROM python:3.11-slim AS fetcher

ARG GIT_REPO=https://github.com/zBotta/reportingAgent.git
# TODO: after tests change GIT_REF to main
ARG GIT_REF=dev   # branch or tag (for a specific commit, see notes below) 

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

RUN chmod +x /reportAgent/entry_point.sh && chown -R appuser:appuser /reportAgent /reportAgent/entry_point.sh /home/appuser
RUN chown -R appuser:appuser /reportAgent/ /home/appuser

# Drop root
USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/entry_point.sh"]