#!/usr/bin/env bash
set -euo pipefail

# Create .env on the fly using runtime env vars (provided by Spaces).
# Do NOT echo secrets to logs.
umask 177  # files created with 600 perms

ENVFILE=".env"
: > "$ENVFILE"  # truncate or create

# Write your variables (only those you actually need)
# HF_TOKEN is injected by Hugging Face Spaces (Settings → Variables and secrets)
if [[ -n "${HF_TOKEN:-}" ]]; then
  printf 'HF_TOKEN=%s\n' "$HF_TOKEN" >> "$ENVFILE"
fi

# Optional: other runtime vars you want in .env
printf 'STREAMLIT_SERVER_PORT=%s\n' "${STREAMLIT_SERVER_PORT:-8501}" >> "$ENVFILE"

# Launch your app
exec streamlit run /reportAgent/app/reportAgent.py --server.address=0.0.0.0 --server.port="${STREAMLIT_SERVER_PORT:-8501}"