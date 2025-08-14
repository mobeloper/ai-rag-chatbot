#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden with env vars)
: "${INDEX_DIR:=faiss_index_nestle_hr_2012}"
: "${PDF_PATH:=the_nestle_hr_policy_pdf_2012.pdf}"
: "${PORT:=8080}"
: "${WORKERS:=2}"
: "${THREADS:=4}"
: "${TIMEOUT:=120}"

echo "[entrypoint] Starting Nestlé HR Assistant…"
echo "[entrypoint] INDEX_DIR=${INDEX_DIR}  PDF_PATH=${PDF_PATH}  PORT=${PORT}"

if [ ! -d "${INDEX_DIR}" ] || [ -z "$(ls -A "${INDEX_DIR}" 2>/dev/null || true)" ]; then
  echo "[entrypoint] Vector index not found. Running ingestion…"
  python ingest.py
  echo "[entrypoint] Ingestion complete."
else
  echo "[entrypoint] Existing vector index found. Skipping ingestion."
fi

# Start Gunicorn (production-grade WSGI server)
exec gunicorn \
  -w "${WORKERS}" \
  --threads "${THREADS}" \
  -b "0.0.0.0:${PORT}" \
  --timeout "${TIMEOUT}" \
  app:app
