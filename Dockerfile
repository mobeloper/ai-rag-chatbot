# syntax=docker/dockerfile:1

FROM python:3.11-slim

# System deps (curl for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps (use a single layer for speed; pin if you wish)
# NOTE: Keep this in sync with your local pip installs
RUN pip install --no-cache-dir -U \
    flask python-dotenv pydantic \
    langchain langchain-community langchain-openai \
    faiss-cpu pypdf tiktoken gunicorn

# Copy app code
# (Make sure the PDF is present at project root as per your earlier setup.)
COPY app.py ingest.py static/ the_nestle_hr_policy_pdf_2012.pdf ./
COPY entrypoint.sh .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Env defaults (override at runtime)
ENV INDEX_DIR=faiss_index_nestle_hr_2012
ENV PDF_PATH=the_nestle_hr_policy_pdf_2012.pdf
ENV PORT=8080
# OPENAI_API_KEY will be provided at runtime

EXPOSE 8080

# Simple healthcheck (ensure Flask/Gunicorn responds)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null || exit 1

ENTRYPOINT ["./entrypoint.sh"]
