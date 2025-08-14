# ai-rag-chatbot
Chatbot using LLMs with RAG


Environment & prerequisites:
Requires: Python 3.10+
API key: OPENAI_API_KEY (export via env or .env)


# Tools

Flask: (Tailwind CDN + fetch API)

RAG chain: retrieval + gpt-4o-mini answerer (with history awareness)

PyPDFLoader (community import path) reliably pulls text & page metadata. 

FAISS is fast, local, and easy to persist/restore. 

text-embedding-3-small is a strong default for RAG cost/quality balance. 

gpt-4o-mini is optimized for quick, accurate text QA with low latency. 

LangChain’s history-aware retriever + retrieval chain is the current recommended pattern over older “RetrievalQA” one-liners. 



# Prompting guidelines (what the assistant follows)

Grounding: Only answer from retrieved context. If missing, say you can’t find it.

Citations: Always include “Sources” with page numbers—users can spot-check policy text.

Style: Short, direct answers suitable for HR operations.

Safety: No personal/PII inference. Avoid speculative advice—stick to policy excerpts & summaries.

These are enforced in answer_prompt. You can further add a “tone” line if you need a corporate style.




# Run the code:

Put the PDF at ./the_nestle_hr_policy_pdf_2012.pdf.

python ingest.py (one-time, or whenever the document changes).

python app.py and open http://127.0.0.1:5000.

Ask, responses will be grounded and cited with page numbers.






# Testing checklist

Cold start: App boots without re-embedding (thanks to FAISS .load_local). 

Basic Q&A:

“what is the policy on employee relations?”

“How is paid leave defined?”

Follow-ups (history-aware):

Q1: “What’s the parental leave policy?”

Q2: “How long is it for fathers?” (should use history) 

Edge cases: Ask about something not in the document; bot should politely say so.

Latency: First request warms the model; subsequent answers should be fast (FAISS lookup is local).






9) Why these specific tools?








# Build & run locally


Make entrypoint.sh executable:

chmod +x entrypoint.sh


The Dockerfile image:
    Uses python:3.11-slim
    Installs deps once (layer caching)
    Adds a non-root user
    Includes a HEALTHCHECK
    Uses the entrypoint.sh above




## Option A — Single container

### Build the image
docker build -t nestle-hr-assistant:latest .

### Run it (mount a local volume for persistent FAISS index between restarts)
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY="the open ai key" \
  -e PORT=8080 \
  -v "$(pwd)/faiss_index_nestle_hr_2012:/app/faiss_index_nestle_hr_2012" \
  nestle-hr-assistant:latest


## Docker Compose (easier env & volume mgmt)

export OPENAI_API_KEY="the open ai key"
docker compose up --build



# Test
Quick sanity test after deploy:
- Visit the service URL → you should see the chat UI.
- Ask: “what is the policy on employee relations?”
    - You should get a grounded answer with “Sources” (page numbers).




# Deployment

Deployment tips (brief)

Container: Use Gunicorn (gunicorn -w 2 -b 0.0.0.0:8080 app:app) behind Nginx.

Secrets: Set OPENAI_API_KEY via platform secrets manager.

Scaling: Share the persisted FAISS dir as a read-only volume.

Monitoring: Log the generated “search query” if you want to debug retrieval quality.




## Google Cloud Run (serverless, easy)

Pros: Simple, autoscaling, HTTPS by default
Cons: Ephemeral filesystem (index won’t persist across cold starts)

Workarounds:
Accept first-request ingestion per new instance; or
Set minimum instances > 0 to reduce cold starts; or
(Advanced) Store/download FAISS index from GCS at startup.


If you want to avoid re-ingesting on cold starts, either bump --min-instances to 1, or modify entrypoint.sh to download a prebuilt index from Cloud Storage (and upload it during a separate one-off job).

## Steps:

### 1) Authenticate & set project
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT

### 2) Build & push container
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT/nestle-hr-assistant

### 3) Deploy
gcloud run deploy nestle-hr-assistant \
  --image gcr.io/YOUR_GCP_PROJECT/nestle-hr-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=YOUR_OPENAI_KEY,PORT=8080,INDEX_DIR=faiss_index_nestle_hr_2012,PDF_PATH=the_nestle_hr_policy_pdf_2012.pdf \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3



# Operational notes:

- Secrets: NEVER bake OPENAI_API_KEY into the image. Pass via env at deploy time.

- First-run ingestion: The container will build the FAISS index if missing. On platforms with ephemeral storage, you may see ingestion happen again on new instances.

- Performance: Increase WORKERS / THREADS via env if you expect concurrency.

- Scaling: RAG retrieval is local and fast; OpenAI latency dominates. I am using gpt-4o-mini for speed/cost.

- Monitoring: Add access logs at the reverse proxy; you can also wrap /chat to log timing and retrieved doc counts (avoid logging PII or full content).

- Upgrades: If you change the PDF, just redeploy; the index will be rebuilt automatically on the next container start.



# Future work:

OCR fallback (if your PDF is scanned): run OCR (e.g., Tesseract) to produce text first.

Multiple docs: Point the loader at a folder and re-ingest.

Synonyms/HR glossary: Prepend a short glossary for better query rewrite.

Access control: Add auth to the Flask app if hosting on an internal network.

Evaluation: Use small test suites of Q→expected page(s) to regression-test changes.
