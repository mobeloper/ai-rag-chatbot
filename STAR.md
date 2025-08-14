Situation → Task → Action → Result (STAR)

Situation. 
Nestlé’s HR team needs quicker, consistent answers from internal HR documents. Manually searching a long PDF is slow and error-prone.

Task. 
Build a conversational chatbot that (1) ingests ./the_nestle_hr_policy_pdf_2012.pdf, (2) turns it into searchable vectors, (3) retrieves the best passages, and (4) uses an OpenAI model to answer questions with citations—all behind a clean Flask UI.

Action.
Stand up a Python environment and API keys.
Load & chunk the PDF (LangChain PyPDFLoader).
Embed chunks (OpenAI embeddings) and index them in FAISS.
Create a history-aware RAG chain with LangChain + gpt-4o-mini.
Expose a modern chat UI in Flask.
Add good prompting, citations, and guardrails.

Result. 
A fast, accurate, and auditable HR assistant that reduces time-to-answer and improves consistency, while keeping a simple deployment pathway (local first, then container/cloud).

