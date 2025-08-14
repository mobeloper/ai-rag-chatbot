import os
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---------- Config ----------

load_dotenv()
INDEX_DIR = "faiss_index_nestle_hr_2012"

# ---------- App ----------

app = Flask(__name__, static_folder="static")

# Load embeddings & vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = FAISS.load_local(
    INDEX_DIR, embeddings, allow_dangerous_deserialization=True
)

# LLM for both query-rewriting and answering
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # deterministic for policy QA

# --- Prompt 1: make the retriever history-aware (rewrite short search queries) ---
search_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search assistant for Nestlé HR policies. "
     "Given the chat history and latest user question, produce a concise search query "
     "to find the most relevant passages. If the question is already specific, return it unchanged."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Build history-aware retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
history_aware_retriever = create_history_aware_retriever(llm, retriever, search_prompt)

# --- Prompt 2: answer from retrieved context with strict grounding ---
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR assistant answering ONLY from the provided context. "
     "Context are excerpts from Nestlé's HR policy PDF. "
     "Requirements:\n"
     "1) If the answer is not in the context, say you cannot find it in the policy.\n"
     "2) Quote exact policy language sparingly when helpful.\n"
     "3) Always include a short 'Sources' section with page numbers.\n"
     "4) Keep answers clear, concise, and compliant.\n"),
    MessagesPlaceholder("chat_history"),
    ("human",
     "Question: {input}\n\n"
     "Context:\n{context}\n\n"
     "Provide the best possible answer now.")
])

# Combine: retrieved docs -> answer
doc_chain = create_stuff_documents_chain(llm, answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

# ---------- Helpers ----------

def lc_history_from_json(raw: List[Dict[str, str]]):
    """Convert [{role, content}...] to LangChain messages."""
    msgs = []
    for m in raw or []:
        if m.get("role") == "user":
            msgs.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") == "assistant":
            msgs.append(AIMessage(content=m.get("content", "")))
    return msgs

def sources_from_context(ctx_docs: List[Any]):
    """Extract minimal source info for UI."""
    out = []
    for d in ctx_docs:
        meta = d.metadata or {}
        out.append({
            "page": meta.get("page"),
            "source": meta.get("source", "Nestlé HR Policy (2012)"),
            "preview": (d.page_content or "")[:220].strip()
        })
    return out

# ---------- Routes ----------

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    user_input = payload.get("message", "").strip()
    history_raw = payload.get("history", [])
    chat_history = lc_history_from_json(history_raw)

    result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    answer = result.get("answer", "").strip()
    ctx = result.get("context", [])
    srcs = sources_from_context(ctx)
    return jsonify({"answer": answer, "sources": srcs})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
