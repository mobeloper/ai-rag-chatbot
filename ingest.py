import os
from pathlib import Path
from dotenv import load_dotenv,find_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    load_dotenv(find_dotenv())
    pdf_path = Path("./the_nestle_hr_policy_pdf_2012.pdf")
    assert pdf_path.exists(), f"PDF not found at {pdf_path.resolve()}"

    # 1) Load PDF (PyPDFLoader keeps page numbers in metadata)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()  # list[Document] with .page_content and .metadata['page']

    # 2) Split into overlapping chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # tune 600–1200 for PDFs
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(pages)

    # Add a friendly source tag to metadata
    for d in docs:
        d.metadata["source"] = "Nestlé HR Policy (2012)"
        # d.metadata['page'] is already present from PyPDFLoader

    # 3) Embed & index with FAISS
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(docs, embeddings)

    # 4) Persist the index to disk
    # We’re using save_local/load_local so you don’t re-embed every time you start the server.
    index_dir = "faiss_index_nestle_hr_2012"
    vectordb.save_local(index_dir)
    
    
    
    print(f"✅ Ingestion complete. Chunks: {len(docs)}. Index saved to: {index_dir}")

if __name__ == "__main__":
    main()
