import os
from dotenv import load_dotenv
load_dotenv()

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

EMBEDIDNG_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5:9b")

PINECONE_HOST = os.environ.get("PINECONE_HOST", "https://your-pinecone-host.com")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "your-pinecone-api-key")
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

pc = Pinecone(
    api_key=PINECONE_API_KEY,
)

pc_index = pc.Index(name=PINECONE_INDEX_NAME, host=PINECONE_HOST)

if __name__ == "__main__":
    print("Ingesting...")

    loader = UnstructuredLoader(
        file_path="/Users/uttamrabari/Desktop/work/code-snippets/agentic-ai-engineering/5-rag/medium-blog.txt", 
        chunking_strategy="basic", 
        max_characters=1000000
    )

    document = loader.load()

    print("splitting...")

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    texts = text_splitter.split_documents(document)

    print(f"Created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model=EMBEDIDNG_MODEL)

    print("ingesting...")

    vectorstore = PineconeVectorStore(
        index=pc_index,
        embedding=embeddings
    )

    vectorstore.add_documents(texts)

    print("Ingestion complete!")