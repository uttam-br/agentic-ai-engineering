# System Deps
import asyncio
import os
import ssl
import certifi
from typing import Any, Dict, List

# Langchain Deps
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# User defined deps
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

# Load Environment Variables
from dotenv import load_dotenv

load_dotenv()


EMBEDIDNG_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5:9b")
PINECONE_HOST = os.environ.get("PINECONE_HOST", "https://your-pinecone-host.com")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "your-pinecone-api-key")
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]


embeddings = OllamaEmbeddings(model=EMBEDIDNG_MODEL)

vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(map_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the ingestion process"""
    log_header("DOCUMENTATION INGESTION STARTED")

    log_info(
        "** TavilyCrawl: Starting to Crawl documentation from https:/python.langchain.com/",
        Colors.CYAN,
    )

    # Crawl the documentation site using tavily
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 1,
            "extract_depth": "advanced"
        }
    )

    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]

    log_success("** TavilyCrawl: Crawling completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
