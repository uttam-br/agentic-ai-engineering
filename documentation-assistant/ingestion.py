# System Deps
import asyncio
import os
import urllib3
from typing import Any, Dict, List

import truststore
truststore.inject_into_ssl()

# 1. Mute the expected security warnings from bypassing SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Langchain Deps
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# User defined deps
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

# Load Environment Variablesa
from dotenv import load_dotenv

load_dotenv()

EMBEDIDNG_MODEL = os.environ["EMBEDDING_MODEL"]
LLM_MODEL = os.environ["LLM_MODEL"]
PINECONE_HOST = os.environ["PINECONE_HOST"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]


embeddings = OllamaEmbeddings(model=EMBEDIDNG_MODEL)

vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(map_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(docs: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously and index them into Pinecone"""
    log_header("VECTOR STORAGE PHASE")

    log_info(
        f"VectorStore Indexing: Preparing to add {len(docs)} documents to vector store"
    )

    # create batches
    batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

    log_info(
        f"VectorStore Indexing: Created {len(batches)} batches of size {batch_size} for processing."
    )

    async def process_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully indexed batch {batch_num}/{len(batches)} with {len(batch)} documents."
            )
        except Exception as e:
            log_error(
                f"VectorStore Indexing: Error indexing batch {batch_num + 1}/{len(batches)}: {str(e)}"
            )
            return False

        return True

    # process batches
    tasks = [process_batch(batch, i + 1) for i, batch in enumerate(batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully. Total batches: {len(batches)}"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Some batches failed to process. Successful batches: {successful}/{len(batches)}"
        )


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
            "extract_depth": "advanced",
        }
    )

    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]

    log_success("** TavilyCrawl: Crawling completed successfully.")

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")

    log_info(
        f"Text Splitting: Processing {len(all_docs)} documents for chunking. 4000 chunk size and 200 overlap",
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    split_docs = text_splitter.split_documents(all_docs)

    log_success(
        f"Text Splitting: Created {len(split_docs)} chunks from {len(all_docs)} documents."
    )

    await index_documents_async(split_docs, batch_size=500)

    log_header("PIPELINE COMPLETED SUCCESSFULLY")

    log_success("Documentation ingestion and indexing completed successfully.")

    log_info("Summary: ")
    log_info(f"     URL Mapped : {len(res['results'])} pages")
    log_info(f"     Documents   : {len(all_docs)} documents")
    log_info(f"     Chunks      : {len(split_docs)} chunks")


if __name__ == "__main__":
    asyncio.run(main())
