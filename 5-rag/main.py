import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

print("Initializing components...")

EMBEDIDNG_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5:9b")

PINECONE_HOST = os.environ.get("PINECONE_HOST", "https://your-pinecone-host.com")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "your-pinecone-api-key")
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]


embeddings = OllamaEmbeddings(model=EMBEDIDNG_MODEL)

llm = ChatOllama(model=LLM_MODEL)

pc = Pinecone(
    api_key=PINECONE_API_KEY,
)

pc_index = pc.Index(name=PINECONE_INDEX_NAME, host=PINECONE_HOST)

vectorstore = PineconeVectorStore(
    index=pc_index,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

context = ""

prompt_template = ChatPromptTemplate.from_template(
    """
        Answer the questions based only on the following context:
        
        {context}

        Question: {query}

        Provide a detailed answer:
    """
)


def format_docs(docs):
    """Format retrived documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)




if __name__ == "__main__":
    print("Retrieving")

    query = "What is Pinecone in machine learning"

    result_raw = llm.invoke(HumanMessage(content=query))

    print("Raw result:", result_raw)