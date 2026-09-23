import os
from typing import Any, Dict, List, cast
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

EMBEDIDNG_MODEL = os.environ["EMBEDDING_MODEL"]
LLM_MODEL = os.environ["LLM_MODEL"]
PINECONE_HOST = os.environ["PINECONE_HOST"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

# initialize embeddings
embeddings = OllamaEmbeddings(model=EMBEDIDNG_MODEL)

vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

model = init_chat_model(model=LLM_MODEL, model_provider="ollama")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrive relevant documentation to help answer user query about LangChain"""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # serialize retrieved documents
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer the user query using the retrieved documentation.

    Args:
        query: The user's questions

    Returns:
        Dictionary containing
            - answer: The generated answer
            - context: List of retrieved documents
    """

    # create agent with retrievel tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so"
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build the message list
    messages = [{"role": "user", "content": query}]

    # Invoke agent
    response = agent.invoke(cast(Any, {"messages": messages}))

    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == '__main__':
    result = run_llm("What are deep agents?")
    print("Answer:", result["answer"])
    print("Context:", result["context"])

