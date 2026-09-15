import os
from operator import itemgetter

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

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

vectorstore = PineconeVectorStore(index=pc_index, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

context = ""

prompt_template = ChatPromptTemplate.from_template("""
        Answer the questions based only on the following context:
        
        {context}

        Question: {question}

        Provide a detailed answer:
    """)


def format_docs(docs):
    """Format retrived documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """Retrieval chain without LCEL"""

    # Step 1: Retrieve relevant documents from Pinecone
    docs = retriever.invoke(query)

    # Step 2: Format the retrieved documents into a single string
    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content


def retrieval_chain_with_lcel():
    """
    Create a retrieval chain with LCEL (LangChain Execution Language)
    Returns a chain that can be invoked with { "question": "..." }

    Advantages over non-LCEL approach:
        - Declarative and composable: Easy to chain operations with pipe operator
        - Built-in streaming: chain.stream() works out of the box
        - Built-in async: chain.ainvoke() and chain.astream() available
        - Batch processing: chain.batch() for multiple inputs
        - Type safety: Better integration with LangChain's types system
        - Less code: more consise and readable
        - Reusable: Chain can be saved, shared, and composed with other chains
        - Better debugging: LangChain provides better observability and debugging tools for LCEL chains
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=(itemgetter("question") | retriever | format_docs)
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving")

    query = "What is Pinecone in machine learning"

    # without context
    # result_raw = llm.invoke([HumanMessage(content=query)])

    # without lcel
    # result_raw = retrieval_chain_without_lcel(query)

    # with lcel
    result_raw = retrieval_chain_with_lcel().invoke({"question": query})

    print("Raw result:", result_raw)
