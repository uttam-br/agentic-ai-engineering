from typing import List

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel): 
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The sources used by the agent")


llm = ChatOllama(model="llama3.1")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke({
        "messages": HumanMessage(content="What is the weather in Tokyo?")
    })
    print(result)


if __name__ == "__main__":
    main()
