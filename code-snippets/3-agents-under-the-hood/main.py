import os;

from dotenv import load_dotenv
load_dotenv();

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "qwen3.5:9b"

# Tools

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product."""
    print(f"Looking up price for {product}")
    prices = {"laptop": 999.99, "phone": 599.99, "keyboard": 99.99}
    return prices.get(product, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price. Available tiers: bronze, silver, gold"""
    print(f"Applying discount of {discount_tier} to price {price}")
    discounts = {"bronze": 0.1, "silver": 0.2, "gold": 0.3}
    return price - (price * discounts.get(discount_tier, 0.0))


# Agent Loop

def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", base_url=os.getenv("LLM_URL"), temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. " 
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "- Check if product is present or not by looking up using get_product_price"
                ", if non zero value, then product is present"
                "- NEVER guess or assume any product price "
                "You MUST call get_product_price first to get the real price.\n"
                "- Only call apply_discount AFTER you have receieved "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number\n"
                "- NEVER calculate discounts yourself using the math. "
                "Always use the apply_discount tool.\n"
                "- If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n------ITERATION {iteration} ------")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # if no tool calls, print the final answer
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"    [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print(f"Max iterations reached.")
    return None


def main():
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(result)

if __name__ == "__main__":
    main()
