import re
import inspect
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv()

import ollama
from langfuse import observe

MAX_ITERATIONS = 10
LLM_MODEL = "qwen3.5:9b"


@observe(as_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of the a product in the catalog"""
    print(f"    >> Executing get_product_price(product={product})")
    prices = {"laptop": 999.99, "mouse": 29.99, "keyboard": 79.99}
    return prices.get(product, 0.0)


@observe(as_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price"""
    print(f"    >> Executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_rates = {"gold": 0.2, "silver": 0.1, "bronze": 0.05}
    discount = discount_rates.get(discount_tier, 0)
    return price * (1 - discount)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_description(tools_dict):
    description = []
    for tool_name, tool_func in tools_dict.items():
        original_function = getattr(tool_func, '__wrapped__', tool_func)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        description.append(f"- {tool_name}{signature}: {docstring}")

    return "\n".join(description)


tool_description = get_tool_description(tools)
tool_names = ",".join(tools.keys())


react_prompt = f"""
    You are a helpful shopping assistant. You have access to a product catalog tool and a discount tool.
    STRICT RULES - you must follow these exactly
        - Check if product is present or not by looking up using get_product_price, if non zero value, then product is present
        - NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
        - Only call apply_discount AFTER you have receieved a price from get_product_price. Pass the exact price returned by get_product_price - do NOT pass a made-up number
        - NEVER calculate discounts yourself using the math. Always use the apply_discount tool.
        - If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.

    Answer the following questions as best as you can. You have access to the following tools:
    
    {tool_description}

    Use the following format:

    Question: input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action, as comma separated values
    Observation: the result of the action
    ... (This Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {{question}}
    Thought:"""


@observe(name="Ollama Chat", as_type="generation")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


@observe(name="Ollama Agent Loop", as_type="agent")
def run_agent(question: str):

    print(f"Question: {question}")
    print("="*60)

    prompt = react_prompt.format(question=question)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---Iteration : {iteration} ---")

        response = ollama_chat_traced(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"stop":["\nObservation"], "temperature": 0}
        )

        output = response.message.content

        print(f"LLM Output: {output}")

        print(f"    [Parsing] Looking for Fianl Answer in LLM output...")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output) # type: ignore

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "="*60)
            print(f"    [Parsed] Final Answer: {final_answer}")
            print("\n" + "="*60)
            print(f"Final Answer: {final_answer}")
            return final_answer


        print(f"    [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output) # type: ignore
        action_input_match = re.search(r"Action Input:\s*(.+)", output) # type: ignore

        if not action_match or not action_input_match:
            print(f"    [Error] Could not parse Action or Action Input from LLM output.")
            continue

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"    [Parsed] Tool Name: {tool_name}")
        print(f"    [Parsed] Tool Input: {tool_input_raw}")

        # split tool input
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"    [Tool Executing] {tool_name} with arguments: {args}...")

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            observation = str(tools[tool_name](*args))

        print(f"    [Tool Result] {tool_name} returned: {observation}")

        prompt += f"{output}\nObservation: {observation}"


def main():
    print("Hello from 4-raw-react-agent!")
    run_agent("What is the price of the laptop")


if __name__ == "__main__":
    main()
