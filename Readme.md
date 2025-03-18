# Nemo Integration with LangChain

This package provides **NeMo integration with LangChain**, addressing a key limitation in existing NeMo runnables. Specifically, it enables **configurable options for NeMo**, offering greater control over what is generated. Additionally, it allows flexibility in using **different LLMs for NeMo and text generation**, enhancing adaptability for various use cases.

## Features
- **Customizable NeMo options**: Control what NeMo generates.
- **Flexible LLM integration**: Use different LLMs for NeMo and response generation.
- **Guardrails integration**: Ensure AI-generated responses adhere to predefined constraints.

## Installation
Ensure you have the necessary dependencies installed:
```bash
pip install langchain_openai nemoguardrails langchain_guardrails
```

## Usage
Below is an example demonstrating how to integrate NeMo with LangChain and apply guardrails to control responses.

```python
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig
from langchain_guardrails import NemoRails
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Load RailsConfig (configure with the actual path to your NeMo guardrails config)
rails_config = RailsConfig.from_path("./tests/config")

# Instantiate NemoRails with configuration and LLM
nemorails = NemoRails(config=rails_config, llm=llm, options={"rails": ["input"]})

# Sample user input
user_input = [HumanMessage(content="Tell me about the Avengers movie")]

# Define a simple passthrough function with an exit condition
@chain
def passthrough_or_exit(message_dict):
    if message_dict["stop"]:
        return "I'm sorry, I can't respond to that."
    return llm.invoke(message_dict["original"])

# Create the guardrail processing chain
guardrail_chain = nemorails.create_guardrail_chain()
response_chain = ChatPromptTemplate.from_messages(user_input) | guardrail_chain | passthrough_or_exit

# Invoke the chain and generate response
response = response_chain.invoke({})

# Print the output
print(response)
```

## Notes
- The `RailsConfig` must be configured with the correct NeMo guardrails path.
- Modify `options` to customize how NeMo processes inputs.
- The `passthrough_or_exit` function ensures controlled responses.

This package enhances **NeMo's capabilities within LangChain**, making it more configurable and adaptable for diverse AI applications.

