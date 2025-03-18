# Nemo Integration with LangChain

This package provides **NeMo integration with LangChain**, addressing a key limitation in existing NeMo runnables. Specifically, it enables **configurable options for NeMo**, offering greater control over what is generated. Additionally, it allows flexibility in using **different LLMs for NeMo and text generation**, enhancing adaptability for various use cases.

## Features
- **Customizable NeMo options**: Control what NeMo generates by utilizing generation options described [here](https://docs.nvidia.com/nemo/guardrails/latest/user-guides/advanced/generation-options.html)
- **Flexible LLM integration**: Use different LLMs for NeMo and response generation. Utilize tools binding as needed.
- **Guardrails integration**: Ensure AI-generated responses adhere to predefined constraints.

## Installation
Ensure you have the necessary dependencies installed:
```bash
pip install langchain_guardrails
```
## Input
Default behaviour is to execute only rails and generation is left for surround to handle i.e default for options is set to `{"rails": ["input"]}`

## Output
Output from guarrail is a dictionary:
```
{"original": <original message", "stop": True/False}
```
Utilize `stop` signal to execute next steps.

## Usage
Below is an example demonstrating how to integrate NeMo with LangChain and apply guardrails to control responses. Here we use **custom generator function** `passthrough_or_exit`

## Inbuilt generator function
This package has **inbuilt function for generation** - `generate_or_exit` -  which is enabled when `generator_llm` is passed.  You can use that as well to complete your generation.
```python
# Create the guardrail processing chain
guardrail_chain = nemorails.create_guardrail_chain()
res = ChatPromptTemplate.from_messages(test_input) 
chain = res | guardrail_chain | nemorails.generate_or_exit

# Invoke the guardrail chain
response = [chain.invoke({})]
```

## Custom generator function
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
- Tools messages are **not checked**

## Credits
This package uses code from [nemoguardrails](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/integrations/langchain/runnable_rails.py), licensed under [ Apache-2.0].

