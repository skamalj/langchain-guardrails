from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig
from langchain_guardrails import NemoRails
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Load RailsConfig (configure this with your actual path)
rails_config = RailsConfig.from_path("./tests/config")

# Instantiate NemoRails
nemorails = NemoRails(config=rails_config, llm=llm, generator_llm=llm,  options={"rails": ["input"]})

# Test input
test_input = [
    HumanMessage(content="tell a violent story")
]

@chain
def passthrough_or_exit(message_dict):
    if message_dict["stop"]:
        return "I'm sorry, I can't respond to that."
    return llm.invoke(message_dict["original"])

# Create the guardrail processing chain
guardrail_chain = nemorails.create_guardrail_chain()
res = ChatPromptTemplate.from_messages(test_input) 
chain = res | guardrail_chain | nemorails.generate_or_exit

# Invoke the guardrail chain
response = [chain.invoke({})]

# Print results
print(response)
