from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from nemoguardrails import LLMRails, RailsConfig
from langchain_core.runnables import chain, RunnableLambda, RunnableParallel, RunnablePassthrough

class NemoRails:
    def __init__(self, config: RailsConfig, llm: Optional[Any], generator_llm: Optional[Any] = None, verbose: bool = True, options: Optional[Dict[str, Any]] = None):
        """Initialize NemoRails with a given RailsConfig and optional LLM, verbosity, and options."""
        self.llm = llm
        self.generator_llm = generator_llm
        self.verbose = verbose
        self.options = options or {"rails": ["input"]}
        self.rails = LLMRails(config=config, llm=llm, verbose=verbose)
        if generator_llm:
            self.generate_or_exit = RunnableLambda(self._passthrough_or_exit)

    def _prepare_messages(self, _input: Any) -> List[Dict[str, Any]]:
        """Transforms input into the expected format for rails.generate."""
        messages = []
        if isinstance(_input, ChatPromptValue):
            for msg in _input.messages:
                if isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
        elif isinstance(_input, StringPromptValue):
            messages.append({"role": "user", "content": _input.text})
        elif isinstance(_input, dict):
            if "input" not in _input:
                raise ValueError("No `input` key found in the input dictionary.")
            user_input = _input["input"]
            if isinstance(user_input, str):
                messages.append({"role": "user", "content": user_input})
            elif isinstance(user_input, list):
                for msg in user_input:
                    assert "role" in msg and "content" in msg
                    messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                raise ValueError(f"Unsupported input type: {type(user_input).__name__}")
        else:
            raise ValueError(f"Can't handle input of type {type(_input).__name__}")
        return messages

    def _execute(self, input: Any) -> Any:
        """Executes rails.generate with the given input."""
        messages = self._prepare_messages(input)
        res = self.rails.generate(messages=messages, options=self.options)
        return res.response
    

    def _passthrough_or_exit(self, message_dict):
        """Processes messages and applies guardrails."""
        if message_dict["stop"]:
            return "I'm sorry, I can't respond to that."
        return self.generator_llm.invoke(message_dict["original"])
    
    def create_guardrail_chain(self) -> Any:
        """Returns a chainable function to process messages with LLMRails."""
        
        @chain
        def process_message(messages: List[Dict[str, Any]]) -> Any:
            return self._execute(messages)

        @chain
        def evaluate_response(output):
            """Evaluates the processed response and determines if the conversation should stop."""
            original = output["original"]
            processed = output["processed"]

            refusal_keywords = ["I'm sorry", "I can't", "I'm unable", "I won't", "I am sorry"]
            stop = any(
                any(keyword in message.get("content", "") for keyword in refusal_keywords)
                for message in processed
            )

            return {"original": original, "stop": stop}

        guardrail = RunnableParallel(
            original=RunnablePassthrough(),
            processed=process_message,
        ) | evaluate_response

        return guardrail
