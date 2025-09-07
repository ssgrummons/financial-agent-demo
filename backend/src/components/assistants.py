from typing import List, Dict, Optional, Any
import base64
from clients.models import AzureOpenAIModelFactory, OllamaModelFactory, GoogleModelFactory, AnthropicModelFactory
import logging
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from .message_strategies import MessageBuildStrategy

logger = logging.getLogger(__name__)

class MultiModalAsssitant:
    """
    Reusable multimodal AI assistant with configurable behavior
    """
    
    def __init__(self,
                 message_strategy: MessageBuildStrategy,
                 provider: str = "google", # future use with Generic Model Factory
                 model: str = "gemini-1.5-pro",
                 tools: List = None,
                 verbose: bool = True,
                 logprobs: bool = True,
                 streaming: bool = False,
                 reasoning_effort: str = 'minimal',
                 max_tokens: int = None):
        """
        """
        self.message_strategy = message_strategy
        self.provider = provider
        self.model = model
        self.verbose = verbose
        self.streaming = streaming
        self.logprobs = logprobs
        self.reasoning_effort = reasoning_effort # Possible Values 'minimal', 'low', 'medium', and 'high'
        self.max_tokens = max_tokens
        self.tools = tools
        self.chat_model = self._create_chat_model()

    def _create_chat_model(self):
        """
        Create and configure the chat model
        """
        factory_map: Dict[str, Any] = {
            "azure": AzureOpenAIModelFactory,
            "ollama": OllamaModelFactory,
            "google": GoogleModelFactory,
            "anthropic": AnthropicModelFactory
        }

        factory_cls = factory_map.get(self.provider)
        if factory_cls is None:
            raise ValueError(f"Unsupported provider: {self.provider!r}")

        factory = factory_cls()
        model = factory.create_model(
                model_name=self.model,
                verbose=self.verbose,
                streaming=self.streaming,
                logprobs=self.logprobs,
                reasoning_effort=self.reasoning_effort,
                max_tokens=self.max_tokens
            )
        if self.tools:
            return model.bind_tools(self.tools)
        return model
    
    def __call__(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """
        Main Node Function - Can be used directly in LangGraph
        """
        try:
            messages = state.get("messages", [])
            
            logger.debug(f"Assistant called with {len(messages)} existing messages")
            
            # Ensure system message exists
            if not messages or not isinstance(messages[0], SystemMessage):
                system_prompt = state.get("system_prompt", "You are a helpful AI Assistant.")
                messages = [SystemMessage(content=system_prompt)] + messages
                logger.debug("Added system message to conversation")
            
            # Always check for new user input and add it
            if self._has_new_user_input(state):
                new_user_messages = self._build_user_messages(state)
                messages.extend(new_user_messages)
                state["user_prompt"] = None
                logger.debug(f"Added {len(new_user_messages)} new user messages")
            
            # Invoke the model with current messages
            response = self.chat_model.invoke(messages)
            logger.info("Model response received successfully")

            # Update state with all messages including the response
            messages.append(response)
            state["messages"] = messages
            return state

        except Exception as e:
            logger.error(f"Error in MultiModalAssistant: {e}")
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(AIMessage(content=self._handle_error(e)))
            return state

    def _has_new_user_input(self, state: Dict[str, Any]) -> bool:
        """Check if there's new user input to process"""
        return bool(state.get("user_prompt") or state.get("image_data"))

    def _build_user_messages(self, state: Dict[str, Any]) -> List:
        """Build user messages from current state input"""
        messages = []
        user_prompt = state.get("user_prompt", "")
        
        # Handle multimodal input (images + text)
        if state.get("image_data"):
            processing_cfg = state.get("processing_config", {})
            detail_level = processing_cfg.get("detail_level", "low")
            image_messages = self.message_strategy.build_messages(
                image_data=state["image_data"],
                detail_level=detail_level,
                context=user_prompt
            )
            messages.extend(image_messages)
            logger.debug(f"Built {len(image_messages)} multimodal messages")
        
        # Handle text-only input
        elif user_prompt:
            messages.append(HumanMessage(content=user_prompt))
            logger.debug("Built text-only user message")
        
        return messages
    
    def _handle_error(self, error: Exception) -> str:
        """
        Clean Error Handling
        """
        error_str = str(error).lower()
        if "context_length_exceeded" in error_str:
            message = "The conversation has become too long. Please start a new session"
        elif "image" in error_str:
            message = "I couldn't process the images. Please try uploading them again."
        else:
            logger.error(f"Unexpected error: {error}")
            message = "I encountered an error processing your request. Please try again."
        
        return message