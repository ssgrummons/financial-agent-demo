from abc import ABC, abstractmethod
from typing import Optional, List, Any, AsyncGenerator
from pydantic_settings import BaseSettings
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import logging

# Configure logging
logger = logging.getLogger(__name__)
        
class AzureOpenAISettings(BaseSettings):
    """Settings for Azure OpenAI model configuration"""
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = None
    AZURE_OPENAI_API_VERSION: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    MAX_TOKENS: int = 4000  # Added default value
    TEMPERATURE: float = 0.7  # Added default value
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

class OllamaSettings(BaseSettings):
    """Settings for Ollama model configuration."""
    OLLAMA_MODEL: Optional[str] 
    MAX_TOKENS: int 
    TEMPERATURE: float 
    OLLAMA_HOST: str = "granite3.3"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
        
class GoogleSettings(BaseSettings):
    """Settings for Google Gemini model configuration."""
    GOOGLE_API_KEY: Optional[str]
    GOOGLE_MODEL: Optional[str] = "gemini-1.5-pro"  # Default model
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.7
    # Google-specific parameters with defaults
    TOP_K: Optional[int] = None
    TOP_P: Optional[float] = None
    SAFETY_SETTINGS: Optional[dict] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

class AnthropicSettings(BaseSettings):
    """Settings for Anthropic Claude model configuration."""
    ANTHROPIC_API_KEY: Optional[str]
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"  # Default model
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.7
    # Anthropic-specific parameters with defaults
    TOP_K: Optional[int] = None
    TOP_P: Optional[float] = None
    STOP_SEQUENCES: Optional[List[str]] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

class ModelFactory(ABC):
    """Abstract factory for creating chat models."""
    
    @abstractmethod
    def create_model(self, model_name: Optional[str] = None) -> BaseChatModel:
        """Create a chat model instance."""
        pass

class AzureOpenAIModelFactory(ModelFactory):
    """Factory for creating Azure OpenAI chat models."""
    
    def __init__(self, settings: Optional[AzureOpenAISettings] = None):
        """Initialize the factory with settings."""
        self.settings = settings or AzureOpenAISettings()
        
    def create_model(self, 
                     model_name: Optional[str] = None, 
                     verbose: Optional[bool] = True, 
                     streaming: Optional[bool] = False,
                     logprobs: Optional[bool] = False,
                     reasoning_effort: Optional[str] = 'minimal',
                     max_tokens: Optional[int] = None
                     ) -> BaseChatModel:
        """Create an Azure OpenAI chat model instance.
        
        Args:
            model_name: Optional model name to override the default from settings.
            verbose: Optional verbose parameter.
            streaming: Whether to enable streaming mode.
            
        Returns:
            A configured AzureChatOpenAI instance.
        """
        deployment_name = model_name or self.settings.AZURE_OPENAI_DEPLOYMENT_NAME
        max_tokens = max_tokens or self.settings.MAX_TOKENS
        logging.info(f"Creating Azure OpenAI model with deployment: {deployment_name}, streaming: {streaming}")
        
        # Create the model with streaming support
        model = AzureChatOpenAI(
            api_key=self.settings.AZURE_OPENAI_API_KEY,
            azure_deployment=deployment_name,
            api_version=self.settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT,
            temperature=self.settings.TEMPERATURE,
            max_tokens=self.settings.MAX_TOKENS,
            verbose=verbose,
            streaming=streaming,
            logprobs=logprobs,
            reasoning_effort=reasoning_effort
        )
        
        return model

class OllamaModelFactory(ModelFactory):
    """Factory for creating Ollama chat models."""
    
    def __init__(self, settings: Optional[OllamaSettings] = None):
        """Initialize the factory with settings."""
        self.settings = settings or OllamaSettings()
        ## Generic System Prompt
    
    def create_model(self, 
                     model_name: Optional[str] = None, 
                     verbose: Optional[bool] = True, 
                     streaming: Optional[bool] = False,
                     logprobs: Optional[bool] = False,
                     reasoning_effort: Optional[str] = 'minimal',
                     max_tokens: Optional[int] = None
                     ) -> BaseChatModel:
        """Create an Ollama chat model instance.
        
        Args:
            model_name: Optional model name to override the default from settings.
            
        Returns:
            A configured ChatOllama instance.
        """
        model=(model_name or self.settings.OLLAMA_MODEL).strip()
        max_tokens = max_tokens or self.settings.MAX_TOKENS
        logging.info(f"Calling Ollama model with name: {model}")

        return ChatOllama(
            model=model,
            base_url=self.settings.OLLAMA_HOST,
            temperature=self.settings.TEMPERATURE,
            num_predict=self.settings.MAX_TOKENS,
            verbose=verbose,
            disable_streaming=not streaming
        )

class GoogleModelFactory(ModelFactory):
    """Factory for creating Google Gemini chat models."""
    
    def __init__(self, settings: Optional[GoogleSettings] = None):
        """Initialize the factory with settings."""
        self.settings = settings or GoogleSettings()
    
    def create_model(self, 
                     model_name: Optional[str] = None, 
                     verbose: Optional[bool] = True, 
                     streaming: Optional[bool] = False,
                     logprobs: Optional[bool] = False,
                     reasoning_effort: Optional[str] = 'minimal',
                     max_tokens: Optional[int] = None
                     ) -> BaseChatModel:
        """Create a Google Gemini chat model instance.
        
        Args:
            model_name: Optional model name to override the default from settings.
            verbose: Optional verbose parameter.
            streaming: Whether to enable streaming mode.
            logprobs: Ignored for Google models.
            reasoning_effort: Ignored for Google models.
            max_tokens: Optional max tokens to override settings.
            
        Returns:
            A configured ChatGoogleGenerativeAI instance.
        """
        model = model_name or self.settings.GOOGLE_MODEL
        max_output_tokens = max_tokens or self.settings.MAX_TOKENS
        
        logging.info(f"Creating Google Gemini model: {model}, streaming: {streaming}")
        
        # Build generation config
        generation_config = {
            "temperature": self.settings.TEMPERATURE,
            "max_output_tokens": max_output_tokens,
        }
        
        # Add Google-specific parameters if they're set
        if self.settings.TOP_K is not None:
            generation_config["top_k"] = self.settings.TOP_K
        if self.settings.TOP_P is not None:
            generation_config["top_p"] = self.settings.TOP_P
            
        return ChatGoogleGenerativeAI(
            google_api_key=self.settings.GOOGLE_API_KEY,
            model=model,
            generation_config=generation_config,
            safety_settings=self.settings.SAFETY_SETTINGS,
            streaming=streaming,
            verbose=verbose
        )

class AnthropicModelFactory(ModelFactory):
    """Factory for creating Anthropic Claude chat models."""
    
    def __init__(self, settings: Optional[AnthropicSettings] = None):
        """Initialize the factory with settings."""
        self.settings = settings or AnthropicSettings()
    
    def create_model(self, 
                     model_name: Optional[str] = None, 
                     verbose: Optional[bool] = True, 
                     streaming: Optional[bool] = False,
                     logprobs: Optional[bool] = False,
                     reasoning_effort: Optional[str] = 'minimal',
                     max_tokens: Optional[int] = None
                     ) -> BaseChatModel:
        """Create an Anthropic Claude chat model instance.
        
        Args:
            model_name: Optional model name to override the default from settings.
            verbose: Optional verbose parameter.
            streaming: Whether to enable streaming mode.
            logprobs: Ignored for Anthropic models.
            reasoning_effort: Ignored for Anthropic models.
            max_tokens: Optional max tokens to override settings.
            
        Returns:
            A configured ChatAnthropic instance.
        """
        model = model_name or self.settings.ANTHROPIC_MODEL
        max_tokens_output = max_tokens or self.settings.MAX_TOKENS
        
        logging.info(f"Creating Anthropic Claude model: {model}, streaming: {streaming}")
        
        # Build kwargs for Anthropic-specific parameters
        kwargs = {
            "anthropic_api_key": self.settings.ANTHROPIC_API_KEY,
            "model": model,
            "temperature": self.settings.TEMPERATURE,
            "max_tokens": max_tokens_output,
            "streaming": streaming,
            "verbose": verbose
        }
        
        # Add Anthropic-specific parameters if they're set
        if self.settings.TOP_K is not None:
            kwargs["top_k"] = self.settings.TOP_K
        if self.settings.TOP_P is not None:
            kwargs["top_p"] = self.settings.TOP_P
        if self.settings.STOP_SEQUENCES is not None:
            kwargs["stop"] = self.settings.STOP_SEQUENCES
            
        return ChatAnthropic(**kwargs)

class ToolHandler(ABC):
    """Abstract base class for tool handling."""
    
    @abstractmethod
    def handle_tool_call(self, model: BaseChatModel, messages: List[BaseMessage], tools: List[BaseTool]) -> Any:
        """Handle a tool call and return the response."""
        pass

class StreamingToolHandler(ToolHandler):
    """Handler for streaming tool calls."""
    
    async def handle_tool_call(
        self,
        model: BaseChatModel,
        messages: List[BaseMessage],
        tools: List[BaseTool]
    ) -> AsyncGenerator[str, None]:
        """Handle streaming tool calls and yield responses."""
        try:
            logger.debug(f"Starting streaming with {len(messages)} messages")
            
            # Track if we've seen any tool calls
            has_tool_calls = False
            tool_call_buffer = []
            
            # Get the initial response stream
            async for chunk in model.astream(messages):
                logger.debug(f"Received chunk: {type(chunk)}, content: {getattr(chunk, 'content', 'No content')}")
                
                # Handle regular content chunks (no tool calls)
                if hasattr(chunk, 'content') and chunk.content:
                    if not has_tool_calls:  # Only yield content if we haven't seen tool calls
                        yield chunk.content
                    continue
                
                # Handle tool calls
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    has_tool_calls = True
                    tool_call_buffer.extend(chunk.tool_calls)
                    continue
                
                # Handle additional_kwargs for tool calls (alternative format)
                if hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs.get('tool_calls'):
                    has_tool_calls = True
                    tool_call_buffer.extend(chunk.additional_kwargs['tool_calls'])
                    continue
            
            # If we collected tool calls, execute them
            if has_tool_calls and tool_call_buffer:
                logger.debug(f"Processing {len(tool_call_buffer)} tool calls")
                
                for tool_call in tool_call_buffer:
                    # Find the tool to execute
                    tool_name = tool_call.get('function', {}).get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                    
                    # Handle different tool call formats
                    if not tool_name and hasattr(tool_call, 'function'):
                        tool_name = tool_call.function.name if hasattr(tool_call.function, 'name') else None
                    
                    if not tool_name:
                        logger.warning(f"Could not extract tool name from: {tool_call}")
                        continue
                    
                    tool = next((t for t in tools if t.name == tool_name), None)
                    
                    if tool:
                        # Get the arguments
                        args = {}
                        if isinstance(tool_call, dict):
                            args = tool_call.get('function', {}).get('arguments', {})
                            if isinstance(args, str):
                                import json
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    logger.error(f"Failed to parse tool arguments: {args}")
                                    continue
                        elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                            args = tool_call.function.arguments
                            if isinstance(args, str):
                                import json
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    logger.error(f"Failed to parse tool arguments: {args}")
                                    continue
                        
                        logger.debug(f"Executing tool {tool_name} with args: {args}")
                        
                        # Execute the tool
                        try:
                            tool_result = await tool.ainvoke(args) if hasattr(tool, 'ainvoke') else tool.invoke(args)
                            logger.debug(f"Tool result: {tool_result}")
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            yield f"Error executing tool {tool_name}: {str(e)}"
                            continue
                        
                        # Add the tool result to the conversation
                        messages.append(SystemMessage(content=f"""You have just run the tool '{tool_name}'. The tool provided the context you need to answer the user's question.
                                                      Now, respond with a detailed answer to the user's question based on this information.  
                                                      Your answer should be in natural language. Do not use any more tools.
                                                      Do not make up any information.  
                                                      Only use the context provided by the tool to answer the question accurately and truthfully.
                                                      Context: {tool_result}"""))
                
                # Stream the final response after tool execution
                logger.debug("Streaming final response after tool execution")
                async for final_chunk in model.astream(messages):
                    if hasattr(final_chunk, 'content') and final_chunk.content:
                        yield final_chunk.content
            
            # If no tool calls and no content was yielded, this might be an empty response
            if not has_tool_calls:
                logger.debug("No tool calls detected, checking if we need to yield a basic response")
                
        except Exception as e:
            logger.error(f"Error in streaming tool handler: {e}")
            yield f"Error: {str(e)}"

class NonStreamingToolHandler(ToolHandler):
    """Handler for non-streaming tool calls."""
    
    def handle_tool_call(
        self,
        model: BaseChatModel,
        messages: List[BaseMessage],
        tools: List[BaseTool]
    ) -> str:
        """Handle non-streaming tool calls and return the final response."""
        # Get the initial response from the model
        response = model.invoke(messages)

        # If there are tool calls, execute them and continue the conversation
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                # Find the tool to execute
                tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                tool = next((t for t in tools if t.name == tool_name), None)
                
                if tool:
                    # Get the arguments
                    args = tool_call.get('args') if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                    
                    # Execute the tool
                    tool_result = tool.invoke(args)
                    
                    # Add the tool result to the conversation
                    messages.append(SystemMessage(content=f"The result of the {tool_name} operation is: {tool_result}. Now, please respond with a detailed answer and explanation in natural language without calling any tools."))
            
            # Get the final response
            final_response = model.invoke(messages)
            return final_response.content if hasattr(final_response, 'content') else str(final_response)
        
        # If no tool calls, return the content from the initial response
        return response.content if hasattr(response, 'content') else str(response)

class ModelManager:
    """Manages chat model instances and their lifecycle."""
    
    def __init__(self, factory: Optional[ModelFactory] = None):
        """Initialize the model manager with a factory."""
        self.factory = factory or AzureOpenAIModelFactory()
        self.streaming_handler = StreamingToolHandler()
        self.non_streaming_handler = NonStreamingToolHandler()
    
    def get_model(self, model_name: Optional[str] = None, streaming: bool = False) -> BaseChatModel:
        """Get a chat model instance.
        
        Args:
            model_name: Optional model name to override the default.
            streaming: Whether to enable streaming mode.
            
        Returns:
            A configured chat model instance.
        """
        return self.factory.create_model(model_name, streaming=streaming)

    def bind_tools(self, model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
        """Bind tools to a chat model.
        
        Args:
            model: The chat model to bind tools to.
            tools: List of tools to bind.
            
        Returns:
            The chat model with tools bound.
        """
        return model.bind_tools(tools)

    async def handle_streaming_tool_call(
        self,
        model: BaseChatModel,
        messages: List[BaseMessage],
        tools: List[BaseTool]
    ) -> AsyncGenerator[str, None]:
        """Handle streaming tool calls."""
        async for chunk in self.streaming_handler.handle_tool_call(model, messages, tools):
            yield chunk

    def handle_tool_call(
        self,
        model: BaseChatModel,
        messages: List[BaseMessage],
        tools: List[BaseTool]
    ) -> str:
        """Handle non-streaming tool calls."""
        return self.non_streaming_handler.handle_tool_call(model, messages, tools)

# Create a singleton instance for backward compatibility
_model_manager = ModelManager()

def get_model(model_name: Optional[str] = None, streaming: bool = False) -> BaseChatModel:
    """Get a chat model instance (backward compatibility function).
    
    Args:
        model_name: Optional model name to override the default.
        streaming: Whether to enable streaming mode.
        
    Returns:
        A configured chat model instance.
    """
    return _model_manager.get_model(model_name, streaming=streaming)

def bind_tools(model: BaseChatModel, tools: List[BaseTool]) -> BaseChatModel:
    """Bind tools to a chat model (backward compatibility function).
    
    Args:
        model: The chat model to bind tools to.
        tools: List of tools to bind.
        
    Returns:
        The chat model with tools bound.
    """
    return _model_manager.bind_tools(model, tools)

def handle_tool_call(model: BaseChatModel, messages: List[BaseMessage], tools: List[BaseTool]) -> str:
    """Handle non-streaming tool calls (backward compatibility function).
    
    Args:
        model: The chat model instance.
        messages: List of messages in the conversation.
        tools: List of available tools.
        
    Returns:
        The final response from the model.
    """
    return _model_manager.handle_tool_call(model, messages, tools)

async def handle_streaming_tool_call(
    model: BaseChatModel,
    messages: List[BaseMessage],
    tools: List[BaseTool]
) -> AsyncGenerator[str, None]:
    """Handle streaming tool calls (backward compatibility function).
    
    Args:
        model: The chat model instance.
        messages: List of messages in the conversation.
        tools: List of available tools.
        
    Returns:
        An asynchronous generator yielding responses.
    """
    async for chunk in _model_manager.handle_streaming_tool_call(model, messages, tools):
        yield chunk