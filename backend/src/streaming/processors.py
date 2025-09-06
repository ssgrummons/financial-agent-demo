"""
Chunk processors for different types of graph outputs.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List
import logging
from .events import StreamEvent, EventEmitter

logger = logging.getLogger(__name__)


class ChunkProcessor(ABC):
    """Abstract base class for processing graph chunks."""
    
    @abstractmethod
    async def process(self, chunk: Dict[str, Any], emitter: EventEmitter) -> AsyncGenerator[str, None]:
        """Process a chunk and emit appropriate events."""
        pass
    
    @abstractmethod
    def can_handle(self, chunk: Dict[str, Any]) -> bool:
        """Check if this processor can handle the given chunk."""
        pass


class AssistantChunkProcessor(ChunkProcessor):
    """Processes chunks from the assistant node."""
    
    def can_handle(self, chunk: Dict[str, Any]) -> bool:
        return "assistant" in chunk
    
    async def process(self, chunk: Dict[str, Any], emitter: EventEmitter) -> AsyncGenerator[str, None]:
        """Process assistant responses and emit streaming events."""
        assistant_data = chunk.get("assistant", {})
        messages = assistant_data.get("messages", [])
        
        if not messages:
            return
        
        last_message = messages[-1]
        
        # Check if this is a tool call or regular response
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Assistant is calling tools
            tool_names = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
                tool_names.append(tool_name)
            
            tools_text = ", ".join(tool_names)
            content = f"Using {tools_text} to help with your request..."
            
            async for event_chunk in emitter.emit_tool_execution(content, tool_name=tools_text):
                yield event_chunk
                
        else:
            # Regular assistant response - stream the content
            content = getattr(last_message, 'content', str(last_message))
            if content:
                async for event_chunk in emitter.emit_assistant_response(content):
                    yield event_chunk


class ToolsChunkProcessor(ChunkProcessor):
    """Processes chunks from tool execution nodes."""
    
    def can_handle(self, chunk: Dict[str, Any]) -> bool:
        return "tools" in chunk
    
    async def process(self, chunk: Dict[str, Any], emitter: EventEmitter) -> AsyncGenerator[str, None]:
        """Process tool execution results."""
        tools_data = chunk.get("tools", {})
        messages = tools_data.get("messages", [])
        
        if messages:
            # Tool execution completed, emit a status update
            content = "Tools executed successfully, generating response..."
            async for event_chunk in emitter.emit_thinking(content):
                yield event_chunk


class DefaultChunkProcessor(ChunkProcessor):
    """Default processor for unhandled chunk types."""
    
    def can_handle(self, chunk: Dict[str, Any]) -> bool:
        # This is the fallback processor
        return True
    
    async def process(self, chunk: Dict[str, Any], emitter: EventEmitter) -> AsyncGenerator[str, None]:
        """Process unknown chunk types with generic thinking message."""
        logger.debug(f"Processing unknown chunk type: {list(chunk.keys())}")
        async for event_chunk in emitter.emit_thinking("Processing..."):
            yield event_chunk


class ProcessorRegistry:
    """Registry for managing chunk processors."""
    
    def __init__(self):
        self.processors: List[ChunkProcessor] = []
        self._setup_default_processors()
    
    def _setup_default_processors(self):
        """Set up the default processors."""
        # Order matters - more specific processors first
        self.register(AssistantChunkProcessor())
        self.register(ToolsChunkProcessor())
        # Default processor last (catches everything)
        self.register(DefaultChunkProcessor())
    
    def register(self, processor: ChunkProcessor):
        """Register a new processor."""
        self.processors.append(processor)
    
    def get_processor(self, chunk: Dict[str, Any]) -> ChunkProcessor:
        """Get the appropriate processor for a chunk."""
        for processor in self.processors:
            if processor.can_handle(chunk):
                return processor
        
        # This should never happen since DefaultChunkProcessor catches everything
        raise ValueError(f"No processor found for chunk: {chunk}")
    
    async def process_chunk(self, chunk: Dict[str, Any], emitter: EventEmitter) -> AsyncGenerator[str, None]:
        """Process a chunk using the appropriate processor."""
        try:
            processor = self.get_processor(chunk)
            async for event_chunk in processor.process(chunk, emitter):
                yield event_chunk
        except Exception as e:
            logger.error(f"Error processing chunk {chunk}: {e}")
            async for event_chunk in emitter.emit_error(f"Error processing response: {str(e)}"):
                yield event_chunk