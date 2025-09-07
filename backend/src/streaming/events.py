"""
Enhanced event types and emitters for streaming responses with better LangChain integration.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, AsyncGenerator, List, Union
import json
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of event types for better type safety."""
    ASSISTANT_RESPONSE = "assistant_response"
    TOOL_EXECUTION = "tool_execution" 
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    ERROR = "error"
    COMPLETION = "completion"
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"


@dataclass
class StreamEvent:
    """Base class for all streaming events with enhanced functionality."""
    type: str
    content: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_id: Optional[int] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "content": self.content,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "sequence_id": self.sequence_id
        }

    def to_sse_format(self) -> str:
        """Convert event to Server-Sent Events format."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class ChainStartEvent(StreamEvent):
    """Event for starting a chain of operations."""
    chain_type: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.type:
            self.type = EventType.CHAIN_START.value
        if self.chain_type:
            self.metadata["chain_type"] = self.chain_type


@dataclass
class AssistantResponseEvent(StreamEvent):
    """Event for assistant text responses with reasoning tracking."""
    is_intermediate: bool = False
    reasoning_step: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.type:
            self.type = EventType.ASSISTANT_RESPONSE.value
        
        self.metadata.update({
            "is_intermediate": self.is_intermediate,
            "reasoning_step": self.reasoning_step
        })


@dataclass  
class ToolExecutionEvent(StreamEvent):
    """Event for tool execution notifications."""
    tool_name: Optional[str] = None  # Fixed: Made optional with default value
    execution_status: str = "starting"  # starting, running, completed, failed
    tool_input: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.type:
            self.type = EventType.TOOL_EXECUTION.value
            
        self.metadata.update({
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "execution_status": self.execution_status
        })


@dataclass
class ToolResultEvent(StreamEvent):
    """Event for tool execution results."""
    result: Any = None
    success: bool = True
    error_message: Optional[str] = None
    tool_name: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.type:
            self.type = EventType.TOOL_RESULT.value
            
        self.metadata.update({
            "tool_name": self.tool_name,
            "success": self.success,
            "error_message": self.error_message
        })
        
        # Store result in metadata for complex objects
        if isinstance(self.result, (dict, list)):
            self.metadata["result"] = self.result
        else:
            self.metadata["result"] = str(self.result)


@dataclass
class ThinkingEvent(StreamEvent):
    """Event for processing/thinking notifications."""
    thinking_type: str = "processing"  # processing, analyzing, planning, etc.
    
    def __post_init__(self):
        super().__post_init__()
        if not self.type:
            self.type = EventType.THINKING.value
        self.metadata["thinking_type"] = self.thinking_type


@dataclass
class ErrorEvent(StreamEvent):
    """Event for error notifications."""
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    recoverable: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if not self.type:
            self.type = EventType.ERROR.value
            
        self.metadata.update({
            "error_code": self.error_code,
            "error_type": self.error_type,
            "recoverable": self.recoverable
        })


class EventEmitter:
    """Enhanced event emitter with chain tracking and better LangChain integration."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.sequence_counter = 0
        self.active_chains: Dict[str, Any] = {}
        
    def _get_next_sequence_id(self) -> int:
        """Get the next sequence ID for ordering events."""
        self.sequence_counter += 1
        return self.sequence_counter
    
    async def emit(self, event: StreamEvent) -> AsyncGenerator[str, None]:
        """Emit a single event with automatic session and sequence handling."""
        if self.session_id and not event.session_id:
            event.session_id = self.session_id
            
        if event.sequence_id is None:
            event.sequence_id = self._get_next_sequence_id()
        
        logger.debug(f"Emitting event: {event.type} (seq: {event.sequence_id})")
        yield event.to_sse_format()

    async def emit_chain_start(self, content: str = "Starting analysis...", 
                              chain_type: str = "langchain", **kwargs) -> AsyncGenerator[str, None]:
        """Emit a chain start event."""
        event = ChainStartEvent(content=content, chain_type=chain_type, 
                               session_id=self.session_id, **kwargs)
        async for chunk in self.emit(event):
            yield chunk

    async def emit_thinking(self, content: str = "Processing your request...", 
                           thinking_type: str = "processing", **kwargs) -> AsyncGenerator[str, None]:
        """Emit a thinking/processing event."""
        event = ThinkingEvent(content=content, thinking_type=thinking_type, 
                             session_id=self.session_id, **kwargs)
        async for chunk in self.emit(event):
            yield chunk

    async def emit_assistant_response(self, content: str, is_intermediate: bool = False,
                                    reasoning_step: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Emit an assistant response event."""
        event = AssistantResponseEvent(
            content=content, 
            is_intermediate=is_intermediate,
            reasoning_step=reasoning_step,
            session_id=self.session_id, 
            **kwargs
        )
        async for chunk in self.emit(event):
            yield chunk

    async def emit_tool_execution(self, tool_name: str, content: str, 
                                 tool_input: Optional[Dict[str, Any]] = None,
                                 execution_status: str = "starting", **kwargs) -> AsyncGenerator[str, None]:
        """Emit a tool execution event."""
        event = ToolExecutionEvent(
            content=content, 
            tool_name=tool_name,
            tool_input=tool_input,
            execution_status=execution_status,
            session_id=self.session_id, 
            **kwargs
        )
        async for chunk in self.emit(event):
            yield chunk

    async def emit_tool_result(self, tool_name: str, result: Any, content: str = "",
                              success: bool = True, error_message: Optional[str] = None, 
                              **kwargs) -> AsyncGenerator[str, None]:
        """Emit a tool result event."""
        if not content:
            content = f"Tool {tool_name} {'completed successfully' if success else 'failed'}"
            
        event = ToolResultEvent(
            content=content,
            tool_name=tool_name,
            result=result,
            success=success,
            error_message=error_message,
            session_id=self.session_id,
            **kwargs
        )
        async for chunk in self.emit(event):
            yield chunk

    async def emit_error(self, content: str, error_code: Optional[str] = None,
                        error_type: Optional[str] = None, recoverable: bool = True, 
                        **kwargs) -> AsyncGenerator[str, None]:
        """Emit an error event."""
        event = ErrorEvent(
            content=content, 
            error_code=error_code,
            error_type=error_type,
            recoverable=recoverable,
            session_id=self.session_id, 
            **kwargs
        )
        async for chunk in self.emit(event):
            yield chunk

    async def emit_completion(self, content: str = "Request completed", **kwargs) -> AsyncGenerator[str, None]:
        """Emit a completion event."""
        event = StreamEvent(
            type=EventType.COMPLETION.value, 
            content=content, 
            session_id=self.session_id, 
            **kwargs
        )
        async for chunk in self.emit(event):
            yield chunk