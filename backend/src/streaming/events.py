"""
Event types and emitters for streaming responses.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, AsyncGenerator
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Base class for all streaming events."""
    type: str
    content: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "content": self.content,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }

    def to_sse_format(self) -> str:
        """Convert event to Server-Sent Events format."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class AssistantResponseEvent(StreamEvent):
    """Event for assistant text responses."""
    type: str = "assistant_response"


@dataclass
class ToolExecutionEvent(StreamEvent):
    """Event for tool execution notifications."""
    type: str = "tool_execution"
    tool_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        if self.tool_name:
            data["metadata"]["tool_name"] = self.tool_name
        return data


@dataclass
class ThinkingEvent(StreamEvent):
    """Event for processing/thinking notifications."""
    type: str = "thinking"


@dataclass
class ErrorEvent(StreamEvent):
    """Event for error notifications."""
    type: str = "error"
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        if self.error_code:
            data["metadata"]["error_code"] = self.error_code
        return data


@dataclass
class CompletionEvent(StreamEvent):
    """Event signaling completion of streaming."""
    type: str = "done"
    content: str = ""


class EventEmitter:
    """Handles emission of streaming events."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
    
    async def emit(self, event: StreamEvent) -> AsyncGenerator[str, None]:
        """Emit a single event."""
        if self.session_id and not event.session_id:
            event.session_id = self.session_id
        
        logger.debug(f"Emitting event: {event.type}")
        yield event.to_sse_format()
    
    async def emit_assistant_response(self, content: str, **kwargs) -> AsyncGenerator[str, None]:
        """Emit an assistant response event."""
        event = AssistantResponseEvent(content=content, session_id=self.session_id, **kwargs)
        async for chunk in self.emit(event):
            yield chunk
    
    async def emit_tool_execution(self, content: str, tool_name: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Emit a tool execution event."""
        event = ToolExecutionEvent(content=content, tool_name=tool_name, session_id=self.session_id, **kwargs)
        async for chunk in self.emit(event):
            yield chunk
    
    async def emit_thinking(self, content: str = "Processing your request...", **kwargs) -> AsyncGenerator[str, None]:
        """Emit a thinking/processing event."""
        event = ThinkingEvent(content=content, session_id=self.session_id, **kwargs)
        async for chunk in self.emit(event):
            yield chunk
    
    async def emit_error(self, content: str, error_code: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Emit an error event."""
        event = ErrorEvent(content=content, error_code=error_code, session_id=self.session_id, **kwargs)
        async for chunk in self.emit(event):
            yield chunk
    
    async def emit_completion(self, **kwargs) -> AsyncGenerator[str, None]:
        """Emit a completion event."""
        event = CompletionEvent(session_id=self.session_id, **kwargs)
        async for chunk in self.emit(event):
            yield chunk