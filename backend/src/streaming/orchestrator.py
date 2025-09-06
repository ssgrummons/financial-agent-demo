"""
Main streaming orchestrator that coordinates graph streaming.
"""
from typing import Dict, Any, AsyncGenerator, Optional
import logging
from .events import EventEmitter
from .processors import ProcessorRegistry

logger = logging.getLogger(__name__)


class StreamingOrchestrator:
    """Orchestrates streaming responses from graph execution."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.emitter = EventEmitter(session_id)
        self.processor_registry = ProcessorRegistry()
        self.response_accumulator = ResponseAccumulator()
    
    async def stream_graph_execution(
        self, 
        graph, 
        initial_state: Dict[str, Any], 
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream the execution of a graph with clean event handling.
        
        Args:
            graph: The graph instance to execute
            initial_state: Initial state for graph execution
            config: Optional configuration for graph execution
            
        Yields:
            Formatted SSE chunks for the frontend
        """
        try:
            logger.info(f"Starting streaming for session: {self.session_id}")
            
            # Stream graph execution
            async for chunk in graph.astream(initial_state, config):
                logger.debug(f"Received chunk: {list(chunk.keys())}")
                
                # Process chunk and emit events
                async for event_chunk in self.processor_registry.process_chunk(chunk, self.emitter):
                    # Track assistant responses for session storage
                    self.response_accumulator.process_event_chunk(event_chunk)
                    yield event_chunk
            
            # Emit completion event
            async for completion_chunk in self.emitter.emit_completion():
                yield completion_chunk
            
            logger.info(f"Streaming completed for session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error in streaming orchestrator: {e}")
            async for error_chunk in self.emitter.emit_error(f"An error occurred: {str(e)}"):
                yield error_chunk
    
    def get_accumulated_response(self) -> str:
        """Get the accumulated assistant response for session storage."""
        return self.response_accumulator.get_full_response()


class ResponseAccumulator:
    """Accumulates assistant responses for session storage."""
    
    def __init__(self):
        self.assistant_content = []
    
    def process_event_chunk(self, event_chunk: str):
        """Process an event chunk and accumulate assistant content."""
        try:
            # Parse the SSE format
            if event_chunk.startswith("data: "):
                import json
                data = json.loads(event_chunk[6:])  # Remove "data: " prefix
                
                if data.get("type") == "assistant_response":
                    content = data.get("content", "")
                    if content:
                        self.assistant_content.append(content)
                        
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Could not parse event chunk for accumulation: {e}")
    
    def get_full_response(self) -> str:
        """Get the full accumulated assistant response."""
        return "".join(self.assistant_content)
    
    def reset(self):
        """Reset the accumulator."""
        self.assistant_content.clear()