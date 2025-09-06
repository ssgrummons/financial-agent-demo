
# backend/src/app.py
import logging
import uuid
import json
from typing import Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import ChatRequest, ChatResponse, SessionRequest, NewSessionResponse
from graphs.chat_graph import ChatGraph
from services.session_service import SessionService
from config.settings import get_settings

# Configure logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

# Get module logger
logger = logging.getLogger(__name__)
logger.info("Logging initialized. %s log level is set to: %s (%d)", __name__, log_level_str, log_level)

# Global variables for dependency injection
chat_graph: ChatGraph = None
session_service: SessionService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global chat_graph, session_service
    
    # Startup
    logger.info("Starting GAgent Financial Advisor API...")
    settings = get_settings()
    
    # Initialize services
    session_service = SessionService()
    chat_graph = ChatGraph(
        provider=settings.model_provider,
        model=settings.model_name,
        verbose=settings.verbose,
        logprobs=settings.enable_logprobs,
        reasoning_effort=settings.reasoning_effort,
        max_tokens=settings.max_tokens,
    )
    
    logger.info("Services initialized successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down GAgent Financial Advisor API...")
    if session_service:
        await session_service.cleanup()

# Create FastAPI app
app = FastAPI(
    title="GAgent Financial Advisor API",
    description="Agentic AI Financial Advisor with streaming capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
def get_chat_graph() -> ChatGraph:
    """Dependency to get chat graph instance."""
    if chat_graph is None:
        raise HTTPException(status_code=500, detail="Chat graph not initialized")
    return chat_graph

def get_session_service() -> SessionService:
    """Dependency to get session service instance."""
    if session_service is None:
        raise HTTPException(status_code=500, detail="Session service not initialized")
    return session_service

# Routes
@app.post("/sessions", response_model=NewSessionResponse)
async def create_session(
    session_svc: SessionService = Depends(get_session_service)
) -> NewSessionResponse:
    """Create a new chat session."""
    try:
        session_id = await session_svc.create_session()
        logger.info(f"Created new session: {session_id}")
        
        return NewSessionResponse(
            session_id=session_id,
            message="Session created successfully"
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_svc: SessionService = Depends(get_session_service)
):
    """Delete a chat session."""
    try:
        await session_svc.delete_session(session_id)
        logger.info(f"Deleted session: {session_id}")
        return {"message": "Session deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

@app.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    session_svc: SessionService = Depends(get_session_service)
):
    """Get conversation history for a session."""
    try:
        history = await session_svc.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get history for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session history")

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    graph: ChatGraph = Depends(get_chat_graph),
    session_svc: SessionService = Depends(get_session_service)
):
    """Stream chat responses from the financial advisor agent."""
    try:
        # Validate session exists
        if not await session_svc.session_exists(request.session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Processing streaming chat for session: {request.session_id}")
        
        async def generate_response() -> AsyncGenerator[str, None]:
            try:
                # Get session configuration for graph
                config = await session_svc.get_session_config(request.session_id)
                
                # Prepare initial state
                initial_state = {
                    "messages": [{"role": "user", "content": request.user_prompt}],
                    "session_id": request.session_id,
                    "user_id": "demo_user",  # Single user for demo
                    "user_prompt": request.user_prompt,
                }
                
                # Track response content for session storage
                full_response = ""
                
                # Stream the graph execution
                async for chunk in graph.astream(initial_state, config):
                    chunk_data = await process_graph_chunk(chunk)
                    if chunk_data:
                        # Accumulate assistant responses
                        if chunk_data.get("type") == "assistant_response":
                            full_response += chunk_data.get("content", "")
                        
                        # Yield formatted chunk
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Store conversation in session
                await session_svc.add_message(request.session_id, "user", request.user_prompt)
                if full_response:
                    await session_svc.add_message(request.session_id, "assistant", full_response)
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done', 'session_id': request.session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in chat stream: {e}")
                error_data = {
                    "type": "error",
                    "content": f"An error occurred: {str(e)}",
                    "session_id": request.session_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process chat request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")

async def process_graph_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Process a chunk from the graph stream and format for frontend."""
    try:
        # Handle different types of chunks from LangGraph
        if "assistant" in chunk:
            # Assistant reasoning or response
            messages = chunk["assistant"].get("messages", [])
            if messages:
                last_message = messages[-1]
                content = getattr(last_message, 'content', str(last_message))
                
                return {
                    "type": "assistant_response",
                    "content": content,
                    "timestamp": None  # Add timestamp if needed
                }
        
        elif "tools" in chunk:
            # Tool execution
            tool_messages = chunk["tools"].get("messages", [])
            if tool_messages:
                return {
                    "type": "tool_execution",
                    "content": "🔧 Using financial tools...",
                    "details": str(tool_messages[-1]) if tool_messages else None
                }
        
        # Handle other chunk types as needed
        return {
            "type": "thinking",
            "content": "💭 Processing your request...",
        }
        
    except Exception as e:
        logger.error(f"Error processing graph chunk: {e}")
        return {
            "type": "error",
            "content": f"Error processing response: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "GAgent Financial Advisor API",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "GAgent Financial Advisor API",
        "version": "1.0.0",
        "description": "Agentic AI Financial Advisor with streaming capabilities",
        "endpoints": {
            "health": "/health",
            "create_session": "POST /sessions",
            "chat_stream": "POST /chat/stream",
            "session_history": "GET /sessions/{session_id}/history",
            "delete_session": "DELETE /sessions/{session_id}"
        }
    }