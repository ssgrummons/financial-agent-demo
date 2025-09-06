# backend/src/app.py
import logging
import uuid
import os
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
from streaming.orchestrator import StreamingOrchestrator

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
    
    try:
        # Load and validate settings
        settings = get_settings()
        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  - Provider: {settings.assistant_config.provider}")
        logger.info(f"  - Model: {settings.assistant_config.model}")
        logger.info(f"  - Max Tokens: {settings.assistant_config.max_tokens}")
        logger.info(f"  - System Prompt: {len(settings.system_prompt)} characters")
        
        # Validate API keys early
        settings.validate_api_keys()
        logger.info("API key validation passed")
        
        # Initialize session service
        session_service = SessionService()
        logger.info("Session service initialized")
        
        # Initialize chat graph with proper settings mapping
        model_config = settings.get_model_config()
        chat_graph = ChatGraph(
            provider=model_config["provider"],
            model=model_config["model"],
            verbose=model_config["verbose"],
            logprobs=model_config["logprobs"],
            reasoning_effort=model_config["reasoning_effort"],
            max_tokens=model_config["max_tokens"],
        )
        logger.info("Chat graph initialized successfully")
        
        # Store settings in app state for access in routes
        app.state.settings = settings
        
        logger.info("✅ GAgent Financial Advisor API startup complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down GAgent Financial Advisor API...")
    try:
        if session_service:
            await session_service.cleanup()
            logger.info("Session service cleanup complete")
        
        logger.info("✅ GAgent Financial Advisor API shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

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
            # Create streaming orchestrator
            orchestrator = StreamingOrchestrator(session_id=request.session_id)
            
            # Get session configuration
            config = await session_svc.get_session_config(request.session_id)
            
            # Prepare initial state
            initial_state = {
                "messages": [{"role": "user", "content": request.user_prompt}],
                "session_id": request.session_id,
                "user_id": "demo_user",
                "user_prompt": request.user_prompt,
            }
            
            # Stream the graph execution - that's it!
            async for chunk in orchestrator.stream_graph_execution(graph, initial_state, config):
                yield chunk
            
            # Store conversation in session (using accumulated response)
            await session_svc.add_message(request.session_id, "user", request.user_prompt)
            full_response = orchestrator.get_accumulated_response()
            if full_response:
                await session_svc.add_message(request.session_id, "assistant", full_response)
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process chat request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")

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

if __name__ == "__main__":
    import uvicorn
    
    # Load settings for local development
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "app:app",  # module:app
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
    )