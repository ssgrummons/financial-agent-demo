# GAgent Financial Advisor - Implementation Plan

## Project Overview
Building an agentic AI financial advisor using LangGraph, FastAPI, and Streamlit with Docker deployment.

## Architecture Decisions Made
- **Frontend**: Streamlit in separate container
- **Backend**: LangGraph + FastAPI for agent logic and API
- **Database**: SQLite with pre-populated transaction data
- **Deployment**: Docker Compose with images on Docker Hub
- **Model**: Gemini (Google) for cost-effectiveness
- **Memory**: LangGraph's built-in memory system
- **Streaming**: FastAPI → Streamlit real-time responses

## Phase 1: Basic Chat Foundation

### 1.1 State Design (`src/models/states.py`)
```python
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ChatState(TypedDict):
    """Core chat state for financial advisor"""
    ## Required for Assistant Node
    messages: Optional[List[AnyMessage]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    image_data: Optional[List[Dict]] = None
    
    ## Session Management
    session_id: str
    user_id: str
    user_profile: Optional[dict]  # Risk tolerance, portfolio, etc.
    conversation_context: Optional[dict]  # Recent topics, preferences
```

### 1.2 Model Factory (`src/clients/models.py`)
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
import os

class ModelFactory:
    @staticmethod
    def create_gemini_model(
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        api_key: Optional[str] = None
    ) -> ChatGoogleGenerativeAI:
        """Create Gemini model instance"""
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            streaming=True
        )
```

### 1.3 Assistant Node (`src/components/assistants.py`)
```python
from langchain_core.messages import SystemMessage
from ..models.states import ChatState
from ..clients.models import ModelFactory

def create_financial_assistant():
    """Create the financial advisor assistant"""
    
    system_prompt = """You are GAgent, a professional financial advisor AI assistant.
    
    Core Guidelines:
    - Always include appropriate disclaimers ("This is not personalized financial advice")
    - Be helpful but conservative with recommendations
    - Use tools when you need current market data or analysis
    - Maintain conversation context about user's portfolio and preferences
    
    Available tools:
    - fetch_stock_data: Get current stock prices and basic info
    - analyze_portfolio: Calculate portfolio metrics and risk analysis  
    - detect_fraud: Analyze transactions for suspicious patterns
    
    Remember: You are an educational tool, not a licensed financial advisor."""
    
    model = ModelFactory.create_gemini_model()
    
    def assistant_node(state: ChatState):
        """Main assistant reasoning node"""
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}
    
    return assistant_node
```

### 1.4 Basic Graph (`src/graphs/base_graph.py`)
```python
from langgraph.graph import StateGraph, END
from langgraph.memory import MemorySaver
from ..models.states import ChatState
from ..components.assistants import create_financial_assistant
from ..components.tools import create_tool_node

def create_chat_graph():
    """Create the basic chat graph"""
    
    # Initialize components
    assistant_node = create_financial_assistant()
    tool_node = create_tool_node()
    
    # Build graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", tool_node)
    
    # Define routing logic
    def should_continue(state: ChatState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return END
    
    # Add edges
    workflow.set_entry_point("assistant")
    workflow.add_conditional_edges("assistant", should_continue)
    workflow.add_edge("tools", "assistant")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Global graph instance
chat_graph = create_chat_graph()
```

### 1.5 FastAPI Routes (`src/app.py`)
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any
import json
import uuid
from .graphs.base_graph import chat_graph

app = FastAPI(title="GAgent Financial Advisor API")

class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    user_id: str = "john_doe"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Non-streaming chat endpoint"""
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Configure thread for this session
        config = {"configurable": {"thread_id": session_id}}
        
        # Add user message to state
        initial_state = {
            "messages": [{"role": "user", "content": request.message}],
            "session_id": session_id,
            "user_id": request.user_id
        }
        
        # Run the graph
        result = chat_graph.invoke(initial_state, config)
        
        # Extract AI response
        ai_message = result["messages"][-1].content
        
        return ChatResponse(response=ai_message, session_id=session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    session_id = request.session_id or str(uuid.uuid4())
    
    async def generate_response():
        try:
            config = {"configurable": {"thread_id": session_id}}
            
            initial_state = {
                "messages": [{"role": "user", "content": request.message}],
                "session_id": session_id,
                "user_id": request.user_id
            }
            
            # Stream the graph execution
            for chunk in chat_graph.stream(initial_state, config):
                # Format chunk for frontend
                if "assistant" in chunk:
                    content = chunk["assistant"]["messages"][-1].content
                    yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
                elif "tools" in chunk:
                    yield f"data: {json.dumps({'type': 'tool_call', 'data': 'Using tools...'})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/plain")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## Phase 2: Streamlit Frontend

### 2.1 Frontend Structure (`frontend/src/app.py`)
```python
import streamlit as st
import requests
import json
import uuid
from typing import Generator

# Configuration
BACKEND_URL = "http://backend:8000"  # Docker service name

def initialize_session():
    """Initialize session state"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

def stream_chat_response(message: str) -> Generator[str, None, None]:
    """Stream response from backend"""
    payload = {
        "message": message,
        "session_id": st.session_state.session_id
    }
    
    response = requests.post(
        f"{BACKEND_URL}/chat/stream",
        json=payload,
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode().replace('data: ', ''))
                if data['type'] == 'content':
                    yield data['data']
                elif data['type'] == 'tool_call':
                    yield f"\n🔧 {data['data']}\n"
                elif data['type'] == 'done':
                    break
            except json.JSONDecodeError:
                continue

def main():
    st.set_page_config(page_title="GAgent Financial Advisor", page_icon="💰")
    st.title("💰 GAgent Financial Advisor")
    st.caption("Your AI-powered financial assistant")
    
    initialize_session()
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your finances..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Stream AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in stream_chat_response(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
```

## Phase 3: Docker Deployment

### 3.1 Backend Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev

# Copy source code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.2 Docker Compose (`deployment/docker-compose.yml`)
```yaml
version: '3.8'

services:
  backend:
    image: yourusername/gagent-backend:latest
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ../data/financial_advisor.db:/app/data/financial_advisor.db:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    image: yourusername/gagent-frontend:latest
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend

networks:
  default:
    name: gagent-network
```

## Implementation Order

### Week 1: Foundation
1. **Day 1**: Set up state, model factory, basic assistant node
2. **Day 2**: Create basic graph with placeholder tools
3. **Day 3**: Build FastAPI endpoints (non-streaming first)
4. **Day 4**: Create Streamlit frontend
5. **Day 5**: Docker integration and deployment testing

### Week 2: Tools & Testing
1. **Day 1**: Stock data tool (yfinance integration)
2. **Day 2**: Portfolio analysis tool (basic calculations)
3. **Day 3**: Fraud detection tool (rule-based + simple ML)
4. **Day 4**: Database integration and testing
5. **Day 5**: Final testing, documentation, submission prep

## Key Testing Strategy
- **Unit tests**: Model factory, individual tools, state management
- **Integration tests**: Graph execution, API endpoints
- **E2E tests**: Full chat workflows via API
- **Manual testing**: Docker Compose deployment

## Questions to Address

1. **Environment Variables**: How do you want to handle the Google API key? `.env` file in repo or documentation?

2. **Database Location**: Should we mount the SQLite file or build it during container startup?

3. **Error Handling**: What level of error recovery do you want in the streaming responses?

4. **Tool Complexity**: Should the portfolio analysis tool use real calculations or simplified demos?

5. **Memory Persistence**: Do you want chat sessions to persist across container restarts?

Ready to start with Phase 1? I recommend beginning with the state design and model factory since those are the foundation for everything else.