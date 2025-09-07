"""
Central test configuration and fixtures for the GAgent Financial Advisor.
This file provides reusable test fixtures and utilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import sys
import os

# Simulate the Docker environment where we cd into the src directory
# Add both the src directory AND change working directory context
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

# Also add the parent directory so relative imports work
parent_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_path)

# Change the working directory for imports to match Docker behavior
original_cwd = os.getcwd()
os.chdir(src_path)

# Now import - this will work because we're "in" the src directory
from graphs.chat_graph import ChatGraph
from models.states import ChatState  
from services.session_service import SessionService

# Change back to original directory after imports
os.chdir(original_cwd)


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# MOCK FIXTURES - For fast, isolated testing
# ============================================================================

@pytest.fixture
def mock_assistant():
    """Mock MultiModalAssistant for testing"""
    assistant = Mock()
    
    # Mock invoke method for sync calls
    mock_response = Mock()
    mock_response.content = "Mocked assistant response"
    mock_response.tool_calls = []
    assistant.invoke.return_value = {"messages": [mock_response]}
    
    # Mock async invoke if needed
    assistant.ainvoke = AsyncMock(return_value={"messages": [mock_response]})
    
    return assistant


@pytest.fixture
def mock_tools():
    """Mock finance tools for testing"""
    tools = []
    
    # Stock price tool
    stock_tool = Mock()
    stock_tool.name = "get_stock_price"
    stock_tool.invoke.return_value = {
        "symbol": "AAPL",
        "price": 150.00,
        "currency": "USD",
        "timestamp": "2025-01-01T10:00:00Z"
    }
    tools.append(stock_tool)
    
    # Portfolio analysis tool
    portfolio_tool = Mock()
    portfolio_tool.name = "analyze_portfolio"
    portfolio_tool.invoke.return_value = {
        "total_value": 25000.00,
        "expected_return": 0.08,
        "risk_score": "moderate",
        "diversification_score": 0.7
    }
    tools.append(portfolio_tool)
    
    # Fraud detection tool
    fraud_tool = Mock()
    fraud_tool.name = "detect_fraud"
    fraud_tool.invoke.return_value = {
        "is_suspicious": False,
        "risk_score": 0.2,
        "factors": ["normal_amount", "known_account"]
    }
    tools.append(fraud_tool)
    
    return tools


@pytest.fixture
def mock_finance_tools_factory():
    """Mock FinanceTools factory"""
    factory = Mock()
    tools_instance = Mock()
    
    # Mock the load_tools method
    tools_instance.load_tools.return_value = [
        Mock(name="get_stock_price"),
        Mock(name="analyze_portfolio"),
        Mock(name="detect_fraud")
    ]
    
    factory.return_value = tools_instance
    return factory


@pytest.fixture
def mock_message_strategy():
    """Mock message strategy"""
    strategy = Mock()
    strategy.format_message.return_value = "Formatted message"
    return strategy


# ============================================================================
# STATE FIXTURES - For creating test states
# ============================================================================

@pytest.fixture
def base_chat_state():
    """Basic chat state for testing"""
    return {
        "session_id": "test_session_123",
        "user_id": "test_user",
        "user_prompt": "What's the current price of AAPL?",
        "messages": [],
        "system_prompt": "You are a helpful financial advisor."
    }


@pytest.fixture
def chat_state_with_messages(base_chat_state):
    """Chat state with conversation history"""
    mock_user_msg = Mock()
    mock_user_msg.content = "Hello, I'm interested in investing"
    mock_user_msg.type = "user"
    
    mock_assistant_msg = Mock()
    mock_assistant_msg.content = "Hello! I'd be happy to help with your investment questions."
    mock_assistant_msg.type = "assistant"
    mock_assistant_msg.tool_calls = []
    
    base_chat_state["messages"] = [mock_user_msg, mock_assistant_msg]
    return ChatState(**base_chat_state)


# ============================================================================
# GRAPH FIXTURES - For testing the ChatGraph
# ============================================================================

@pytest.fixture
def base_graph_config():
    """Standard configuration for ChatGraph"""
    return {
        "provider": "openai",
        "model": "gpt-4",
        "verbose": False,
        "logprobs": False,
        "reasoning_effort": "medium",
        "max_tokens": 1000
    }


@pytest.fixture
def production_chat_graph(base_graph_config):
    """Production ChatGraph instance - uses real dependencies"""
    with patch('components.tools.FinanceTools') as mock_finance_tools_class:
        # Mock the FinanceTools class to avoid real API calls
        mock_instance = Mock()
        mock_instance.load_tools.return_value = [Mock(), Mock(), Mock()]
        mock_finance_tools_class.return_value = mock_instance
        
        graph = ChatGraph(**base_graph_config)
        yield graph


@pytest.fixture 
def mocked_chat_graph(base_graph_config, mock_assistant, mock_tools, mock_message_strategy):
    """Fully mocked ChatGraph for fast testing"""
    graph = ChatGraph(
        **base_graph_config,
        assistant=mock_assistant,
        tools=mock_tools,
        message_strategy=mock_message_strategy
    )
    return graph


# ============================================================================
# SESSION SERVICE FIXTURES
# ============================================================================

@pytest.fixture
async def session_service():
    """Session service for testing"""
    service = SessionService()
    yield service
    # Cleanup after test
    await service.cleanup()


@pytest.fixture
async def session_with_history(session_service):
    """Session service with a session that has conversation history"""
    session_id = await session_service.create_session()
    
    # Add some conversation history
    await session_service.add_message(session_id, "user", "Hello")
    await session_service.add_message(session_id, "assistant", "Hi there!")
    await session_service.add_message(session_id, "user", "What's AAPL trading at?")
    
    return session_service, session_id


# ============================================================================
# API TESTING FIXTURES  
# ============================================================================

@pytest.fixture
def mock_app_dependencies():
    """Mock all app dependencies for API testing"""
    with patch('app.chat_graph') as mock_graph, \
         patch('app.session_service') as mock_session_svc:
        
        # Setup mock graph
        mock_graph.build_graph.return_value.compile.return_value = Mock()
        
        # Setup mock session service
        mock_session_svc.create_session = AsyncMock(return_value="test_session")
        mock_session_svc.session_exists = AsyncMock(return_value=True)
        mock_session_svc.get_session_config = AsyncMock(return_value={})
        mock_session_svc.add_message = AsyncMock()
        
        yield mock_graph, mock_session_svc


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_user_input_state(user_prompt: str, **kwargs) -> ChatState:
    """Helper to create state with user input"""
    base_state = {
        "session_id": "test_session",
        "user_id": "test_user",
        "user_prompt": user_prompt,
        "messages": [],
        "system_prompt": "You are a financial advisor."
    }
    base_state.update(kwargs)
    return ChatState(**base_state)


def create_tool_call_message(tool_name: str, args: Dict[str, Any]):
    """Helper to create a message with tool calls"""
    message = Mock()
    message.tool_calls = [{"name": tool_name, "args": args}]
    message.content = f"I'll use the {tool_name} tool to help you."
    return message


def create_tool_response_message(tool_name: str, result: Dict[str, Any]):
    """Helper to create a tool response message"""
    message = Mock()
    message.content = f"Tool {tool_name} returned: {result}"
    message.type = "tool"
    message.tool_calls = []
    return message


@asynccontextmanager
async def temporary_session(session_service: SessionService):
    """Context manager for temporary test sessions"""
    session_id = await session_service.create_session()
    try:
        yield session_id
    finally:
        try:
            await session_service.delete_session(session_id)
        except:
            pass  # Ignore cleanup errors in tests


# ============================================================================
# TEST DATA
# ============================================================================

# Financial queries for testing
FINANCIAL_TEST_QUERIES = [
    {
        "query": "What's the current price of AAPL stock?",
        "category": "stock_query",
        "should_use_tools": True,
        "expected_tools": ["get_stock_price"]
    },
    {
        "query": "Analyze my portfolio: 50% AAPL, 30% TSLA, 20% bonds",
        "category": "portfolio_analysis", 
        "should_use_tools": True,
        "expected_tools": ["analyze_portfolio"]
    },
    {
        "query": "Is this transaction suspicious: $5000 transfer at midnight?",
        "category": "fraud_detection",
        "should_use_tools": True,
        "expected_tools": ["detect_fraud"]
    },
    {
        "query": "Hello, how are you?",
        "category": "greeting",
        "should_use_tools": False,
        "expected_tools": []
    },
    {
        "query": "Thank you for your help!",
        "category": "gratitude",
        "should_use_tools": False, 
        "expected_tools": []
    }
]

# Sample portfolio data
SAMPLE_PORTFOLIO = {
    "user": "John Doe",
    "risk_tolerance": "moderate",
    "portfolio": {
        "AAPL": 100,
        "TSLA": 50,
        "bonds": 200
    },
    "total_value": 25000.00
}

# Sample fraud scenarios
FRAUD_TEST_CASES = [
    {
        "transaction": {
            "amount": 5000,
            "time": "2025-01-01T23:45:00Z",
            "recipient": "unknown_account",
            "type": "transfer"
        },
        "expected_suspicious": True
    },
    {
        "transaction": {
            "amount": 50,
            "time": "2025-01-01T14:30:00Z", 
            "recipient": "known_merchant",
            "type": "purchase"
        },
        "expected_suspicious": False
    }
]


# ============================================================================
# PYTEST MARKERS
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower, real dependencies)"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )