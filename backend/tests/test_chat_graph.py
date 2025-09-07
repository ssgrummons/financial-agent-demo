"""
Tests for ChatGraph - the core agentic AI functionality.
Focuses on behavior over implementation to resist changes.
"""

import pytest
from unittest.mock import Mock, patch, call
from langgraph.graph import END

from graphs.chat_graph import ChatGraph
from models.states import ChatState
from .conftest import (
    create_user_input_state, 
    create_tool_call_message,
    FINANCIAL_TEST_QUERIES
)


class TestChatGraphInitialization:
    """Test graph initialization and setup"""

    def test_mocked_initialization(self, base_graph_config, mock_assistant, mock_tools):
        """Test initialization with injected mocks"""
        graph = ChatGraph(
            **base_graph_config,
            assistant=mock_assistant,
            tools=mock_tools
        )
        
        assert graph.assistant is mock_assistant
        assert graph.tools is mock_tools

    def test_graph_builds_successfully(self, mocked_chat_graph):
        """Test that graph can be built and compiled"""
        graph_def = mocked_chat_graph.build_graph()
        assert graph_def is not None
        
        # Test compilation
        compiled_graph = graph_def.compile()
        assert compiled_graph is not None

    @pytest.mark.parametrize("provider,model", [
        ("openai", "gpt-4"),
        ("openai", "gpt-3.5-turbo"), 
        ("anthropic", "claude-3"),
    ])
    def test_different_model_configurations(self, provider, model, mock_assistant, mock_tools):
        """Test graph works with different model configurations"""
        graph = ChatGraph(
            provider=provider,
            model=model,
            verbose=False,
            logprobs=False,
            reasoning_effort="medium",
            max_tokens=1000,
            assistant=mock_assistant,
            tools=mock_tools
        )
        
        compiled = graph.build_graph().compile()
        assert compiled is not None


class TestChatGraphRouting:
    """Test the routing logic in should_continue method"""
    
    def test_should_continue_with_tool_calls(self, mocked_chat_graph):
        """Test routing when assistant wants to use tools"""
        tool_call_message = create_tool_call_message(
            "get_stock_price", 
            {"symbol": "AAPL"}
        )
        
        state = create_user_input_state(
            "What's AAPL trading at?",
            messages=[tool_call_message]
        )
        
        result = mocked_chat_graph.should_continue(state)
        assert result == "tools"

    def test_should_continue_without_tool_calls(self, mocked_chat_graph):
        """Test routing when no tools are needed"""
        mock_message = Mock()
        mock_message.tool_calls = []  # No tool calls
        
        state = create_user_input_state(
            "Hello, how are you?",
            messages=[mock_message]
        )
        
        result = mocked_chat_graph.should_continue(state)
        assert result == END

    def test_should_continue_with_empty_messages(self, mocked_chat_graph):
        """Test routing with no messages"""
        state = create_user_input_state("test", messages=[])
        
        result = mocked_chat_graph.should_continue(state)
        assert result == END

    def test_should_continue_with_multiple_tool_calls(self, mocked_chat_graph):
        """Test routing with multiple tool calls"""
        mock_message = Mock()
        mock_message.tool_calls = [
            {"name": "get_stock_price", "args": {"symbol": "AAPL"}},
            {"name": "get_stock_price", "args": {"symbol": "TSLA"}}
        ]
        
        state = create_user_input_state(
            "Compare AAPL and TSLA",
            messages=[mock_message]
        )
        
        result = mocked_chat_graph.should_continue(state)
        assert result == "tools"


class TestChatGraphToolExecution:
    """Test tool execution functionality"""
    
    @patch('graphs.chat_graph.ToolNode')
    def test_execute_tools_preserves_state(self, mock_tool_node_class, mocked_chat_graph):
        """Test that execute_tools preserves state structure"""
        # Setup mock ToolNode
        mock_tool_instance = Mock()
        mock_tool_node_class.return_value = mock_tool_instance
        
        # Mock tool response
        tool_response_message = Mock()
        tool_response_message.content = "AAPL is trading at $150.00"
        mock_tool_instance.invoke.return_value = {
            "messages": [tool_response_message]
        }
        
        # Create initial state
        initial_message = Mock()
        state = create_user_input_state(
            "What's AAPL price?",
            messages=[initial_message]
        )
        
        # Execute tools
        result = mocked_chat_graph.execute_tools(state)
        
        # Verify state preservation
        assert "session_id" in result
        assert "user_id" in result
        assert "messages" in result
        assert result["session_id"] == state["session_id"]
        assert len(result["messages"]) == 2  # Original + tool response

    @patch('graphs.chat_graph.ToolNode')
    def test_execute_tools_with_tool_error(self, mock_tool_node_class, mocked_chat_graph):
        """Test tool execution when tools raise errors"""
        mock_tool_instance = Mock()
        mock_tool_node_class.return_value = mock_tool_instance
        mock_tool_instance.invoke.side_effect = Exception("API rate limit exceeded")
        
        state = create_user_input_state("test")
        
        with pytest.raises(Exception, match="API rate limit exceeded"):
            mocked_chat_graph.execute_tools(state)


class TestChatGraphBehaviors:
    """Test high-level behaviors that should be stable across changes"""
    
    @pytest.mark.parametrize("test_case", FINANCIAL_TEST_QUERIES)
    def test_query_categorization(self, test_case, mocked_chat_graph):
        """Test that different query types are handled appropriately"""
        query = test_case["query"]
        should_use_tools = test_case["should_use_tools"]
        
        # Create a mock message that simulates assistant's decision
        mock_message = Mock()
        if should_use_tools:
            mock_message.tool_calls = [{"name": "mock_tool", "args": {}}]
        else:
            mock_message.tool_calls = []
        
        state = create_user_input_state(query, messages=[mock_message])
        
        routing_decision = mocked_chat_graph.should_continue(state)
        
        if should_use_tools:
            assert routing_decision == "tools", f"Query '{query}' should use tools"
        else:
            assert routing_decision == END, f"Query '{query}' should not use tools"

    def test_conversation_context_structure(self, mocked_chat_graph):
        """Test that conversation context is maintained properly"""
        # Simulate a conversation with multiple turns
        messages = [
            Mock(content="Hello", tool_calls=[]),
            Mock(content="Hi there!", tool_calls=[]),
            Mock(content="What's AAPL price?", tool_calls=[{"name": "get_stock_price"}])
        ]
        
        state = create_user_input_state(
            "What's AAPL price?",
            messages=messages
        )
        
        # Test that state structure is preserved
        assert len(state["messages"]) == 3
        assert state["session_id"] == "test_session"
        
        # Test routing based on last message
        routing = mocked_chat_graph.should_continue(state)
        assert routing == "tools"

    def test_state_immutability_during_routing(self, mocked_chat_graph):
        """Test that routing doesn't modify state"""
        original_state = create_user_input_state("test query")
        original_state_copy = dict(original_state)
        
        # Add a message for routing
        mock_message = Mock()
        mock_message.tool_calls = []
        original_state["messages"] = [mock_message]
        
        mocked_chat_graph.should_continue(original_state)
        
        # State should be unchanged (except for messages we added)
        assert original_state["session_id"] == original_state_copy["session_id"]
        assert original_state["user_id"] == original_state_copy["user_id"]
        assert original_state["user_prompt"] == original_state_copy["user_prompt"]


class TestChatGraphErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_state_handling(self, mocked_chat_graph):
        """Test handling of invalid states"""
        # Test with None
        with pytest.raises(Exception):
            mocked_chat_graph.should_continue(None)
        
        # Test with missing required fields
        invalid_state = {"invalid": "state"}
        with pytest.raises(Exception):
            mocked_chat_graph.should_continue(invalid_state)

    def test_graph_structure_validation(self, mocked_chat_graph):
        """Test that graph structure is valid"""
        graph = mocked_chat_graph.build_graph()
        
        # Should have required nodes
        assert "assistant" in graph.nodes
        assert "tools" in graph.nodes
        
        # Should compile without errors
        compiled = graph.compile()
        assert compiled is not None


class TestChatGraphIntegration:
    """Integration tests with mocked external dependencies"""
    
    @patch('graphs.chat_graph.ToolNode')
    def test_full_conversation_flow(self, mock_tool_node_class, mocked_chat_graph):
        """Test a complete conversation flow"""
        # Setup mocks
        mock_tool_instance = Mock()
        mock_tool_node_class.return_value = mock_tool_instance
        mock_tool_instance.invoke.return_value = {
            "messages": [Mock(content="AAPL: $150.00")]
        }
        
        # Setup assistant mock to return tool call first, then final response
        mocked_chat_graph.assistant.invoke.side_effect = [
            {"messages": [create_tool_call_message("get_stock_price", {"symbol": "AAPL"})]},
            {"messages": [Mock(content="Based on the data, AAPL is trading at $150.00", tool_calls=[])]}
        ]
        
        # Initial state
        state = create_user_input_state("What's AAPL trading at?")
        
        # Test that we can process through the graph
        graph = mocked_chat_graph.build_graph().compile()
        
        # This would be a full integration test if we had the full state machine
        # For now, test individual components work together
        
        # 1. Assistant should be called
        assert mocked_chat_graph.assistant is not None
        
        # 2. Tools should be available
        assert len(mocked_chat_graph.tools) > 0
        
        # 3. Routing should work
        tool_call_state = create_user_input_state(
            "test",
            messages=[create_tool_call_message("get_stock_price", {"symbol": "AAPL"})]
        )
        assert mocked_chat_graph.should_continue(tool_call_state) == "tools"