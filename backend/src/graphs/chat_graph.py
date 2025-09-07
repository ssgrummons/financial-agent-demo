import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Dict, Any

from graphs.base_graph import BaseGraph
from components.assistants import MultiModalAsssitant
from components.message_strategies import SingleImageStrategy
from components.tools import FinanceTools
from models.states import ChatState  # <-- Pydantic model

# ------------------------------------------------------------------
# 1.  Configure logger (optional but handy)
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)

class ChatGraph(BaseGraph[ChatState]):
    """
    Graph that validates, converts TIFF → PNG, runs the assistant,
    extracts Markdown, and saves the result.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        verbose: bool,
        logprobs: bool,
        reasoning_effort: str,
        max_tokens: int,
        # Optional dependency injection parameters
        message_strategy=None,
        tools=None,
        assistant=None,
        finance_tools_factory=None,
    ):
        # Allow injection of message strategy, with fallback to default
        self.message_strategy = message_strategy or SingleImageStrategy()
        
        # Allow injection of tools, with fallback to default factory
        if tools is not None:
            self.tools = tools
        elif finance_tools_factory is not None:
            self.tools = finance_tools_factory().load_tools()
        else:
            self.tools = FinanceTools().load_tools()
        
        # Allow injection of complete assistant, with fallback to construction
        if assistant is not None:
            self.assistant = assistant
        else:
            self.assistant = MultiModalAsssitant(
                message_strategy=self.message_strategy,
                provider=provider,
                model=model,
                tools=self.tools,
                verbose=verbose,
                logprobs=logprobs,
                streaming=True,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
            )
        
        super().__init__(state_schema=ChatState)

    # ------------------------------------------------------------------
    # 2.  Build the graph
    # ------------------------------------------------------------------
    def build_graph(self) -> StateGraph[ChatState]:
        graph = StateGraph(state_schema=self.state_schema)
        # Attach the common nodes (validate_input & save_results)
        self.add_common_nodes(graph)

        # ---- Node definitions -----------------------------------------
        graph.add_node("assistant", self.assistant)
        graph.add_node("tools", self.execute_tools)

        # ---- Flow wiring -----------------------------------------------
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges("assistant", self.should_continue, {
            "tools":"tools",
            END:END
        })
        graph.add_edge("tools", "assistant")

        return graph

    def should_continue(self, state: ChatState, config: Dict[str, Any] = None) -> str:
        """Determine if we should continue to tools or end."""
        from langgraph.graph import END  # Import here to avoid circular imports
        
        logger.debug("Routing: Checking if should continue to tools or end")
        
        messages = state["messages"]
        if not messages:
            logger.debug("No messages, routing to END")
            return END
            
        last_message = messages[-1]
        logger.debug(f"Last message type: {type(last_message).__name__}")
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_count = len(last_message.tool_calls)
            logger.info(f"Routing: Found {tool_count} tool calls, routing to tools")
            for i, tool_call in enumerate(last_message.tool_calls):
                logger.debug(f"  Tool {i+1}: {tool_call.get('name', 'unknown')}")
            return "tools"
        
        logger.debug("No tool calls found, routing to END")
        return END
    
    def execute_tools(self, state: ChatState) -> ChatState:
        """Wrapper around ToolNode to ensure proper state handling"""
        # Use the built-in ToolNode but ensure proper state return
        tool_node = ToolNode(tools=self.tools)
        result = tool_node.invoke(state)
        
        # Ensure we return the full state, not just messages
        if isinstance(result, dict) and "messages" in result:
            # Get existing messages and create a new list
            existing_messages = list(state["messages"])  # Copy existing messages
            
            # Extend with new tool messages (don't append the whole result dict)
            existing_messages.extend(result["messages"])  # Add tool results
            
            # Return updated state with all messages
            return {**state, "messages": existing_messages}
        
        return result