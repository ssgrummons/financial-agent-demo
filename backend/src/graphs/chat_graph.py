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
    ):
        # Heavy‑weight objects must exist *before* the base class builds the graph.
        self.message_strategy = SingleImageStrategy()
        self.tools = FinanceTools().load_tools()
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
        graph.add_node("tools", ToolNode(tools=self.tools))

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