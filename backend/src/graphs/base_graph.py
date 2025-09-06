from abc import ABC, abstractmethod
from langgraph.graph import StateGraph
from typing import Type, TypeVar, Generic, Optional, AsyncGenerator, Generator, Any
from components.shared_nodes import SharedNodes
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver

S = TypeVar("S", bound=BaseModel)

class BaseGraph(Generic[S], ABC):
    """
    Base class for all document‑processing graphs.

    Parameters
    ----------
    state_schema : Type[S]
        Pydantic model that describes the shape of the state that will
        flow through the graph.
    """
    def __init__(self, state_schema: Type[S]):
        self.state_schema = state_schema
        # Shared helper nodes
        self.shared_nodes = SharedNodes()
        # The graph instance will be created by the subclass
        self.graph: StateGraph = self.build_graph()
        # Cache the compiled graph
        self._compiled_graph = None
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Sub‑classes must implement this to construct the graph.

        Returns
        -------
        StateGraph
            A fully wired graph that can be compiled.
        """
        pass

    def create_compiled_graph(self):
        """
        Compile the graph after it has been built.
        """
        if self._compiled_graph is None:
            memory = InMemorySaver()
            self._compiled_graph = self.graph.compile(checkpointer=memory)
        return self._compiled_graph

    def add_common_nodes(self, graph: StateGraph) -> StateGraph:
        """
        Attach nodes that every graph should expose.

        Parameters
        ----------
        graph : StateGraph
            The graph instance to which we add the common nodes.

        Returns
        -------
        StateGraph
            The graph instance with the common nodes added.
        """
        graph.add_node("validate_input", self.shared_nodes.validate_input)
        graph.add_node("load_document_content", self.shared_nodes.load_document_content)
        graph.add_node("log_progress", self.shared_nodes.log_progress)
        return graph
    
    # Delegate common LangGraph methods to the compiled graph
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the compiled graph."""
        if name in ['astream', 'stream', 'invoke', 'ainvoke', 'batch', 'abatch']:
            compiled_graph = self.create_compiled_graph()
            return getattr(compiled_graph, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    # Explicit streaming methods for better IDE support and documentation
    async def astream(self, inputs: dict, config: Optional[dict] = None) -> AsyncGenerator[dict, None]:
        """Stream responses asynchronously from the compiled graph."""
        compiled_graph = self.create_compiled_graph()
        async for chunk in compiled_graph.astream(inputs, config=config):
            yield chunk
    
    def stream(self, inputs: dict, config: Optional[dict] = None) -> Generator[dict, None, None]:
        """Stream responses synchronously from the compiled graph."""
        compiled_graph = self.create_compiled_graph()
        for chunk in compiled_graph.stream(inputs, config=config):
            yield chunk
    
    async def ainvoke(self, inputs: dict, config: Optional[dict] = None) -> dict:
        """Invoke the graph asynchronously."""
        compiled_graph = self.create_compiled_graph()
        return await compiled_graph.ainvoke(inputs, config=config)
    
    def invoke(self, inputs: dict, config: Optional[dict] = None) -> dict:
        """Invoke the graph synchronously."""
        compiled_graph = self.create_compiled_graph()
        return compiled_graph.invoke(inputs, config=config)